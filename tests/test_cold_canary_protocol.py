import hashlib
import json
import os
import subprocess
import sys

import pytest

from scripts import cold_canary_fixture as fixture


@pytest.fixture(autouse=True)
def restore_environment():
    original = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original)


@pytest.mark.parametrize('dataset', fixture.DATASETS)
def test_fixed_fixture_uses_production_manifest_validation(tmp_path, dataset):
    from cli.index import _load_corpus_manifest, _source_ids_from_filenames
    corpus, manifest, row = fixture.stage_fixture(tmp_path / dataset, dataset)
    assert _load_corpus_manifest(corpus) == manifest
    ids = _source_ids_from_filenames(sorted(path.name for path in corpus.glob('*.txt')))
    assert len(ids) == manifest['paragraph_count'] == 2
    assert row == fixture.query_record(dataset)
    assert row['cold_fixture'] == fixture.fixture_identity()
    assert {path.name: path.read_bytes() for path in corpus.glob('*.txt')} == {
        document['filename']: document['text'].encode()
        for document in fixture.load_fixture()['documents']
    }
    assert hashlib.sha256(fixture.FIXTURE_PATH.read_bytes()).hexdigest() == fixture.FIXTURE_SHA256
    with pytest.raises(FileExistsError):
        fixture.stage_fixture(tmp_path / dataset, dataset)


def test_dataset_aliases_share_exact_sources_and_question(tmp_path):
    staged = [fixture.stage_fixture(tmp_path / dataset, dataset) for dataset in fixture.DATASETS]
    assert {path.name: path.read_bytes() for path in staged[0][0].glob('*.txt')} == {
        path.name: path.read_bytes() for path in staged[1][0].glob('*.txt')}
    assert staged[0][2]['query'] == staged[1][2]['query']
    assert staged[0][2]['dataset'] != staged[1][2]['dataset']


def test_subprocess_import_and_help_do_not_execute_workflow():
    for args in [ ['-c', 'import scripts.paper_cold_canary'], ['scripts/paper_cold_canary.py', '--help'] ]:
        result = subprocess.run([sys.executable, *args], cwd=fixture.ROOT, capture_output=True, text=True, timeout=15, check=False)
        assert result.returncode == 0, result.stderr
        assert 'cold_native_index_start' not in result.stdout


def test_subprocess_main_loads_project_environment_before_workflow(tmp_path):
    # The real shared dotenv loader reads only this temporary non-secret fixture.
    (tmp_path / '.env').write_text('CANARY_LOADER_TEST=loaded\n')
    script = '''
import asyncio, os, sys
from pathlib import Path
from scripts import runner_environment, paper_cold_canary
runner_environment.ROOT = Path(sys.argv[1])
os.environ.pop('RAG_SKIP_PROJECT_ENV', None)
async def observed(*args):
    assert os.environ['CANARY_LOADER_TEST'] == 'loaded'
    assert asyncio.get_running_loop().is_running()
    print('loader-before-workflow')
paper_cold_canary.workflow = observed
sys.argv = ['paper_cold_canary.py', 'campaign', 'naive', 'hotpotqa', '--attempt', 'new']
paper_cold_canary.main()
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], cwd=fixture.ROOT,
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'loader-before-workflow'


@pytest.mark.asyncio
async def test_small_corpus_workflow_records_answer_and_manifest_without_snapshot_probe(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from cli import index
    from core import paper_policy, runtime_requirements
    from core.neo4j_service import Neo4jService
    from models.prehop.graphrag import GraphRAG
    from scripts import paper_cold_canary as canary

    monkeypatch.setattr(canary, 'ROOT', tmp_path)
    stats_path = tmp_path / 'index-stats.json'

    def configure(*args):
        monkeypatch.setenv('RAG_INDEX_STATS_PATH', str(stats_path))

    async def build(corpus, *args):
        manifest = index._load_corpus_manifest(corpus)
        assert manifest['paragraph_count'] == 2
        stats_path.write_text(json.dumps({
            'status': 'complete', 'index_policy_sha256': 'fixture-policy',
            'official_stats': {'observed': True},
        }))

    answer = AsyncMock(return_value=('Fixture answer', [
        {'source': 'fixture.txt', 'doc': 'Fixture', 'text': 'Supporting passage.'},
    ], []))
    monkeypatch.setattr(paper_policy, 'configure_target_environment', configure)
    monkeypatch.setattr(index, 'run_indexing', build)
    monkeypatch.setattr(GraphRAG, 'run_workflow', answer)
    monkeypatch.setattr(GraphRAG, '__init__', lambda self, **kwargs: None)
    monkeypatch.setattr(runtime_requirements, 'runtime_identity', lambda _: {})
    monkeypatch.setattr(Neo4jService, 'global_close', AsyncMock())

    await canary.workflow('reviewer', 'prehop', 'multihoprag', 'a1')

    base = tmp_path / 'data/results/reviewer/cold_v2/a1/multihoprag/prehop'
    evidence = json.loads((base / 'index_evidence.json').read_text())
    query = json.loads((base / 'query.json').read_text())
    assert evidence['source_count'] == 2
    assert evidence['official_stats'] == {'observed': True}
    assert query['answer'] == 'Fixture answer'
    assert query['documents'][0]['text'] == 'Supporting passage.'
    assert query['active_index_snapshot'] == {'status': 'not_checked'}
    assert (base / 'evidence.json').is_file()
    answer.assert_awaited_once()
