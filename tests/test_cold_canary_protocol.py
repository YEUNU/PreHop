import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import cold_canary_fixture as fixture
from scripts import paper_cold_canary as executor


@pytest.fixture(autouse=True)
def restore_environment():
    original = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original)


@pytest.mark.parametrize('dataset', fixture.DATASETS)
def test_fixed_fixture_uses_production_manifest_validation(tmp_path, dataset):
    from cli.index import _load_corpus_manifest, _validate_staged_snapshot
    corpus, manifest, row = fixture.stage_fixture(tmp_path / dataset, dataset)
    loaded = _load_corpus_manifest(corpus)
    ids = _validate_staged_snapshot(sorted(path.name for path in corpus.glob('*.txt')), loaded, corpus)
    assert len(ids) == manifest['paragraph_count'] == 2
    fixture.validate_fixture(corpus, dataset, row, fixture.fixture_identity())
    assert hashlib.sha256(fixture.FIXTURE_PATH.read_bytes()).hexdigest() == fixture.FIXTURE_SHA256
    with pytest.raises(FileExistsError):
        fixture.stage_fixture(tmp_path / dataset, dataset)


def test_dataset_aliases_share_exact_sources_and_question(tmp_path):
    staged = [fixture.stage_fixture(tmp_path / dataset, dataset) for dataset in fixture.DATASETS]
    assert {path.name: path.read_bytes() for path in staged[0][0].glob('*.txt')} == {
        path.name: path.read_bytes() for path in staged[1][0].glob('*.txt')}
    assert staged[0][2]['query'] == staged[1][2]['query']
    assert staged[0][2]['dataset'] != staged[1][2]['dataset']


@pytest.mark.parametrize('mutation', ['source', 'query', 'identity', 'extra_source'])
def test_fixed_fixture_rejects_content_changes(tmp_path, mutation):
    corpus, _, row = fixture.stage_fixture(tmp_path / 'fixture', 'musique')
    identity = fixture.fixture_identity()
    if mutation == 'source':
        next(corpus.glob('*.txt')).write_text('altered source')
    elif mutation == 'query':
        row['query'] += '?'
    elif mutation == 'identity':
        identity['sha256'] = '0' * 64
    else:
        (corpus / 'extra.txt').write_text('third source')
    with pytest.raises(RuntimeError):
        fixture.validate_fixture(corpus, 'musique', row, identity)


def test_subprocess_import_and_help_do_not_execute_workflow():
    for args in [ ['-c', 'import scripts.paper_cold_canary'], ['scripts/paper_cold_canary.py', '--help'] ]:
        result = subprocess.run([sys.executable, *args], cwd=fixture.ROOT, capture_output=True, text=True, timeout=15, check=False)
        assert result.returncode == 0, result.stderr
        assert 'cold_native_index_start' not in result.stdout


def test_executor_missing_prerequisite_stops_before_staging(tmp_path, monkeypatch):
    from scripts import paper_gate_ledger as gate
    monkeypatch.setattr(executor, 'ROOT', Path.cwd())
    monkeypatch.setattr(gate, 'ready', lambda *args: (_ for _ in ()).throw(RuntimeError('prerequisite missing')))
    # No endpoint/native constructor is installed: an early gate is essential.
    with pytest.raises(RuntimeError, match='prerequisite missing'):
        asyncio.run(executor.workflow('no-live-test', 'naive', 'musique', 'new'))


def test_cold_artifacts_cannot_supply_real_one_query_evidence(tmp_path, monkeypatch):
    from core import admission, paper_policy, runtime_requirements
    from scripts import check_paper_runtime, verify_index_policy
    from scripts import paper_gate_ledger as gate
    corpus, manifest, row = fixture.stage_fixture(tmp_path / 'fixture', 'musique')
    record_path = tmp_path / 'query_record.json'
    record_path.write_text(json.dumps(row))
    index_path = tmp_path / 'index.json'
    index_path.write_text('{}')
    ref = lambda path: {'path': str(path.relative_to(tmp_path)), 'sha256': admission.sha256_file(path)}
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    monkeypatch.setattr(paper_policy, 'configure_target_environment', lambda *args: None)
    monkeypatch.setattr(verify_index_policy, 'verify', lambda *args: None)
    monkeypatch.setattr(check_paper_runtime, 'check', lambda *args: None)
    monkeypatch.setattr(gate, 'runtime_identity', lambda *args: {})
    monkeypatch.setattr(runtime_requirements, 'runtime_identity', lambda *args: {})
    monkeypatch.setattr(admission, 'current_post_query_inventory', lambda *args: {})
    query = {'query_record': ref(record_path), 'query': row['query'], 'query_id': row['_id'],
             'query_records_sha256': admission.sha256_file(record_path), 'runtime_identity': {},
             'post_query_artifact_inventory': {}, 'index_stats_sha256': admission.sha256_file(index_path)}
    index = {'run_id': 'test', 'source_manifest': ref(corpus / 'corpus_manifest.json'),
             'corpus_manifest_fingerprint': manifest['fingerprint'], 'cold_fixture': fixture.fixture_identity()}
    with pytest.raises(RuntimeError, match='Synthetic cold fixtures'):
        gate._validate_canary_artifacts('one_query_matrix_16', 'naive', 'musique', index_path, query, index)


def test_subprocess_main_loads_project_environment_before_workflow(tmp_path):
    # The real shared dotenv loader reads only this temporary non-secret fixture.
    (tmp_path / '.env').write_text('CANARY_LOADER_TEST=loaded\n')
    script = '''
import asyncio, os, sys
from pathlib import Path
from scripts import check_paper_runtime, paper_cold_canary
check_paper_runtime.ROOT = Path(sys.argv[1])
os.environ.pop('RAG_SKIP_PROJECT_ENV', None)
async def observed(*args):
    assert os.environ['CANARY_LOADER_TEST'] == 'loaded'
    assert asyncio.get_running_loop().is_running()
    print('loader-before-workflow')
paper_cold_canary.workflow = observed
sys.argv = ['paper_cold_canary.py', 'campaign', 'naive', 'musique', '--attempt', 'new']
paper_cold_canary.main()
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], cwd=fixture.ROOT,
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'loader-before-workflow'


@pytest.mark.parametrize('collision', ['nodes', 'snapshot', 'schema'])
def test_fresh_namespace_rejects_each_existing_database_artifact(monkeypatch, collision):
    from core import index_namespace, neo4j_service
    seen_loops = []
    class ReadOnlyService:
        async def execute_query(self, query, parameters=None):
            seen_loops.append(asyncio.get_running_loop())
            if query.startswith('SHOW INDEXES'):
                return [{'name': 'run-test-schema', 'labelsOrTypes': []}] if collision == 'schema' else []
            if 'RAGIndexSnapshot' in query:
                return [{'count': int(collision == 'snapshot')}]
            return [{'label': 'run-test'}] if collision == 'nodes' else []
    monkeypatch.setattr(neo4j_service, 'Neo4jService', ReadOnlyService)
    monkeypatch.setattr(index_namespace, 'index_namespace', lambda dataset: 'run-test')
    with pytest.raises(RuntimeError, match='already has'):
        asyncio.run(executor.ensure_fresh_namespace('naive', 'musique'))
    assert len(seen_loops) == 3 and len(set(seen_loops)) == 1
