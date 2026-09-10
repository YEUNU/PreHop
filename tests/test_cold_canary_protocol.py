import hashlib
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
    from cli.index import _load_corpus_manifest, _staged_source_ids
    corpus, manifest, row = fixture.stage_fixture(tmp_path / dataset, dataset)
    loaded = _load_corpus_manifest(corpus)
    ids = _staged_source_ids(sorted(path.name for path in corpus.glob('*.txt')), loaded, corpus)
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
    corpus, _, row = fixture.stage_fixture(tmp_path / 'fixture', 'hotpotqa')
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
        fixture.validate_fixture(corpus, 'hotpotqa', row, identity)


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
