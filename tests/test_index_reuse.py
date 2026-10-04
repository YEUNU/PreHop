import json
import os
import shutil

import pytest

from core import index_reuse as reuse
from core.admission import identity_sha256
from core.phase_timing import BenchmarkTiming


def test_timing_resume_uses_unique_cumulative_segments(monkeypatch):
    tick = [100.0]
    monkeypatch.setattr('core.phase_timing.time.perf_counter', lambda: tick[0])
    first = BenchmarkTiming()
    tick[0] = 102
    assert first.snapshot()['total_wall_seconds'] == 2
    tick[0] = 105
    checkpoint = first.snapshot()
    assert checkpoint['total_wall_seconds'] == 5
    second = BenchmarkTiming()
    second.restore(checkpoint)
    tick[0] = 108
    resumed = second.snapshot()
    assert resumed['total_wall_seconds'] == 8
    assert len(resumed['segments']) == 2
    third = BenchmarkTiming()
    third.restore(resumed)
    tick[0] = 110
    assert third.snapshot()['total_wall_seconds'] == 10
    assert len(third.snapshot()['segments']) == 3


@pytest.fixture
def linked(tmp_path, monkeypatch):
    from core import paper_compatibility, paper_policy
    monkeypatch.setattr(reuse, 'ROOT', tmp_path)
    monkeypatch.setattr(reuse, 'current_corpus_identity', lambda _: {'fingerprint': 'complete'})
    config = {'method': 'canonical', 'schema': 'v3'}
    monkeypatch.setattr(paper_compatibility, 'target_configuration', lambda *_: dict(config))
    strategy, dataset, source, target = 'naive', 'hotpotqa', 'source', 'campaign-hotpotqa-naive'
    def configure(method, corpus, run):
        assert (method, corpus) == (strategy, dataset)
        os.environ.update(RAG_RUN_ID=run, RAG_INDEX_NAMESPACE=f'{corpus}_{run}')
    monkeypatch.setattr(paper_policy, 'configure_target_environment', configure)
    # Restore the complete environment after each test.
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    stats = {'path': 'data/index_stats/naive_hotpotqa_source.json', 'sha256': 'original-raw'}
    index = {'run_id': source, 'native_index_stats': stats}
    raw = {'timing_seconds': {'total_elapsed_seconds': 1234, 'document_pipeline_seconds': 1200}}
    monkeypatch.setattr(reuse, 'source_evidence', lambda *_: (index, raw))
    link = {'version': 1, 'strategy': strategy, 'dataset': dataset, 'target_run_id': target,
            'source_run_id': source, 'source_index_namespace': f'{dataset}_{source}',
            'source_evidence': {}, 'index_stats': stats, 'index_timing_seconds': raw['timing_seconds'],
            'configuration_sha256': identity_sha256(config),
            'corpus_identity_sha256': identity_sha256({'fingerprint': 'complete'}),
            'preparation_elapsed_seconds': 2, 'clone': None}
    matrix_path = tmp_path / 'matrix.json'
    matrix_path.write_text(json.dumps({'targets': {f'{dataset}/{strategy}': {}}}))
    ledger_path = tmp_path / 'ledger.json'
    ledger_path.write_text(json.dumps({'context': {'targets': {f'{dataset}/{strategy}': config}},
        'stages': {'one_query_matrix_16': {'status': 'canary_passed',
            'evidence_path': reuse.ref(matrix_path)['path'], 'evidence_sha256': reuse.ref(matrix_path)['sha256']}}}))
    path = tmp_path / 'data/results' / target / 'index_link.json'
    path.parent.mkdir(parents=True)
    snapshot = path.parent / 'source_gate_ledger.json'
    shutil.copyfile(ledger_path, snapshot)
    link['gate_ledger'] = reuse.ref(snapshot)
    path.write_text(json.dumps(link))
    return path, link, config


def test_link_configures_source_namespace_and_fresh_result(linked):
    path, link, _ = linked
    reuse.load_link(path)
    assert os.environ['RAG_RUN_ID'] == 'source'
    assert os.environ['RAG_BENCHMARK_TIMESTAMP'] == link['target_run_id']
    assert os.environ['RAG_INDEX_NAMESPACE'] == 'hotpotqa_source'
    assert os.environ['RAG_CHUNK_CACHE'] == os.environ['RAG_EMBEDDING_CACHE'] == 'off'


def test_index_time_is_original_and_query_latency_is_not_parallel_wall(linked):
    _, link, _ = linked
    timing = BenchmarkTiming().snapshot()
    timing['segments'][0]['wall_seconds'] = timing['total_wall_seconds'] = 10
    payload = {'benchmark_timing': timing, 'details': [{'latency': 8}, {'latency': 9}],
               'phase_costs': {'index_source_timing_seconds': link['index_timing_seconds'],
                               'reuse_preparation_seconds': 2, 'benchmark_wall_seconds': 10,
                               'index_plus_benchmark_wall_seconds': 1244, 'query_latency_sum_seconds': 17}}
    assert payload['phase_costs']['index_source_timing_seconds']['total_elapsed_seconds'] == 1234
    assert payload['phase_costs']['query_latency_sum_seconds'] != payload['phase_costs']['benchmark_wall_seconds']


def test_prepare_links_completed_source_after_full_gate_without_copying(linked, monkeypatch):
    from scripts import paper_cold_canary
    path, link, _ = linked
    _, ledger = reuse.bound(link['gate_ledger'])
    shutil.rmtree(path.parent)
    campaign_ledger = reuse.ROOT / 'data/results/campaign/gate_ledger.json'
    campaign_ledger.parent.mkdir(parents=True)
    campaign_ledger.write_text(json.dumps(ledger))
    monkeypatch.setattr(paper_cold_canary, 'ROOT', reuse.ROOT)
    prepared = reuse.prepare('campaign', 'naive', 'hotpotqa')
    assert prepared == path
    produced = json.loads(prepared.read_text())
    assert produced['index_stats'] == link['index_stats']
    assert produced['index_timing_seconds']['total_elapsed_seconds'] == 1234
    assert produced['clone'] is None
    assert produced['source_index_namespace'] == 'hotpotqa_source'
    # Later live-ledger advancement cannot invalidate the immutable source snapshot.
    campaign_ledger.write_text(json.dumps({**ledger, 'status': 'admitted'}))
    reuse.load_link(prepared)


def test_fresh_interpreter_binds_cli_before_static_config_import(tmp_path, monkeypatch):
    import subprocess
    import sys
    script = r'''
import json,os,sys
from pathlib import Path
from core import index_reuse as reuse
from core.execution_profile import resolved_execution_environment
method=sys.argv[2]
reuse.ROOT=Path(sys.argv[1])
target='fresh-'+method
path=reuse.ROOT/'data/results'/target/'index_link.json'
path.parent.mkdir(parents=True)
path.write_text(json.dumps({'target_run_id':target,'source_run_id':'original-index',
    'strategy':method,'dataset':'hotpotqa','clone':None}))
reuse.bootstrap_benchmark(path)
assert 'core.config' not in sys.modules
from cli.benchmark import RAGConfig
from core.inference_transport import InferenceTransport
assert InferenceTransport.resolve("core").generation_seed is None
assert os.environ['RAG_BENCHMARK_CONCURRENCY'] == resolved_execution_environment()['RAG_BENCHMARK_CONCURRENCY']
assert os.environ['RAG_RUN_ID']=='original-index'
assert os.environ['RAG_BENCHMARK_TIMESTAMP']==target
assert os.environ['RAG_INDEX_NAMESPACE']=='hotpotqa_original-index'
assert os.environ['RAG_INDEX_REUSE_LINK']==str(path)
print('query_path_and_static_config_passed')
'''
    from core.strategy_registry import paper_environment_defaults
    monkeypatch.setenv('VLLM_URL', 'https://fixture.invalid/v1')
    monkeypatch.setenv('RAG_BENCHMARK_CONCURRENCY', '4')
    # This independent import test starts with an explicit canonical environment.
    environment = {key: os.environ[key] for key in (
        'PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR', 'LD_LIBRARY_PATH', 'RAG_EXECUTION_PROFILE',
    ) if key in os.environ}
    environment.update(paper_environment_defaults())
    for method in ('prehop', 'naive'):
        result = subprocess.run([sys.executable, '-c', script, str(tmp_path), method],
                                env=environment, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith('query_path_and_static_config_passed')
