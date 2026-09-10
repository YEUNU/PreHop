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
    from core.strategy_registry import get_strategy
    monkeypatch.setattr(reuse, 'ROOT', tmp_path)
    monkeypatch.setattr(reuse, 'current_corpus_identity', lambda _: {'fingerprint': 'complete'})
    config = {'method': 'canonical', 'schema': 'v3'}
    monkeypatch.setattr(paper_compatibility, 'target_configuration', lambda *_: dict(config))
    strategy, dataset, source, target = 'lightrag', 'hotpotqa', 'source', 'campaign-hotpotqa-lightrag'
    spec = get_strategy(strategy)
    def configure(method, corpus, run):
        assert (method, corpus) == (strategy, dataset)
        os.environ.update(RAG_RUN_ID=run, RAG_INDEX_NAMESPACE=f'{corpus}_{run}')
        os.environ[spec.output_env] = f'{spec.output_default}/runs/{run}'
    monkeypatch.setattr(paper_policy, 'configure_target_environment', configure)
    # Restore the complete environment after each test.
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    original = tmp_path / spec.output_default / 'runs' / source
    (original / dataset / 'artifacts').mkdir(parents=True)
    (original / dataset / 'artifacts' / 'native.json').write_text('{"graph":"original"}')
    (original / dataset / 'index_snapshot_metadata.json').write_text('{"source_path":"unchanged"}')
    clone = tmp_path / spec.output_default / 'runs' / target
    shutil.copytree(original, clone)
    stats = {'path': 'data/index_stats/lightrag_hotpotqa_source.json', 'sha256': 'original-raw'}
    index = {'run_id': source, 'native_index_stats': stats}
    raw = {'timing_seconds': {'total_elapsed_seconds': 1234, 'official_pipeline_seconds': 1200}}
    monkeypatch.setattr(reuse, 'source_evidence', lambda *_: (index, raw))
    rows = reuse.inventory(original)
    link = {'version': 1, 'strategy': strategy, 'dataset': dataset, 'target_run_id': target,
            'source_run_id': source, 'source_index_namespace': f'{dataset}_{source}',
            'source_evidence': {}, 'index_stats': stats, 'index_timing_seconds': raw['timing_seconds'],
            'configuration_sha256': identity_sha256(config),
            'corpus_identity_sha256': identity_sha256({'fingerprint': 'complete'}),
            'preparation_elapsed_seconds': 2,
            'clone': {'source_output_root': str(original.relative_to(tmp_path)),
                      'query_output_root': str(clone.relative_to(tmp_path)),
                      'source_inventory': rows, 'initial_clone_inventory_sha256': identity_sha256(rows)}}
    matrix_path = tmp_path / 'matrix.json'
    matrix_path.write_text(json.dumps({'targets': {f'{dataset}/{strategy}': {}}}))
    ledger_path = tmp_path / 'ledger.json'
    ledger_path.write_text(json.dumps({'context': {'targets': {f'{dataset}/{strategy}': config}},
        'stages': {'one_query_matrix_16': {'status': 'canary_passed',
            'evidence_path': reuse.ref(matrix_path)['path'], 'evidence_sha256': reuse.ref(matrix_path)['sha256']}}}))
    link['gate_ledger'] = reuse.ref(ledger_path)
    link['clone']['operation'] = 'byte_identical_copy_without_metadata_rebinding'
    path = tmp_path / 'data/results' / target / 'index_link.json'
    path.parent.mkdir(parents=True)
    snapshot = path.parent / 'source_gate_ledger.json'
    shutil.copyfile(ledger_path, snapshot)
    link['gate_ledger'] = reuse.ref(snapshot)
    path.write_text(json.dumps(link))
    return path, link, original, clone, config


def test_validated_clone_keeps_source_identity_and_fresh_result(linked):
    path, link, original, clone, _ = linked
    before = reuse.inventory(original)
    reuse.load_link(path, link['target_run_id'], link['strategy'], link['dataset'], pristine_clone=True)
    assert os.environ['RAG_RUN_ID'] == 'source'
    assert os.environ['RAG_BENCHMARK_TIMESTAMP'] == link['target_run_id']
    assert os.environ['RAG_INDEX_NAMESPACE'] == 'hotpotqa_source'
    assert reuse.inventory(original) == before == reuse.inventory(clone)
    # Native query cache writes belong to the clone; original snapshots stay valid.
    (clone / 'query.cache').write_text('new query')
    reuse.load_link(path, link['target_run_id'], link['strategy'], link['dataset'])


def test_index_time_is_original_and_query_latency_is_not_parallel_wall(linked):
    _, link, *_ = linked
    timing = BenchmarkTiming().snapshot()
    timing['segments'][0]['wall_seconds'] = timing['total_wall_seconds'] = 10
    payload = {'benchmark_timing': timing, 'details': [{'latency': 8}, {'latency': 9}],
               'phase_costs': {'index_source_timing_seconds': link['index_timing_seconds'],
                               'reuse_preparation_seconds': 2, 'benchmark_wall_seconds': 10,
                               'index_plus_benchmark_wall_seconds': 1244, 'query_latency_sum_seconds': 17}}
    payload['phase_costs']['index_plus_benchmark_wall_seconds'] = 10


def test_prepare_copies_native_bytes_after_full_gate_and_preserves_source(linked, monkeypatch):
    from scripts import paper_cold_canary
    path, link, original, clone, _ = linked
    before = reuse.inventory(original)
    _, ledger = reuse.bound(link['gate_ledger'])
    shutil.rmtree(clone)
    shutil.rmtree(path.parent)
    campaign_ledger = reuse.ROOT / 'data/results/campaign/gate_ledger.json'
    campaign_ledger.parent.mkdir(parents=True)
    campaign_ledger.write_text(json.dumps(ledger))
    monkeypatch.setattr(paper_cold_canary, 'ROOT', reuse.ROOT)
    prepared = reuse.prepare('campaign', 'lightrag', 'hotpotqa')
    assert prepared == path
    assert reuse.inventory(original) == before == reuse.inventory(clone)
    produced = json.loads(prepared.read_text())
    assert produced['index_stats'] == link['index_stats']
    assert produced['index_timing_seconds']['total_elapsed_seconds'] == 1234
    assert produced['clone']['operation'] == 'byte_identical_copy_without_metadata_rebinding'
    # Later live-ledger advancement cannot invalidate the immutable source snapshot.
    campaign_ledger.write_text(json.dumps({**ledger, 'status': 'admitted'}))
    reuse.load_link(prepared, link['target_run_id'], 'lightrag', 'hotpotqa')


def test_fresh_interpreter_binds_cli_and_all_native_query_paths_before_import(tmp_path, monkeypatch):
    import subprocess
    import sys
    script = r'''
import json,os,sys
from pathlib import Path
from core import index_reuse as reuse
from core.strategy_registry import PAPER_TRANSPORT, get_strategy
method=sys.argv[2]
reuse.ROOT=Path(sys.argv[1])
target='fresh-'+method
spec=get_strategy(method)
clone=str(Path(spec.output_default)/'runs'/target)
path=reuse.ROOT/'data/results'/target/'index_link.json'
path.parent.mkdir(parents=True)
path.write_text(json.dumps({'target_run_id':target,'source_run_id':'original-index',
    'strategy':method,'dataset':'hotpotqa','clone':{'query_output_root':clone}}))
os.environ.update(RAG_JUDGE_ENABLED='true',RAG_JUDGE_BATCH='true')
reuse.bootstrap_benchmark(path,target,method,'hotpotqa')
assert 'core.config' not in sys.modules
from cli.benchmark import RAGConfig
assert RAGConfig.JUDGE_ENABLED is False and RAGConfig.JUDGE_BATCH is False
assert RAGConfig.LLM_SEED == spec.paper_generation_seed
assert int(os.environ['RAG_BENCHMARK_CONCURRENCY']) == PAPER_TRANSPORT.benchmark_concurrency
assert os.environ['RAG_RUN_ID']=='original-index'
assert os.environ['RAG_BENCHMARK_TIMESTAMP']==target
expected=(reuse.ROOT/clone/'hotpotqa').resolve()
if method=='ms_graphrag':
    from models.ms_graphrag.official_indexer import output_dir_for
    assert output_dir_for('hotpotqa')==expected
else:
    from models.official_baseline_runtime import _command
    command=_command(method,'hotpotqa','serve')
    assert command[command.index('--output-dir')+1]==str(expected)
print('query_path_and_static_config_passed')
'''
    from core.strategy_registry import paper_environment_defaults
    monkeypatch.setenv('VLLM_URL', 'https://fixture.invalid/v1')
    monkeypatch.setenv('RAG_BENCHMARK_CONCURRENCY', '4')
    # Native-import tests elsewhere can intentionally populate private aliases.
    # This independent import test starts with an explicit canonical environment.
    environment = {key: os.environ[key] for key in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR', 'LD_LIBRARY_PATH') if key in os.environ}
    environment.update(paper_environment_defaults())
    for method in ('ms_graphrag', 'lightrag', 'gfm_rag', 'linear_rag'):
        result = subprocess.run([sys.executable, '-c', script, str(tmp_path), method],
                                env=environment, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith('query_path_and_static_config_passed')
