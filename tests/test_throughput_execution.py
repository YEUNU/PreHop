"""Execution-profile resolution and cost semantics."""
import json
import subprocess
import sys

from core.amortized_cost import query_cost
from core.execution_profile import execution_profile


def test_normalized_cost_is_inverse_throughput_not_mean_latency():
    cost = query_cost(20.0, 100, complete=True)
    assert cost['seconds_per_unit'] == .2
    assert cost['units_per_second'] == 5
    assert cost['continuous_run_eligible']
    for kwargs in ({'complete': False}, {'complete': True, 'resumed': True}):
        assert query_cost(20, 100, **kwargs)['seconds_per_unit'] is None
    assert query_cost(float('nan'), 100, complete=True)['continuous_run_eligible'] is False
    assert query_cost(10, 0, complete=True)['seconds_per_unit'] is None


def test_profile_is_content_bound_and_registry_cli_uses_it(tmp_path, monkeypatch):
    path = tmp_path / 'profile.json'
    value = {'version': 1, 'name': 'test', 'settings': {'generation_concurrency': 7,
             'embedding_batch_size': 16, 'embedding_concurrency': 2, 'benchmark_concurrency': 4}}
    path.write_text(json.dumps(value))
    monkeypatch.setenv('RAG_EXECUTION_PROFILE', str(path))
    digest = execution_profile()['sha256']
    path.write_text(json.dumps(value, indent=4))
    assert execution_profile()['sha256'] == digest
    result = subprocess.run([sys.executable, 'core/strategy_registry.py', '--paper-defaults-tsv'],
                            check=True, capture_output=True, text=True)
    assert 'RAG_BENCHMARK_CONCURRENCY\t4' in result.stdout
    value['settings']['benchmark_concurrency'] = True
    path.write_text(json.dumps(value))
    assert execution_profile()['settings'] == value['settings']


def test_direct_profile_applies_client_limits(monkeypatch):
    from pathlib import Path

    from core.execution_profile import apply_execution_profile

    path = Path('configs/execution_profiles/direct-8.json').resolve()
    monkeypatch.setenv('RAG_EXECUTION_PROFILE', str(path))
    env = {'RAG_GENERATION_CONCURRENCY': '120', 'RAG_BENCHMARK_CONCURRENCY': '1'}
    apply_execution_profile(env)
    assert env['RAG_GENERATION_CONCURRENCY'] == env['RAG_BENCHMARK_CONCURRENCY'] == '8'
    assert env['RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS'] == '1'
    result = subprocess.run([sys.executable, '-c', "from core.strategy_registry import PAPER_TRANSPORT; assert PAPER_TRANSPORT.generation_concurrency == PAPER_TRANSPORT.benchmark_concurrency == 8; assert PAPER_TRANSPORT.embedding_concurrency == 1"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
