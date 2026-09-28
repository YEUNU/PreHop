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


def test_profile_is_content_bound_and_registry_defaults_are_static(tmp_path, monkeypatch):
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
    assert 'RAG_BENCHMARK_CONCURRENCY\t1' in result.stdout
    value['settings']['benchmark_concurrency'] = True
    path.write_text(json.dumps(value))
    assert execution_profile()['settings'] == value['settings']


def test_direct_profile_applies_client_limits(monkeypatch):
    from pathlib import Path

    from core.execution_profile import apply_execution_profile

    path = Path('configs/execution_profiles/direct-8.json').resolve()
    monkeypatch.setenv('RAG_EXECUTION_PROFILE', str(path))
    env = {'RAG_EXECUTION_PROFILE': str(path), 'RAG_GENERATION_CONCURRENCY': '120', 'RAG_BENCHMARK_CONCURRENCY': '1'}
    apply_execution_profile(env)
    assert env['RAG_GENERATION_CONCURRENCY'] == env['RAG_BENCHMARK_CONCURRENCY'] == '8'
    assert env['RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS'] == '1'
    from core.inference_transport import InferenceTransport
    assert InferenceTransport.resolve('prehop').generation_concurrency == 8


def test_resolution_uses_supplied_environment_and_current_profile(tmp_path, monkeypatch):
    from core.execution_profile import resolved_execution_environment
    from core.inference_transport import InferenceTransport

    # Modules were imported before a profile was chosen. Neither a late
    # selection nor a copied child environment may depend on that import order.
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps({'settings': {'generation_concurrency': 3, 'embedding_batch_size': 5}}))
    monkeypatch.setenv('RAG_EXECUTION_PROFILE', str(path))
    assert InferenceTransport.resolve('prehop').generation_concurrency == 3
    independent = {'RAG_GENERATION_CONCURRENCY': '11'}
    assert resolved_execution_environment(independent)['RAG_GENERATION_CONCURRENCY'] == '11'
    assert InferenceTransport.resolve('prehop', independent).execution_profile['sha256'] is None
    assert independent == {'RAG_GENERATION_CONCURRENCY': '11'}
    selected = {**independent, 'RAG_EXECUTION_PROFILE': str(path)}
    transport = InferenceTransport.resolve('prehop', selected)
    assert transport.generation_concurrency == 3
    assert transport.embedding_batch_size == 5
    path.unlink()
    # Provenance belongs to the resolved transport, even if its file is removed.
    assert transport.policy_dict()['execution_profile']['settings']['generation_concurrency'] == 3
