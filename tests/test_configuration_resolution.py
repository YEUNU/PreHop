"""Configured runtime values have one owner across entrypoints and provenance."""
import pytest

from core.inference_transport import InferenceTransport
from core.vllm_client import VLLMClient


@pytest.mark.parametrize('paper_mode', ['true', 'false'])
def test_client_uses_direct_gateway_despite_stale_proxy_environment(monkeypatch, paper_mode):
    for name, value in {
        'RAG_PAPER_MODE': paper_mode,
        'RAG_INFERENCE_BASE_URL': 'http://gateway.test/v1/',
        'RAG_INFERENCE_API_KEY': 'fixture-key',
        'RAG_GENERATION_MODEL': 'fixture-model',
        'RAG_EMBEDDING_MODEL': 'fixture-embedding',
        'RAG_INFERENCE_TIMEOUT': '19',
        'LLM_REQUEST_TIMEOUT': 'invalid-retired-setting',
        'RAG_INFERENCE_RETRY_ATTEMPTS': '4',
        'RAG_QUEUE_PROXY_URL': 'http://queue.test/v1/',
        'RAG_QUEUE_TOKEN': 'fixture-queue-token',
        'RAG_LLM_SEED': '42',
    }.items():
        monkeypatch.setenv(name, value)
    transport = InferenceTransport.resolve('core')
    client = VLLMClient()
    assert client.vllm_url == client.embed_url == transport.generation_base_url == 'http://gateway.test/v1'
    assert client.api_key == transport.api_key == 'fixture-key'
    assert client._request_timeout == transport.timeout_seconds == 19
    assert client._retry_attempts == transport.retry_attempts == 4
    assert transport.generation_seed == (None if paper_mode == 'true' else 42)


@pytest.mark.parametrize('paper_mode', ['true', 'false'])
def test_recorded_index_policy_uses_effective_method_settings(monkeypatch, paper_mode):
    from cli.index import _resolved_index_policy
    from core.strategy_registry import method_setting
    monkeypatch.setenv('RAG_PAPER_MODE', paper_mode)
    monkeypatch.setenv('RAG_LIGHTRAG_TOP_K', '17')
    monkeypatch.setenv('RAG_INFERENCE_TIMEOUT', '21')
    monkeypatch.setenv('RAG_GENERATION_MODEL', 'actual-model')
    monkeypatch.setenv('RAG_EMBEDDING_MODEL', 'actual-embedding')
    policy = _resolved_index_policy('lightrag', 'default', 'fixture')
    assert policy['retrieval_top_k'] == method_setting('lightrag', 'retrieval_top_k') == 17
    assert policy['indexing_model'] == 'actual-model'
    assert policy['embedding_model'] == 'actual-embedding'
    assert policy['operational_config']['timeout_seconds'] == 21


def test_fixed_upstream_model_settings_have_no_inert_overrides(monkeypatch):
    from core.strategy_registry import get_strategy, method_setting
    monkeypatch.setenv('RAG_LINEAR_RAG_MPNET_MODEL', 'unused-override')
    assert method_setting('linear_rag', 'official_embedding_model') == get_strategy('linear_rag').paper_embedding_model
    assert 'RAG_LINEAR_RAG_MPNET_MODEL' not in get_strategy('linear_rag').index_environment_defaults()


def test_isolated_worker_preserves_explicit_transport_overrides(monkeypatch):
    from models.official_baseline_runtime import _runtime_env
    from scripts.runner_environment import safe_environment

    monkeypatch.setenv('RAG_LIGHTRAG_EMBEDDING_BATCH_SIZE', '7')
    monkeypatch.setenv('RAG_LIGHTRAG_EMBEDDING_CONCURRENCY', '2')
    monkeypatch.setenv('RAG_LIGHTRAG_EMBEDDING_RETRY_ATTEMPTS', '3')
    monkeypatch.setenv('NEO4J_VECTOR_DIMENSIONS', '128')
    monkeypatch.setenv('MAX_EMBEDDING_LENGTH', '512')
    monkeypatch.setenv('RAG_UNRELATED_TEST_SETTING', 'discard')
    expected = InferenceTransport.resolve('lightrag')
    filtered = safe_environment()
    assert 'RAG_UNRELATED_TEST_SETTING' not in filtered
    # Follow the actual supervisor -> native worker boundary.
    monkeypatch.setattr('os.environ', filtered)
    native = _runtime_env('lightrag')
    observed = InferenceTransport.resolve('lightrag', native)
    for field in ('embedding_batch_size', 'embedding_concurrency', 'retry_attempts',
                  'embedding_dimensions', 'embedding_max_input_tokens'):
        assert getattr(observed, field) == getattr(expected, field)
    assert native['RAG_EMBEDDING_BATCH_SIZE'] == '7'
    assert native['RAG_EMBEDDING_CONCURRENCY'] == '2'


def test_core_query_settings_keep_types_normalization_and_fixed_parameters(monkeypatch):
    import runpy

    monkeypatch.setenv('RAG_GRAPH_HOP_DEPTH', '0')
    monkeypatch.setenv('RAG_GRAPH_PATH_DECAY', '0.25')
    monkeypatch.setenv('RAG_ABLATION_Q_PLUS', 'false')
    monkeypatch.setenv('RAG_GRAPH_EDGE_VARIANT', ' NEXT_ONLY ')
    monkeypatch.setenv('RAG_DEFAULT_TOP_K', '99')
    config = runpy.run_path('core/config.py')['RAGConfig']
    assert config.GRAPH_HOP_DEPTH == 0
    assert config.GRAPH_PATH_DECAY == 0.25
    assert config.ABLATION_Q_PLUS is False
    assert config.GRAPH_EDGE_VARIANT == 'next_only'
    assert config.DEFAULT_TOP_K == 12
    monkeypatch.setenv('RAG_ABLATION_Q_PLUS', 'not-a-bool')
    with pytest.raises(ValueError, match='RAG_ABLATION_Q_PLUS'):
        runpy.run_path('core/config.py')
