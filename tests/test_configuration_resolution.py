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
def test_recorded_index_policy_uses_effective_transport_settings(monkeypatch, paper_mode):
    from cli.index import _resolved_index_policy
    monkeypatch.setenv('RAG_PAPER_MODE', paper_mode)
    monkeypatch.setenv('RAG_INFERENCE_TIMEOUT', '21')
    monkeypatch.setenv('RAG_GENERATION_MODEL', 'actual-model')
    monkeypatch.setenv('RAG_EMBEDDING_MODEL', 'actual-embedding')
    policy = _resolved_index_policy('prehop', 'default')
    assert policy['indexing_model'] == 'actual-model'
    assert policy['embedding_model'] == 'actual-embedding'
    assert policy['operational_config']['timeout_seconds'] == 21
    assert policy['operational_config']['transport_profile'] == 'openai_compatible_litellm'


def test_core_query_settings_keep_types_normalization_and_fixed_parameters(monkeypatch):
    import runpy

    monkeypatch.setenv('RAG_GRAPH_HOP_DEPTH', '0')
    monkeypatch.setenv('RAG_GRAPH_PATH_DECAY', '0.25')
    monkeypatch.setenv('RAG_ABLATION_Q_MINUS', 'false')
    monkeypatch.setenv('RAG_ABLATION_Q_PLUS', 'false')
    monkeypatch.setenv('RAG_PRECOMPUTE_RECIPROCAL_HOPS', 'false')
    monkeypatch.setenv('RAG_GRAPH_EDGE_VARIANT', ' NEXT_ONLY ')
    monkeypatch.setenv('RAG_DEFAULT_TOP_K', '99')
    config = runpy.run_path('core/config.py')['RAGConfig']
    assert config.GRAPH_HOP_DEPTH == 1
    assert config.GRAPH_PATH_DECAY == 0.5
    assert config.ABLATION_Q_MINUS is True
    assert config.ABLATION_Q_PLUS is True
    assert config.PRECOMPUTE_RECIPROCAL_HOPS is True
    assert config.GRAPH_EDGE_VARIANT == 'next_only'
    assert config.DEFAULT_TOP_K == 12
    monkeypatch.setenv('RAG_GRAPH_EDGE_VARIANT', 'unsupported')
    with pytest.raises(ValueError, match='RAG_GRAPH_EDGE_VARIANT'):
        runpy.run_path('core/config.py')
