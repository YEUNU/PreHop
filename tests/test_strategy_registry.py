
from core.inference_transport import InferenceTransport
from core.strategy_registry import ALL_STRATEGIES, BY_NAME, PRIMARY_STRATEGIES, get_strategy


def test_primary_matrix_and_legacy_admission_are_centralized():
    assert PRIMARY_STRATEGIES == ("prehop", "naive")
    assert ALL_STRATEGIES == PRIMARY_STRATEGIES
    assert set(BY_NAME) == set(ALL_STRATEGIES)
    assert not {"browsenet", "proprag", "youtu_graphrag"} & set(ALL_STRATEGIES)


def test_unknown_strategy_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="unknown strategy"):
        get_strategy("unregistered")


def test_strategy_embedding_override_is_not_a_public_input(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", "16")
    monkeypatch.setenv("RAG_PREHOP_EMBEDDING_BATCH_SIZE", "3")
    assert InferenceTransport.resolve("prehop").embedding_batch_size == 16
    assert InferenceTransport.resolve("naive").embedding_batch_size == 16


def test_transport_consumes_only_canonical_single_endpoint(monkeypatch):
    monkeypatch.setenv("RAG_INFERENCE_BASE_URL", "http://litellm/v1")
    monkeypatch.setenv("RAG_INFERENCE_API_KEY", "key")
    monkeypatch.setenv("RAG_GENERATION_MODEL", "generation")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "embedding")
    transport = InferenceTransport.resolve("prehop")
    assert transport.generation_base_url == transport.embedding_base_url == "http://litellm/v1"
    monkeypatch.delenv("RAG_INFERENCE_BASE_URL")
    monkeypatch.setenv("VLLM_URL", "http://legacy-generation/v1")
    monkeypatch.setenv("VLLM_EMBED_URL", "http://legacy-embedding/v1")
    assert InferenceTransport.resolve("prehop").generation_base_url == ""
