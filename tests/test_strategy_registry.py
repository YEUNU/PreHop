import pytest

from core.embedding_policy import EmbeddingOperationalConfig
from core.inference_transport import InferenceTransport
from core.strategy_registry import (
    ALL_STRATEGIES,
    BY_NAME,
    PRIMARY_STRATEGIES,
    RESEARCH_EXTERNAL_STRATEGIES,
)
from models.official_baseline_runtime import OFFICIAL_REPOSITORIES, OFFICIAL_REVISIONS, _command


def test_primary_matrix_and_legacy_admission_are_centralized():
    assert PRIMARY_STRATEGIES == (
        "prehop",
        "naive",
        "hoprag",
        "ms_graphrag",
        "lightrag",
        "hipporag2",
        "gfm_rag",
        "linear_rag",
    )
    assert "hoprag" in ALL_STRATEGIES
    assert not {"browsenet", "proprag"} & set(ALL_STRATEGIES)
    assert set(RESEARCH_EXTERNAL_STRATEGIES) == {"lightrag", "hipporag2", "gfm_rag", "linear_rag"}


def test_runtime_revision_repo_and_worker_are_registry_views(monkeypatch, tmp_path):
    for name, spec in BY_NAME.items():
        if spec.external:
            assert OFFICIAL_REVISIONS[name] == spec.revision
            assert OFFICIAL_REPOSITORIES[name] == spec.repository
    monkeypatch.setenv("RAG_LIGHTRAG_PYTHON", str(tmp_path / "python"))
    command = _command("lightrag", "corpus", "index")
    assert command[1].endswith("scripts/research_baseline_worker.py")


def test_license_boundaries_are_explicit():
    assert "GPL" in BY_NAME["linear_rag"].license_note


def test_strategy_embedding_override_only_applies_to_isolated_worker(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", "16")
    monkeypatch.setenv("RAG_PREHOP_EMBEDDING_BATCH_SIZE", "3")
    monkeypatch.setenv("RAG_LIGHTRAG_EMBEDDING_BATCH_SIZE", "7")
    assert EmbeddingOperationalConfig.resolve("prehop").batch_size == 16
    assert EmbeddingOperationalConfig.resolve("lightrag").batch_size == 7


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
    with pytest.raises(RuntimeError, match="RAG_INFERENCE_BASE_URL"):
        InferenceTransport.resolve("prehop")
