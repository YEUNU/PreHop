import os

import pytest

from core.inference_transport import InferenceTransport
from core.paper_policy import canonical_operational_policy, canonical_semantic_index_policy
from core.semantic_config import semantic_index_policy
from core.strategy_registry import BY_NAME
from utils.provenance import _is_generated_path


def _canonical_transport(monkeypatch):
    from core.strategy_registry import PAPER_TRANSPORT, paper_environment_defaults
    for name, value in paper_environment_defaults().items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("EMBEDDING_QUERY_INSTRUCTION", PAPER_TRANSPORT.query_instruction)
    monkeypatch.setenv("NEO4J_VECTOR_DIMENSIONS", "2560")
    monkeypatch.setenv("MAX_EMBEDDING_LENGTH", "32768")
    for name in tuple(os.environ):
        if name.startswith(("VLLM_", "AZURE_OPENAI")) or name in {
            "OPENAI_API_BASE",
            "OPENAI_BASE_URL",
            "OPENAI_PROVIDER",
        }:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAG_INFERENCE_BASE_URL", "http://litellm.test/v1")
    monkeypatch.setenv("RAG_INFERENCE_API_KEY", "test-key")
    monkeypatch.setenv("RAG_GENERATION_MODEL", "gemma-4-31b-it")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "qwen3-embedding-4b")
    monkeypatch.setenv("RAG_LLM_SEED", "42")


def test_every_strategy_records_the_single_litellm_transport_profile(monkeypatch):
    _canonical_transport(monkeypatch)
    profiles = {InferenceTransport.resolve(name).policy_dict()["transport_profile"] for name in BY_NAME}
    assert profiles == {"openai_compatible_litellm"}


def test_canonical_policy_records_shared_dimensions_context_and_reserve():
    policy = canonical_semantic_index_policy("prehop")
    assert policy["embedding_dimensions"] == 2560
    assert policy["embedding_max_input_tokens"] == 32768
    assert policy["embedding_token_reserve"] == 0
    assert policy["generation_max_context_tokens"] == 262144
    naive = canonical_semantic_index_policy("naive")
    assert naive["question_schema"] == "legacy"
    assert naive["q_minus_enabled"] is naive["q_plus_enabled"] is True
    assert naive["precompute_reciprocal_hops"] is True


@pytest.mark.parametrize("strategy", ["prehop", "naive"])
def test_paper_index_builder_emits_registry_canonical_semantics(monkeypatch, strategy):
    from cli.index import _resolved_index_policy

    monkeypatch.setenv("RAG_PAPER_MODE", "true")
    _canonical_transport(monkeypatch)
    observed = semantic_index_policy(_resolved_index_policy(strategy, "default"))
    assert observed == {
        **canonical_semantic_index_policy(strategy),
        "operational_config": canonical_operational_policy(strategy),
    }


def test_code_provenance_excludes_failed_and_generated_output_roots_without_reading_them():
    assert _is_generated_path(b"data/failed_runs/opaque.json")
    assert _is_generated_path(b"data/results/run/result.json")
    assert _is_generated_path(b"data/index_stats/prehop_hotpotqa_run.json")
    assert not _is_generated_path(b"core/paper_policy.py")



@pytest.mark.parametrize("seed", ["41", "42", "invalid", ""])
def test_paper_generation_omits_ambient_seed(monkeypatch, seed):
    _canonical_transport(monkeypatch)
    monkeypatch.setenv("RAG_PAPER_MODE", "true")
    monkeypatch.setenv("RAG_LLM_SEED", seed)
    assert InferenceTransport.resolve("prehop").generation_seed is None
