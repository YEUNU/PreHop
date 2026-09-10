import json
import os
from pathlib import Path

import pytest

from core.admission import admission_bindings, verifier_sources
from core.inference_transport import InferenceTransport
from core.paper_policy import canonical_operational_policy, canonical_semantic_index_policy
from core.semantic_config import semantic_index_policy
from core.strategy_registry import BY_NAME
from utils.provenance import _is_generated_path


def _canonical_transport(monkeypatch, strategy="prehop"):
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
            "RAG_MS_CONCURRENT_REQUESTS",
        }:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAG_INFERENCE_BASE_URL", "http://litellm.test/v1")
    monkeypatch.setenv("RAG_INFERENCE_API_KEY", "test-key")
    monkeypatch.setenv("RAG_GENERATION_MODEL", "gemma-4-31b-it")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "qwen3-embedding-4b")
    if strategy == "lightrag":
        monkeypatch.delenv("RAG_LLM_SEED", raising=False)
    else:
        monkeypatch.setenv("RAG_LLM_SEED", "42")


def _complete_policy(monkeypatch, strategy, dataset):
    _canonical_transport(monkeypatch, strategy)
    return {
        **canonical_semantic_index_policy(strategy, dataset),
        "operational_config": canonical_operational_policy(strategy),
    }


def test_every_strategy_has_one_explicit_litellm_transport_profile():
    assert {spec.transport_profile for spec in BY_NAME.values()} == {"openai_compatible_litellm"}


def test_canonical_policy_records_shared_dimensions_context_and_reserve():
    policy = canonical_semantic_index_policy("prehop", "hotpotqa")
    assert policy["embedding_dimensions"] == 2560
    assert policy["embedding_max_input_tokens"] == 32768
    assert policy["embedding_token_reserve"] == 0
    assert policy["generation_max_context_tokens"] == 262144
    naive = canonical_semantic_index_policy("naive", "hotpotqa")
    assert naive["question_schema"] == "legacy"
    assert naive["q_minus_enabled"] is naive["q_plus_enabled"] is True
    assert naive["precompute_reciprocal_hops"] is True


@pytest.mark.parametrize(
    "strategy",
    ["prehop", "naive", "ms_graphrag", "lightrag", "linear_rag"],
)
def test_paper_index_builder_emits_registry_canonical_semantics(monkeypatch, strategy):
    from cli.index import _resolved_index_policy

    monkeypatch.setenv("RAG_PAPER_MODE", "true")
    _canonical_transport(monkeypatch, strategy)
    observed = semantic_index_policy(_resolved_index_policy(strategy, "default", "hotpotqa"))
    assert observed == {
        **canonical_semantic_index_policy(strategy, "hotpotqa"),
        "operational_config": canonical_operational_policy(strategy),
    }


def test_runtime_requirements_are_valid_json_and_cover_external_primary_methods():
    payload = json.loads(Path("configs/paper_runtime_requirements.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert {"ms_graphrag", "lightrag", "gfm_rag", "linear_rag"} <= payload.keys()


def test_admission_bindings_change_with_result_or_details(tmp_path):
    result = tmp_path / "result.json"
    details = tmp_path / "result.details.jsonl"
    payload = {"index_provenance": {"policy_sha256": "a" * 64, "code": {"revision": "x"}}}
    result.write_text(json.dumps(payload), encoding="utf-8")
    details.write_text('{"idx":1}\n', encoding="utf-8")
    first = admission_bindings(result, payload)
    result.write_text(json.dumps({**payload, "status": "changed"}), encoding="utf-8")
    second = admission_bindings(result, payload)
    assert first["result_sha256"] != second["result_sha256"]
    details.write_text('{"idx":2}\n', encoding="utf-8")
    third = admission_bindings(result, payload)
    assert second["details_sha256"] != third["details_sha256"]


def test_admission_verifier_identity_covers_transitive_policy_inputs():
    relative = {path.relative_to(Path.cwd()).as_posix() for path in verifier_sources()}
    assert {
        "core/paper_policy.py",
        "core/semantic_config.py",
        "core/runtime_requirements.py",
        "core/inference_transport.py",
        "configs/paper_runtime_requirements.json",
    } <= relative


def test_code_provenance_excludes_failed_and_generated_output_roots_without_reading_them():
    assert _is_generated_path(b"data/failed_runs/opaque.json")
    assert _is_generated_path(b"data/results/run/result.json")
    assert _is_generated_path(b"data/linear_rag_output/run/artifact.bin")
    assert not _is_generated_path(b"core/paper_policy.py")
    assert {
        "configs/runtime_constraints/lightrag.txt",
        "configs/runtime_constraints/gfm_rag.txt",
        "configs/runtime_constraints/linear_rag.txt",
    } <= {path.relative_to(Path.cwd()).as_posix() for path in verifier_sources()}


@pytest.mark.parametrize("seed", ["41", "42", "invalid", ""])
def test_paper_generation_omits_ambient_seed(monkeypatch, seed):
    _canonical_transport(monkeypatch)
    monkeypatch.setenv("RAG_PAPER_MODE", "true")
    monkeypatch.setenv("RAG_LLM_SEED", seed)
    assert InferenceTransport.resolve("prehop").generation_seed is None


def test_hoprag_cli_selects_its_pinned_runtime_without_resolving_launcher(tmp_path, monkeypatch):
    import sys

    from core import runtime_requirements as runtime
    from models.hoprag import native_runtime
    runtime_home = tmp_path / "hoprag"
    launcher = runtime_home / "main-env/bin/python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(sys.executable)
    monkeypatch.setattr(native_runtime, "RUNTIME_HOME", runtime_home)
    argv = ["scripts/paper_cold_canary.py", "fixture", "hoprag", "hotpotqa", "--attempt", "a1"]
    monkeypatch.setattr(sys, "argv", argv)
    captured = {}
    def execute(path, args, env):
        captured.update(path=path, args=args, env=env)
    monkeypatch.setattr(runtime.os, "execve", execute)
    runtime.ensure_method_runtime("hoprag")
    assert captured["path"] == str(launcher)
    assert captured["args"] == [str(launcher), *argv]
    assert captured["env"]["PYTHON_BIN"] == str(launcher)
    assert captured["env"]["UV_PROJECT_ENVIRONMENT"] == str(runtime_home / "main-env")
    captured.clear()
    runtime.ensure_method_runtime("prehop")
    assert captured == {}
