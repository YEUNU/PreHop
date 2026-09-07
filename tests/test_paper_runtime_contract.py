import hashlib
import json
import os
from pathlib import Path

import pytest

from core.admission import admission_bindings, verifier_sources
from core.inference_transport import InferenceTransport
from core.paper_policy import (
    canonical_operational_policy,
    canonical_semantic_index_policy,
    validate_canonical_index_policy,
    validate_paper_semantic_environment,
)
from core.semantic_config import semantic_config_sha256, semantic_index_policy
from core.strategy_registry import BY_NAME
from models.official_baseline_runtime import _runtime_env
from scripts import check_paper_runtime
from scripts.verify_index_policy import verify as verify_index_policy
from utils.provenance import _is_generated_path


def _canonical_transport(monkeypatch, strategy="prehop"):
    from core.strategy_registry import PAPER_TRANSPORT, paper_environment_defaults
    for name, value in paper_environment_defaults().items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("EMBEDDING_QUERY_INSTRUCTION", "" if strategy == "hipporag2" else PAPER_TRANSPORT.query_instruction)
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
    monkeypatch.setattr("core.inference_transport._approved_gateway_identity", lambda: hashlib.sha256(b"http://litellm.test/v1").hexdigest())
    monkeypatch.setenv("RAG_INFERENCE_API_KEY", "test-key")
    monkeypatch.setenv("RAG_GENERATION_MODEL", "gemma-4-31b-it")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "qwen3-embedding-4b")
    if strategy == "youtu_graphrag":
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
    assert BY_NAME["youtu_graphrag"].paper_embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert dict(BY_NAME["youtu_graphrag"].dataset_aliases) == {
        "multihoprag": "hotpot",
        "musique": "musique",
    }


def test_transport_rejects_invalid_timeout_and_parses_seed(monkeypatch):
    _canonical_transport(monkeypatch)
    transport = InferenceTransport.resolve("prehop")
    assert transport.generation_seed == 42
    for value in ("-1", "nan", "inf"):
        monkeypatch.setenv("RAG_INFERENCE_TIMEOUT", value)
        with pytest.raises(ValueError, match="timeout"):
            InferenceTransport.resolve("prehop")


def test_paper_common_semantic_overrides_fail_closed(monkeypatch):
    monkeypatch.setenv("EMBEDDING_QUERY_INSTRUCTION", "ambient drift")
    with pytest.raises(RuntimeError, match="EMBEDDING_QUERY_INSTRUCTION"):
        validate_paper_semantic_environment("lightrag", "musique")
    monkeypatch.delenv("EMBEDDING_QUERY_INSTRUCTION")
    monkeypatch.setenv("MAX_EMBEDDING_LENGTH", "8192")
    with pytest.raises(RuntimeError, match="MAX_EMBEDDING_LENGTH"):
        validate_paper_semantic_environment("prehop", "musique")
    monkeypatch.setenv("MAX_EMBEDDING_LENGTH", "32768")
    monkeypatch.setenv("RAG_MS_REPORT_MAX_TOKENS", "2048")
    with pytest.raises(RuntimeError, match="RAG_MS_REPORT_MAX_TOKENS"):
        validate_paper_semantic_environment("ms_graphrag", "musique")
    monkeypatch.delenv("RAG_MS_REPORT_MAX_TOKENS")
    # The superseded 0.6B width must be rejected by the current 4B contract.
    monkeypatch.setenv("NEO4J_VECTOR_DIMENSIONS", "1024")
    with pytest.raises(RuntimeError, match="NEO4J_VECTOR_DIMENSIONS"):
        validate_paper_semantic_environment("lightrag", "musique")
    monkeypatch.setenv("NEO4J_VECTOR_DIMENSIONS", "2560")
    monkeypatch.setenv("RAG_MAX_CONTEXT_LENGTH", "16384")
    with pytest.raises(RuntimeError, match="RAG_MAX_CONTEXT_LENGTH"):
        validate_paper_semantic_environment("prehop", "musique")
    monkeypatch.setenv("RAG_MAX_CONTEXT_LENGTH", "262144")
    monkeypatch.setenv("RAG_EMBEDDING_TOKEN_RESERVE", "128")
    with pytest.raises(RuntimeError, match="RAG_EMBEDDING_TOKEN_RESERVE"):
        validate_paper_semantic_environment("prehop", "musique")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RAG_QUESTION_SCHEMA", "linked_v2"),
        ("RAG_ABLATION_Q_MINUS", "false"),
        ("RAG_ABLATION_Q_PLUS", "false"),
        ("RAG_PRECOMPUTE_RECIPROCAL_HOPS", "false"),
        ("RAG_SENTENCE_CHANNEL_ENABLED", "true"),
        ("RAG_GRAPH_HOP_DEPTH", "0"),
        ("RAG_GRAPH_PATH_DECAY", "0.75"),
        ("RAG_GRAPH_EDGE_VARIANT", "next_only"),
        ("RAG_HOP_EDGE_FILTER", "reciprocal"),
        ("RAG_QPLUS_HOP_ACTIVATION", "exact"),
        ("RAG_CONTINUATION_EDGES_ENABLED", "true"),
        ("RAG_CONTINUATION_ANCHOR_POLICY", "all_grounded"),
        ("RAG_HOP_SEMANTIC_VARIANT", "bridge_only"),
        ("RAG_QUERY_REWRITE_VARIANT", "none"),
        ("RAG_QUERY_REWRITE_MAX_WORDS", "16"),
        ("RAG_QUERY_REFINEMENT_MAX_ROUNDS", "1"),
        ("RAG_CANDIDATE_POOL_MULTIPLIER", "2"),
        ("RAG_HYPO_CHANNEL_VARIANT", "body_only"),
        ("RAG_SOURCE_SELECTION_VARIANT", "global"),
        ("RAG_CANDIDATE_ORDER_INPUT_ORDER", "shuffle"),
        ("RAG_CANDIDATE_ORDER_SHUFFLE_SEED", "1"),
        ("RAG_FINAL_RANK_VARIANT", "semantic_only"),
    ],
)
def test_core_method_semantic_overrides_fail_closed(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        validate_paper_semantic_environment("prehop", "musique")
    with pytest.raises(RuntimeError, match=name):
        validate_paper_semantic_environment("naive", "musique")


def test_typed_transport_rejects_ambient_embedding_semantic_drift(monkeypatch):
    _canonical_transport(monkeypatch)
    monkeypatch.setenv("RAG_PAPER_MODE", "true")
    monkeypatch.setenv("EMBEDDING_QUERY_INSTRUCTION", "ambient drift")
    with pytest.raises(RuntimeError, match="EMBEDDING_QUERY_INSTRUCTION"):
        InferenceTransport.resolve("lightrag")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RAG_INFERENCE_TIMEOUT", "599"),
        ("RAG_GENERATION_CONCURRENCY", "29"),
        ("RAG_INFERENCE_RETRY_ATTEMPTS", "4"),
        ("RAG_EMBEDDING_BATCH_SIZE", "15"),
        ("RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS", "2"),
        ("RAG_LLM_SEED", "41"),
    ],
)
def test_paper_transport_operational_and_seed_drift_fail_closed(monkeypatch, name, value):
    _canonical_transport(monkeypatch)
    monkeypatch.setenv("RAG_PAPER_MODE", "true")
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        InferenceTransport.resolve("prehop")


def test_paper_transport_rejects_legacy_alias_even_with_canonical_gateway(monkeypatch):
    _canonical_transport(monkeypatch)
    monkeypatch.setenv("RAG_PAPER_MODE", "true")
    monkeypatch.setenv("VLLM_URL", "http://bypass.invalid/v1")
    with pytest.raises(RuntimeError, match="ambient provider/legacy aliases"):
        InferenceTransport.resolve("prehop")


def test_youtu_rejects_false_seed_claim(monkeypatch):
    _canonical_transport(monkeypatch, "youtu_graphrag")
    monkeypatch.setenv("RAG_PAPER_MODE", "true")
    monkeypatch.setenv("RAG_LLM_SEED", "42")
    with pytest.raises(RuntimeError, match="RAG_LLM_SEED"):
        InferenceTransport.resolve("youtu_graphrag")


def test_paper_semantic_boolean_parser_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("RAG_LINEAR_RAG_VECTORIZED", "maybe")
    with pytest.raises(RuntimeError, match="invalid boolean"):
        validate_paper_semantic_environment("linear_rag", "musique")


def test_unknown_method_semantic_override_fails_closed(monkeypatch):
    monkeypatch.setenv("RAG_YOUTU_UNREGISTERED_MODE", "native-ish")
    with pytest.raises(RuntimeError, match="unknown paper method environment override"):
        validate_paper_semantic_environment("youtu_graphrag", "musique")


def test_preflight_rejects_transport_model_drift(monkeypatch):
    _canonical_transport(monkeypatch)
    monkeypatch.setenv("RAG_GENERATION_MODEL", "wrong-generation")
    with pytest.raises(RuntimeError, match="generation model"):
        check_paper_runtime._check_transport("prehop")
    monkeypatch.setenv("RAG_GENERATION_MODEL", "gemma-4-31b-it")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "qwen3-embedding-4b")
    check_paper_runtime._check_transport("prehop")


def test_canonical_youtu_policy_uses_native_twenty_and_explicit_alias(monkeypatch):
    policy = _complete_policy(monkeypatch, "youtu_graphrag", "multihoprag")
    assert policy["dataset_alias"] == "hotpot"
    assert policy["no_chunk"] is True
    assert policy["retrieval_top_k"] == policy["retrieval_top_k_filter"] == 20
    assert policy["generation_seed"] is None
    assert policy["generation_seed_control"] == "upstream_native_unseeded"
    digest = semantic_config_sha256(policy)
    assert validate_canonical_index_policy("youtu_graphrag", "multihoprag", policy, digest) == digest
    with pytest.raises(RuntimeError, match="checked-in paper policy"):
        validate_canonical_index_policy(
            "youtu_graphrag",
            "multihoprag",
            {**policy, "retrieval_top_k": 5},
            semantic_config_sha256({**policy, "retrieval_top_k": 5}),
        )
    runtime_policy = {**policy, "schema_path": "/runtime/source/schemas/hotpot.json"}
    runtime_digest = semantic_config_sha256(runtime_policy)
    assert (
        validate_canonical_index_policy("youtu_graphrag", "multihoprag", runtime_policy, runtime_digest)
        == runtime_digest
    )
    wrong_order = {**policy, "schema_path": "/runtime/hotpot.json/schemas"}
    with pytest.raises(RuntimeError, match="schema_path"):
        validate_canonical_index_policy(
            "youtu_graphrag", "multihoprag", wrong_order, semantic_config_sha256(wrong_order)
        )


def test_canonical_youtu_policy_uses_no_chunk_for_both_upstream_aliases():
    assert canonical_semantic_index_policy("youtu_graphrag", "multihoprag")["no_chunk"] is True
    assert canonical_semantic_index_policy("youtu_graphrag", "musique")["no_chunk"] is True


def test_canonical_policy_records_shared_dimensions_context_and_reserve():
    policy = canonical_semantic_index_policy("prehop", "musique")
    assert policy["embedding_dimensions"] == 2560
    assert policy["embedding_max_input_tokens"] == 32768
    assert policy["embedding_token_reserve"] == 0
    assert policy["generation_max_context_tokens"] == 262144
    naive = canonical_semantic_index_policy("naive", "musique")
    assert naive["question_schema"] == "legacy"
    assert naive["q_minus_enabled"] is naive["q_plus_enabled"] is True
    assert naive["precompute_reciprocal_hops"] is True


@pytest.mark.parametrize(
    "strategy",
    ["prehop", "naive", "ms_graphrag", "lightrag", "hipporag2", "linear_rag"],
)
def test_paper_index_builder_emits_registry_canonical_semantics(monkeypatch, strategy):
    from cli.index import _resolved_index_policy

    monkeypatch.setenv("RAG_PAPER_MODE", "true")
    _canonical_transport(monkeypatch, strategy)
    observed = semantic_index_policy(_resolved_index_policy(strategy, "default", "musique"))
    assert observed == {
        **canonical_semantic_index_policy(strategy, "musique"),
        "operational_config": canonical_operational_policy(strategy),
    }


def test_youtu_child_forces_single_provider_and_canonical_aliases(monkeypatch):
    _canonical_transport(monkeypatch, "youtu_graphrag")
    monkeypatch.setenv("OPENAI_PROVIDER", "azure")
    monkeypatch.setenv("API_VERSION", "ambient")
    env = _runtime_env("youtu_graphrag")
    assert not env.get("OPENAI_PROVIDER")
    assert "API_VERSION" not in env
    assert env["LLM_BASE_URL"] == env["RAG_INFERENCE_BASE_URL"] == "http://litellm.test/v1"
    assert env["LLM_MODEL"] == env["RAG_GENERATION_MODEL"] == "gemma-4-31b-it"
    assert not env.get("RAG_LLM_SEED")


def test_runtime_preflight_consumes_manifest_and_checks_local_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(check_paper_runtime, "validate_runtime", lambda strategy: None)
    monkeypatch.setattr(check_paper_runtime, "official_python", lambda strategy: Path("/runtime/python"))
    monkeypatch.setattr(check_paper_runtime, "_check_main_runtime", lambda: None)
    monkeypatch.setattr(check_paper_runtime, "_check_frozen_runtime", lambda python: None)
    requirements = check_paper_runtime.load_runtime_requirements()
    monkeypatch.setattr(check_paper_runtime, "load_runtime_requirements", lambda: requirements)
    checked = []
    monkeypatch.setattr(
        check_paper_runtime, "_check_python_contract", lambda python, requirement: checked.append((python, requirement))
    )
    snapshots = []
    monkeypatch.setattr(
        check_paper_runtime,
        "_check_local_snapshot",
        lambda strategy, subdir, model, revision, *args, **kwargs: snapshots.append((model, revision)),
    )
    schema = tmp_path / "source/schemas/musique.json"
    schema.parent.mkdir(parents=True)
    schema.write_text("{}", encoding="utf-8")
    import hashlib

    requirements["youtu_graphrag"]["approved_schemas"]["musique"]["sha256"] = hashlib.sha256(
        schema.read_bytes()
    ).hexdigest()
    monkeypatch.setattr("core.paper_policy.runtime_requirement", lambda strategy: requirements[strategy])
    monkeypatch.setattr(check_paper_runtime, "official_root", lambda strategy: tmp_path / "source")

    check_paper_runtime.check("youtu_graphrag", "musique")

    assert checked[0][1]["python_version"] == "3.10"
    assert snapshots == [
        (
            "sentence-transformers/all-MiniLM-L6-v2",
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        )
    ]


def test_runtime_preflight_rejects_missing_manifest_selected_schema(monkeypatch):
    monkeypatch.setattr(check_paper_runtime, "validate_runtime", lambda strategy: None)
    monkeypatch.setattr(check_paper_runtime, "official_python", lambda strategy: Path("/runtime/python"))
    monkeypatch.setattr(check_paper_runtime, "official_root", lambda strategy: Path("/missing/source"))
    monkeypatch.setattr(check_paper_runtime, "_check_main_runtime", lambda: None)
    monkeypatch.setattr(check_paper_runtime, "_check_frozen_runtime", lambda python: None)
    monkeypatch.setattr(check_paper_runtime, "_check_python_contract", lambda python, requirement: None)
    monkeypatch.setattr(check_paper_runtime, "_check_local_snapshot", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="approved schema"):
        check_paper_runtime.check("youtu_graphrag", "musique")


def test_paper_runner_and_matrix_bind_preflight_seed_and_admission():
    runner = Path("scripts/run_paper_target.sh").read_text(encoding="utf-8")
    matrix = Path("scripts/run_paper_matrix.sh").read_text(encoding="utf-8")
    assert 'export RAG_LLM_SEED=""' in runner
    assert 'export RAG_LLM_SEED="${RAG_LLM_SEED:-$generation_seed}"' in runner
    assert 'check_paper_runtime.py --strategy "$strategy"' in runner
    assert "--output-tsv" in runner
    assert "RAG_MS_OUTPUT_ROOT=" not in runner
    assert "verify_paper_target.py" in runner
    assert "verify_paper_target.py" in matrix
    assert 'failed_targets+=("$dataset/$strategy:$rc")' in matrix
    assert "RAG_PAPER_GENERATION_CONCURRENCY" not in runner
    assert '--output "data/results/$run_id/admission.json"' in runner
    assert "verify_index_policy.py" in runner
    assert "paper_gate_ledger.py" in matrix


def test_youtu_driver_fails_closed_on_upstream_source_failures():
    source = Path("models/external_research/drivers/youtu_graphrag.py").read_text(encoding="utf-8")
    assert 'status = "swallowed_error"' in source
    assert '"native_extraction_swallowed_error_count"' in source
    assert '"native_extraction_success_complete"' in source
    assert "AZURE_OPENAI_ENDPOINT" in source


def test_runtime_requirements_are_valid_json_and_cover_external_primary_methods():
    payload = json.loads(Path("configs/paper_runtime_requirements.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert {"ms_graphrag", "lightrag", "hipporag2", "gfm_rag", "linear_rag", "youtu_graphrag"} <= payload.keys()


def test_preflight_main_uses_runner_environment_and_typed_transport():
    source = Path("scripts/check_paper_runtime.py").read_text(encoding="utf-8")
    assert "load_dotenv(env_path, override=False)" in source
    assert "InferenceTransport.resolve(strategy)" in source


def test_index_reuse_rejects_policy_mutation(tmp_path, monkeypatch):
    policy = _complete_policy(monkeypatch, "prehop", "musique")
    stats = tmp_path / "index.json"
    payload = {
        "status": "complete",
        "strategy": "prehop",
        "corpus_tag": "musique",
        "run_id": "run",
        "index_policy": policy,
        "index_policy_sha256": semantic_config_sha256(policy),
    }
    stats.write_text(json.dumps(payload), encoding="utf-8")
    verify_index_policy(stats, "prehop", "musique", "run")
    payload["index_policy"] = {**policy, "default_top_k": 8}
    payload["index_policy_sha256"] = semantic_config_sha256(payload["index_policy"])
    stats.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checked-in paper policy"):
        verify_index_policy(stats, "prehop", "musique", "run")


def test_index_reuse_rejects_stale_incomplete_effective_policy(tmp_path, monkeypatch):
    policy = _complete_policy(monkeypatch, "prehop", "musique")
    policy.pop("embedding_dimensions")
    stats = tmp_path / "index.json"
    stats.write_text(
        json.dumps(
            {
                "status": "complete",
                "strategy": "prehop",
                "corpus_tag": "musique",
                "run_id": "run",
                "index_policy": policy,
                "index_policy_sha256": semantic_config_sha256(policy),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="checked-in paper policy"):
        verify_index_policy(stats, "prehop", "musique", "run")


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
        "configs/runtime_constraints/hipporag2.txt",
        "configs/runtime_constraints/gfm_rag.txt",
        "configs/runtime_constraints/linear_rag.txt",
        "configs/runtime_constraints/youtu_graphrag.txt",
    } <= {path.relative_to(Path.cwd()).as_posix() for path in verifier_sources()}


def test_frozen_runtime_rejects_dependency_drift(tmp_path, monkeypatch):
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    (tmp_path / "runtime.freeze.txt").write_text("package==1\n", encoding="utf-8")

    class Result:
        stdout = "package==2\n"

    monkeypatch.setattr(check_paper_runtime.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(RuntimeError, match="dependencies differ"):
        check_paper_runtime._check_frozen_runtime(python)


def test_frozen_runtime_runs_installed_metadata_consistency_check(tmp_path, monkeypatch):
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    (tmp_path / "runtime.freeze.txt").write_text("package==1\n", encoding="utf-8")
    calls = []

    class Result:
        stdout = "package==1\n"

    def capture(args, **kwargs):
        calls.append(args)
        return Result()

    monkeypatch.setattr(check_paper_runtime.subprocess, "run", capture)
    check_paper_runtime._check_frozen_runtime(python)
    assert ["uv", "pip", "check", "--python", str(python)] in calls


def test_checked_in_runtime_constraints_are_content_bound():
    requirements = check_paper_runtime.load_runtime_requirements()
    for strategy in ("lightrag", "hipporag2", "gfm_rag", "linear_rag", "youtu_graphrag"):
        check_paper_runtime._check_approved_constraints(requirements[strategy])
    mutated = dict(requirements["linear_rag"])
    mutated["constraints_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="constraints are missing or modified"):
        check_paper_runtime._check_approved_constraints(mutated)
    setup = Path("scripts/setup_official_baselines.sh").read_text(encoding="utf-8")
    assert "constraint_path linear_rag" in setup
    assert "constraint_path youtu_graphrag" in setup
    assert "constraint_path lightrag" in setup
    assert "constraint_path hipporag2" in setup
    assert "constraint_path gfm_rag" in setup
    assert '--constraint "$linear_constraints"' in setup
    assert '--constraint "$youtu_constraints"' in setup
    gfm_constraints = Path("configs/runtime_constraints/gfm_rag.txt").read_text(encoding="utf-8")
    assert "torch==2.8.0" in gfm_constraints
    assert "vllm==0.10.2" in gfm_constraints
    assert "openai==2.29.0" in gfm_constraints
    assert "langchain-openai==0.3.35" in gfm_constraints
    assert "torch==2.4.1" not in gfm_constraints


def test_spacy_distribution_version_is_part_of_executable_preflight():
    source = Path("scripts/check_paper_runtime.py").read_text(encoding="utf-8")
    assert "metadata.version" in source
    assert "spaCy model version mismatch" in source
    setup = Path("scripts/setup_official_baselines.sh").read_text(encoding="utf-8")
    assert "runtime.freeze.txt" in setup
