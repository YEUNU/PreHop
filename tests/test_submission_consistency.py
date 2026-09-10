import json
from pathlib import Path

from core.admission import admission_bindings
from scripts import verify_submission_consistency
from scripts.verify_submission_consistency import (
    _approved_ablation_policy,
    _expected_embedding_config,
    _registered_revision_error,
    _semantic_model_config_sha256,
    _validate_admission,
)


def _payload(*, revision: str, namespace: str, top_k: int = 12):
    return {
        "query_provenance": {"revision": revision, "source_tree_sha256": revision},
        "models": {
            "default": "generation-model",
            "generation_revision": "generation-v1",
            "embedding": "embedding-model",
            "embedding_revision": "embedding-v1",
            "llm_seed": 42,
        },
        "index_provenance": {
            "policy": {
                "index_namespace": namespace,
                "embedding_model": "embedding-model",
                "default_top_k": top_k,
            }
        },
        "ablation": {"q_minus": True, "q_plus": True},
    }


def test_semantic_config_ignores_commit_and_run_namespace():
    first = _payload(revision="commit-a", namespace="dataset_run_a")
    second = _payload(revision="commit-b", namespace="dataset_run_b")

    assert _semantic_model_config_sha256(first) == _semantic_model_config_sha256(second)


def test_semantic_config_detects_model_setting_change():
    first = _payload(revision="commit-a", namespace="dataset_run_a", top_k=12)
    second = _payload(revision="commit-a", namespace="dataset_run_a", top_k=8)

    assert _semantic_model_config_sha256(first) != _semantic_model_config_sha256(second)


def test_local_embedding_revisions_match_registry_contract():
    assert _expected_embedding_config("linear_rag")[2] == "e8c3b32edf5434bc2275fc9bab85f82640a19130"


def test_registered_official_revision_rejects_consistently_stale_artifact():
    error = _registered_revision_error("lightrag", {"official_revision": "stale"}, Path("artifact.json"))
    assert error is not None and "expected '440d25" in error
    assert _registered_revision_error(
        "lightrag",
        {"official_revision": "440d25b0cbfdd94b3c92f7bfb0c3c2989c779671"},
        Path("artifact.json"),
    ) is None


def test_approved_core_policy_detects_method_defining_changes():
    approved = _approved_ablation_policy("prehop")
    assert approved["query_execution"] == "original-question-single-retrieval-v1"
    assert approved["graph_hop_depth"] == 1
    assert approved["default_top_k"] == 12
    mutated = {**approved, "default_top_k": 8}
    assert mutated != approved


def test_submission_admission_requires_fresh_content_bindings(tmp_path, monkeypatch):
    relative = Path("data/results/run/naive/hotpotqa/seed_42/naive_hotpotqa.json")
    result = tmp_path / relative
    result.parent.mkdir(parents=True)
    payload = {"index_provenance": {"policy_sha256": "a" * 64, "code": {}}, "query_provenance": {}}
    result.write_text(json.dumps(payload), encoding="utf-8")
    result.with_name("naive_hotpotqa.details.jsonl").write_text("{}\n", encoding="utf-8")
    ledger = result.parents[3] / "admission.json"
    ledger.write_text(
        json.dumps(
            {
                "status": "admitted",
                "path": str(result.resolve()),
                "dataset": "hotpotqa",
                "strategy": "naive",
                "bindings": admission_bindings(result.resolve(), payload),
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_submission_consistency, "ROOT", tmp_path)
    assert _validate_admission(relative, payload, "hotpotqa", "naive") == []
    result.write_text(json.dumps({**payload, "changed": True}), encoding="utf-8")
    assert "stale" in " ".join(_validate_admission(relative, payload, "hotpotqa", "naive"))


def test_submission_admission_rejects_runtime_freeze_or_inventory_drift(tmp_path, monkeypatch):
    from core import admission, runtime_requirements

    relative = Path("data/results/run/naive/hotpotqa/seed_42/naive_hotpotqa.json")
    result = tmp_path / relative
    result.parent.mkdir(parents=True)
    payload = {
        "strategy": "naive",
        "corpus_tag": "hotpotqa",
        "index_provenance": {"policy_sha256": "a" * 64, "code": {}},
        "query_provenance": {},
    }
    result.write_text(json.dumps(payload), encoding="utf-8")
    result.with_name("naive_hotpotqa.details.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("core.paper_compatibility.target_configuration", lambda *args: {"fixture": "configuration"})
    monkeypatch.setattr(runtime_requirements, "runtime_identity", lambda strategy: {"freeze": "first"})
    monkeypatch.setattr(admission, "current_post_query_inventory", lambda *args: {"sha256": "first"})
    monkeypatch.setattr(admission, "current_corpus_identity", lambda _: {"fingerprint": "fixture-corpus"})
    ledger = result.parents[3] / "admission.json"
    ledger.write_text(
        json.dumps(
            {
                "status": "admitted",
                "path": str(result.resolve()),
                "dataset": "hotpotqa",
                "strategy": "naive",
                "bindings": admission_bindings(result.resolve(), payload),
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_submission_consistency, "ROOT", tmp_path)
    assert _validate_admission(relative, payload, "hotpotqa", "naive") == []
    monkeypatch.setattr(runtime_requirements, "runtime_identity", lambda strategy: {"freeze": "second"})
    assert "stale" in " ".join(_validate_admission(relative, payload, "hotpotqa", "naive"))
    monkeypatch.setattr("core.paper_compatibility.target_configuration", lambda *args: {"fixture": "configuration"})
    monkeypatch.setattr(runtime_requirements, "runtime_identity", lambda strategy: {"freeze": "first"})
    monkeypatch.setattr(admission, "current_post_query_inventory", lambda *args: {"sha256": "second"})
    assert "stale" in " ".join(_validate_admission(relative, payload, "hotpotqa", "naive"))


def test_primary_scores_recomputed_from_predictions_and_authoritative_gold():
    validate = verify_submission_consistency._validate_primary_row
    expected = {'ground_truth': 'Paris', 'evidence_facts': ['Alpha is in Paris.']}
    row = {'answer': 'Paris', 'retrieved_sources': [{'text': 'Alpha is in Paris.'}],
           'expected_sources': {'docs': [], 'facts': expected['evidence_facts'], 'paragraph_ids': []},
           'official_hits@4': 1.0, 'official_hits@10': 1.0,
           'official_mrr@10': 1.0, 'official_map@10': 1.0}
    assert validate(row, expected, 'multihoprag') == []
    for invalid in (-1.0, 0.0, float('nan'), float('inf')):
        assert validate({**row, 'official_mrr@10': invalid}, expected, 'multihoprag')
    assert validate({**row, 'retrieved_sources': []}, expected, 'multihoprag')
    assert validate({**row, 'expected_sources': {}}, expected, 'multihoprag')


def test_primary_retrieval_recompute_preserves_unanswerable_exclusion():
    validate = verify_submission_consistency._validate_primary_row
    expected = {'evidence_facts': ['Alpha is in Paris.']}
    row = {
        'answer': 'Paris', 'retrieved_sources': [{'text': 'Alpha is in Paris.'}],
        'expected_sources': {'docs': [], 'facts': expected['evidence_facts'], 'paragraph_ids': []},
        **{field.removeprefix('avg_'): 1.0 for field in verify_submission_consistency.DATASETS['multihoprag']['metrics']},
    }
    assert validate(row, expected, 'multihoprag') == []
    assert validate({**row, 'retrieved_sources': []}, expected, 'multihoprag')
    row.update(expected_sources={'docs': [], 'facts': [], 'paragraph_ids': []})
    row.update({field.removeprefix('avg_'): -1.0 for field in verify_submission_consistency.DATASETS['multihoprag']['metrics']})
    assert validate(row, {}, 'multihoprag') == []


def test_average_verification_preserves_summary_when_last_row_ineligible():
    from scripts.verify_submission_consistency import _validate_average_metrics
    rows = [{"official_hits@4": 1.0}, {"official_hits@4": -1.0}]
    assert _validate_average_metrics("result", {"avg_official_hits@4": 1.0}, rows) == []
    errors = _validate_average_metrics("result", {"avg_official_hits@4": 0.5}, rows)
    assert len(errors) == 1 and "recomputed=1.0" in errors[0]
    assert _validate_average_metrics("result", {"avg_official_hits@4": 0.0}, [{}]) == []


def test_compact_details_validate_against_full_rows_and_separate_traces():
    from scripts.verify_submission_consistency import _approved_generation_revisions, _validate_detail_projection
    from utils.reporting import compact_detail_row
    rows = [{"query_id": "q", "query": "question", "answer": "answer", "sources": ["source"]}]
    traces = [{"idx": 1, "query_id": "q", "query": "question", "interaction_trace": [{"step": "native"}]}]
    compact = [compact_detail_row({**rows[0], "interaction_trace": traces[0]["interaction_trace"]}, 1)]
    assert _validate_detail_projection(rows, compact, traces)
    assert not _validate_detail_projection(rows, compact, [])
    assert not _validate_detail_projection(rows, [{**compact[0], "answer": "tampered"}], traces)
    assert not _validate_detail_projection(rows, compact, [{**traces[0], "query_id": "foreign"}])
    assert "4135a98a9b728a548947683219633b25682223ac" in _approved_generation_revisions("gemma-4-31b-it")
    assert "unregistered" not in _approved_generation_revisions("gemma-4-31b-it")
