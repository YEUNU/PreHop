import json
from pathlib import Path

from core.admission import admission_bindings
from core.paper_policy import canonical_semantic_index_policy
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


def test_youtu_cross_dataset_config_ignores_only_dataset_specific_policy():
    hotpot = _payload(revision="same", namespace="hotpot")
    musique = _payload(revision="same", namespace="musique")
    hotpot["index_provenance"]["policy"] = canonical_semantic_index_policy("youtu_graphrag", "multihoprag")
    musique["index_provenance"]["policy"] = canonical_semantic_index_policy("youtu_graphrag", "musique")
    hotpot["index_provenance"]["policy"]["schema_path"] = "/runtime/schemas/hotpot.json"
    musique["index_provenance"]["policy"]["schema_path"] = "/runtime/schemas/musique.json"
    assert _semantic_model_config_sha256(hotpot, "youtu_graphrag") == _semantic_model_config_sha256(
        musique, "youtu_graphrag"
    )
    musique["index_provenance"]["policy"]["retrieval_top_k"] = 19
    assert _semantic_model_config_sha256(hotpot, "youtu_graphrag") != _semantic_model_config_sha256(
        musique, "youtu_graphrag"
    )


def test_local_embedding_revisions_match_registry_contract():
    assert _expected_embedding_config("linear_rag")[2] == "e8c3b32edf5434bc2275fc9bab85f82640a19130"
    assert _expected_embedding_config("youtu_graphrag")[2] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


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
    assert approved["query_rewrite_variant"] == "role_aligned_evidence_iterative"
    assert approved["graph_hop_depth"] == 1
    assert approved["default_top_k"] == 12
    mutated = {**approved, "default_top_k": 8}
    assert mutated != approved


def test_submission_admission_requires_fresh_content_bindings(tmp_path, monkeypatch):
    relative = Path("data/results/run/naive/musique/seed_42/naive_musique.json")
    result = tmp_path / relative
    result.parent.mkdir(parents=True)
    payload = {"index_provenance": {"policy_sha256": "a" * 64, "code": {}}, "query_provenance": {}}
    result.write_text(json.dumps(payload), encoding="utf-8")
    result.with_name("naive_musique.details.jsonl").write_text("{}\n", encoding="utf-8")
    ledger = result.parents[3] / "admission.json"
    ledger.write_text(
        json.dumps(
            {
                "status": "admitted",
                "path": str(result.resolve()),
                "dataset": "musique",
                "strategy": "naive",
                "bindings": admission_bindings(result.resolve(), payload),
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_submission_consistency, "ROOT", tmp_path)
    assert _validate_admission(relative, payload, "musique", "naive") == []
    result.write_text(json.dumps({**payload, "changed": True}), encoding="utf-8")
    assert "stale" in " ".join(_validate_admission(relative, payload, "musique", "naive"))


def test_submission_admission_rejects_runtime_freeze_or_inventory_drift(tmp_path, monkeypatch):
    from core import admission, runtime_requirements

    relative = Path("data/results/run/naive/musique/seed_42/naive_musique.json")
    result = tmp_path / relative
    result.parent.mkdir(parents=True)
    payload = {
        "strategy": "naive",
        "corpus_tag": "musique",
        "index_provenance": {"policy_sha256": "a" * 64, "code": {}},
        "query_provenance": {},
    }
    result.write_text(json.dumps(payload), encoding="utf-8")
    result.with_name("naive_musique.details.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("core.paper_compatibility.target_configuration", lambda *args: {"fixture": "configuration"})
    monkeypatch.setattr(runtime_requirements, "runtime_identity", lambda strategy: {"freeze": "first"})
    monkeypatch.setattr(admission, "current_post_query_inventory", lambda *args: {"sha256": "first"})
    ledger = result.parents[3] / "admission.json"
    ledger.write_text(
        json.dumps(
            {
                "status": "admitted",
                "path": str(result.resolve()),
                "dataset": "musique",
                "strategy": "naive",
                "bindings": admission_bindings(result.resolve(), payload),
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_submission_consistency, "ROOT", tmp_path)
    assert _validate_admission(relative, payload, "musique", "naive") == []
    monkeypatch.setattr(runtime_requirements, "runtime_identity", lambda strategy: {"freeze": "second"})
    assert "stale" in " ".join(_validate_admission(relative, payload, "musique", "naive"))
    monkeypatch.setattr("core.paper_compatibility.target_configuration", lambda *args: {"fixture": "configuration"})
    monkeypatch.setattr(runtime_requirements, "runtime_identity", lambda strategy: {"freeze": "first"})
    monkeypatch.setattr(admission, "current_post_query_inventory", lambda *args: {"sha256": "second"})
    assert "stale" in " ".join(_validate_admission(relative, payload, "musique", "naive"))


def test_primary_scores_recomputed_from_predictions_and_authoritative_gold():
    import copy

    expected = {
        'ground_truth': 'New York', 'answer_aliases': ['NYC'],
        'evidence_paragraph_ids': ['musique:aabbccddeeff'],
    }
    row = {
        'answer': 'NYC', 'retrieved_sources': [{'paragraph_id': 'musique:aabbccddeeff'}],
        'expected_sources': {'docs': [], 'facts': [], 'paragraph_ids': ['musique:aabbccddeeff']},
        'official_answer_em': 1.0, 'official_answer_f1': 1.0,
        'paragraph_support_precision': 1.0, 'paragraph_support_recall': 1.0,
        'paragraph_support_f1': 1.0,
    }
    validate = verify_submission_consistency._validate_primary_row
    assert validate(row, expected, 'musique') == []
    for invalid in (float('inf'), float('nan'), True, 1.1, 0.5, None):
        mutated = {**row, 'official_answer_f1': invalid}
        assert any('official_answer_f1' in error for error in validate(mutated, expected, 'musique'))
    assert validate({**row, 'answer': 'Boston'}, expected, 'musique')
    mutated = copy.deepcopy(row)
    mutated['expected_sources']['paragraph_ids'] = ['musique:ffffffffffff']
    assert any('expected_sources' in error for error in validate(mutated, expected, 'musique'))


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
