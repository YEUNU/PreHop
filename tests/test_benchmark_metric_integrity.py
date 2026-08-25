import hashlib
import json
import os

import pandas as pd
import pytest

from cli.benchmark import (
    OFFICIAL_QUERY_ID_DIGESTS,
    _aggregate_seed_summaries,
    _apply_judge_label,
    _assert_benchmark_complete,
    _evaluation_scope,
    _judge_independence,
    _latest_index_manifest_metadata,
    _load_benchmark_corpus_manifest,
    _matrix_index_code_provenance,
    _recompute_aggregates,
    _update_summary_status,
    _validate_benchmark_data,
    _validate_corpus_index_fingerprint,
    _verify_active_index_snapshot,
)
from cli.index import _load_corpus_manifest, _verify_and_publish_neo4j_snapshot
from data import prepare_musique
from models.ms_graphrag import official_indexer as ms_official_indexer
from models.prehop.indexing.chunking import parse_pages_offline
from scripts.paired_bootstrap import _load, _paired, _validate_artifact_pair


def test_musique_preparation_defaults_to_the_full_split():
    assert prepare_musique.DEFAULT_LIMIT == 0


def test_evaluation_scope_uses_actual_evaluated_count_before_filename():
    assert _evaluation_scope(
        "multihoprag", 2556, "custom_sample_name.json", OFFICIAL_QUERY_ID_DIGESTS["multihoprag"]
    ) == ("full_benchmark", 2556)
    assert _evaluation_scope("musique", 200, "musique_sample200_queries.json", "subset") == (
        "sample_exploratory",
        2417,
    )
    assert _evaluation_scope("musique", 200, "musique_queries.json", "subset") == ("subset_exploratory", 2417)
    with pytest.raises(ValueError, match="query-id digest"):
        _evaluation_scope("musique", 2417, "musique_queries.json", "wrong")


def test_corpus_manifest_is_optional_but_full_benchmark_requires_matching_index(tmp_path):
    corpus_dir = tmp_path / "musique_corpus"
    corpus_dir.mkdir()
    query_digest = OFFICIAL_QUERY_ID_DIGESTS["musique"]
    manifest_payload = {
        "fingerprint": "corpus-fingerprint",
        "paragraph_count": 2,
        "query_ids_sha256": query_digest,
    }
    (corpus_dir / "corpus_manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")
    queries_path = tmp_path / "musique_queries.json"
    queries_path.write_text("[]", encoding="utf-8")
    stats_dir = tmp_path / "index_stats"
    stats_dir.mkdir()
    stats_path = stats_dir / "prehop_musique_run.json"
    stats_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "corpus_manifest_fingerprint": "corpus-fingerprint",
                "corpus_manifest_paragraph_count": 2,
            }
        ),
        encoding="utf-8",
    )

    corpus_manifest = _load_benchmark_corpus_manifest("musique", queries_path)
    assert _load_corpus_manifest(corpus_dir) == {
        "fingerprint": "corpus-fingerprint",
        "paragraph_count": 2,
    }
    index_manifest = _latest_index_manifest_metadata("prehop", "musique", stats_dir)
    assert (
        _validate_corpus_index_fingerprint("musique", "full_benchmark", corpus_manifest, index_manifest, query_digest)
        == "matched"
    )
    bad_manifest = {**corpus_manifest, "query_ids_sha256": "wrong"}
    with pytest.raises(RuntimeError, match="query-id digest"):
        _validate_corpus_index_fingerprint("musique", "full_benchmark", bad_manifest, index_manifest, query_digest)
    assert (
        _validate_corpus_index_fingerprint("multihoprag", "full_benchmark", None, None, query_digest)
        == "manifest_absent"
    )
    with pytest.raises(RuntimeError, match="requires corpus_manifest"):
        _validate_corpus_index_fingerprint("musique", "full_benchmark", None, None, query_digest)

    stats_path.write_text(
        json.dumps(
            {"status": "complete", "corpus_manifest_fingerprint": "stale", "corpus_manifest_paragraph_count": 2}
        ),
        encoding="utf-8",
    )
    stale_index = _latest_index_manifest_metadata("prehop", "musique", stats_dir)
    with pytest.raises(RuntimeError, match="does not match"):
        _validate_corpus_index_fingerprint("musique", "full_benchmark", corpus_manifest, stale_index, query_digest)
    assert (
        _validate_corpus_index_fingerprint(
            "musique", "subset_exploratory", corpus_manifest, stale_index, "subset-digest"
        )
        == "mismatch_exploratory"
    )


def test_latest_failed_index_artifact_is_not_bypassed_by_older_completed(tmp_path):
    completed = tmp_path / "prehop_musique_completed.json"
    failed = tmp_path / "prehop_musique_failed.json"
    completed.write_text(json.dumps({"status": "complete", "corpus_manifest_fingerprint": "old"}), encoding="utf-8")
    failed.write_text(json.dumps({"status": "failed", "corpus_manifest_fingerprint": "new"}), encoding="utf-8")
    os.utime(completed, (1, 1))
    os.utime(failed, (2, 2))

    latest = _latest_index_manifest_metadata("prehop", "musique", tmp_path)

    assert latest["status"] == "failed"
    with pytest.raises(RuntimeError, match="completed index artifact"):
        _validate_corpus_index_fingerprint(
            "musique",
            "full_benchmark",
            {"fingerprint": "new", "query_ids_sha256": OFFICIAL_QUERY_ID_DIGESTS["musique"]},
            latest,
            OFFICIAL_QUERY_ID_DIGESTS["musique"],
        )


def test_matrix_child_resolves_separate_index_code_provenance(tmp_path):
    artifacts_dir = tmp_path / "indexing"
    run_dir = artifacts_dir / "paper-run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "paper-run",
                "code": {
                    "revision": "index-revision",
                    "dirty": False,
                    "source_tree": {"sha256": "index-tree", "file_count": 88},
                },
            }
        ),
        encoding="utf-8",
    )

    provenance = _matrix_index_code_provenance(
        "paper-run_musique__prehop",
        artifacts_dir,
    )

    assert provenance["revision"] == "index-revision"
    assert provenance["source_tree_sha256"] == "index-tree"
    assert provenance["matrix_manifest_path"].endswith("paper-run/manifest.json")


@pytest.mark.asyncio
async def test_neo4j_snapshot_metadata_is_published_only_after_live_source_set_matches():
    writes = []

    class Neo4j:
        async def execute_query(self, query, parameters=None):
            if "RETURN DISTINCT c.source" in query:
                return [{"source": "musique_alpha.txt"}, {"source": "musique_beta.txt"}]
            writes.append((query, parameters))
            return []

    class Engine:
        chunk_label = "PR_musique_Chunk"
        neo4j = Neo4j()

    metadata = await _verify_and_publish_neo4j_snapshot(
        Engine(),
        "prehop",
        "musique",
        ["musique_alpha", "musique_beta"],
        {"fingerprint": "fp", "paragraph_count": 2},
    )

    assert metadata["status"] == "complete"
    assert metadata["source_count"] == 2
    assert len(writes) == 1
    assert "m.status = 'complete'" in writes[0][0]


@pytest.mark.asyncio
async def test_neo4j_snapshot_mismatch_never_publishes_complete_metadata():
    writes = []

    class Neo4j:
        async def execute_query(self, query, parameters=None):
            if "RETURN DISTINCT c.source" in query:
                return [{"source": "musique_alpha.txt"}]
            writes.append((query, parameters))
            return []

    class Engine:
        chunk_label = "PR_musique_Chunk"
        neo4j = Neo4j()

    with pytest.raises(RuntimeError, match="does not match"):
        await _verify_and_publish_neo4j_snapshot(
            Engine(),
            "prehop",
            "musique",
            ["musique_alpha", "musique_beta"],
            {"fingerprint": "fp", "paragraph_count": 2},
        )
    assert not writes


@pytest.mark.asyncio
async def test_full_active_snapshot_gate_checks_metadata_and_live_sources(tmp_path):
    corpus_dir = tmp_path / "musique_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "musique_alpha.txt").write_text("x", encoding="utf-8")
    (corpus_dir / "musique_beta.txt").write_text("x", encoding="utf-8")
    manifest_path = corpus_dir / "corpus_manifest.json"
    manifest_path.write_text(json.dumps({"fingerprint": "fp", "paragraph_count": 2}), encoding="utf-8")
    manifest = {"path": str(manifest_path), "fingerprint": "fp", "paragraph_count": 2}
    digest = hashlib.sha256(b"musique_alpha\nmusique_beta").hexdigest()

    class Neo4j:
        async def execute_query(self, query, parameters=None):
            if "RAGIndexSnapshot" in query:
                return [
                    {
                        "status": "complete",
                        "fingerprint": "fp",
                        "paragraph_count": 2,
                        "source_count": 2,
                        "source_set_sha256": digest,
                        "snapshot_version": 1,
                    }
                ]
            return [{"source": "musique_alpha.txt"}, {"source": "musique_beta.txt"}]

    class Engine:
        chunk_label = "PR_musique_Chunk"
        neo4j = Neo4j()

    verified = await _verify_active_index_snapshot(Engine(), "prehop", "musique", manifest, strict=True)
    assert verified["status"] == "matched"

    class BadNeo4j(Neo4j):
        async def execute_query(self, query, parameters=None):
            if "RAGIndexSnapshot" in query:
                return await super().execute_query(query, parameters)
            return [{"source": "musique_alpha.txt"}]

    class BadEngine:
        chunk_label = "PR_musique_Chunk"
        neo4j = BadNeo4j()

    with pytest.raises(RuntimeError, match="Active index integrity gate failed"):
        await _verify_active_index_snapshot(BadEngine(), "prehop", "musique", manifest, strict=True)
    exploratory = await _verify_active_index_snapshot(BadEngine(), "prehop", "musique", manifest, strict=False)
    assert exploratory["status"] == "mismatch_exploratory"


def test_ms_snapshot_metadata_is_sidecar_and_requires_actual_document_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(ms_official_indexer, "_OUTPUT_ROOT", tmp_path)
    output_dir = ms_official_indexer.output_dir_for("musique")
    output_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "pandas.read_parquet",
        lambda _path: pd.DataFrame({"title": ["musique_alpha.txt", "musique_beta.txt"]}),
    )

    payload = ms_official_indexer._verify_and_publish_snapshot(
        "musique",
        ["musique_alpha", "musique_beta"],
        {"fingerprint": "fp", "paragraph_count": 2},
    )

    assert payload["status"] == "complete"
    persisted = json.loads(ms_official_indexer.snapshot_metadata_path("musique").read_text(encoding="utf-8"))
    assert persisted["source_set_sha256"] == payload["source_set_sha256"]


def test_musique_corpus_keeps_same_title_distinct_paragraphs(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_musique, "CORPUS_DIR", tmp_path / "corpus")
    rows = [
        {
            "id": "2hop__corpus",
            "paragraphs": [
                {"title": "Repeated", "paragraph_text": "first body"},
                {"title": "Repeated", "paragraph_text": "second body"},
            ],
        }
    ]

    mapping = prepare_musique.build_corpus(rows)

    assert len(mapping) == 2
    files = list((tmp_path / "corpus").glob("*.txt"))
    assert len(files) == 2
    assert all("Paragraph-ID: musique:" in path.read_text(encoding="utf-8") for path in files)


def test_musique_corpus_publish_is_verified_and_reproducibly_manifested(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(prepare_musique, "CORPUS_DIR", corpus_dir)
    rows = [
        {
            "id": "2hop__manifest",
            "answerable": True,
            "paragraphs": [
                {"title": "Repeated", "paragraph_text": "first", "is_supporting": True},
                {"title": "Repeated", "paragraph_text": "second", "is_supporting": True},
            ],
        }
    ]

    mapping = prepare_musique.build_corpus(rows)
    manifest = json.loads((corpus_dir / prepare_musique.CORPUS_MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert manifest["paragraph_count"] == 2
    assert manifest["gold_supporting_paragraph_count"] == 2
    assert manifest["gold_supporting_paragraph_coverage"] == 1.0
    assert manifest["query_ids_sha256"] == prepare_musique.query_ids_sha256(rows)
    assert manifest == prepare_musique.build_corpus_integrity(rows, mapping, corpus_dir)


def test_musique_corpus_failed_validation_preserves_existing_target(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    existing = corpus_dir / "existing.txt"
    existing.write_text("do not replace", encoding="utf-8")
    monkeypatch.setattr(prepare_musique, "CORPUS_DIR", corpus_dir)
    rows = [{"answerable": True, "paragraphs": [{"title": "", "paragraph_text": "gold", "is_supporting": True}]}]

    with pytest.raises(ValueError, match="gold paragraph"):
        prepare_musique.build_corpus(rows)

    assert existing.read_text(encoding="utf-8") == "do not replace"
    assert not list(tmp_path.glob(".corpus.tmp-*"))


def test_paired_bootstrap_excludes_negative_sentinel_and_runtime_errors():
    prehop = {
        "valid": {"paragraph_support_f1": 1.0, "expected_sources": {"paragraph_ids": ["p"]}},
        "sentinel": {"paragraph_support_f1": -1.0, "expected_sources": {"paragraph_ids": ["p"]}},
        "failed": {"paragraph_support_f1": 0.0, "error": "boom", "expected_sources": {"paragraph_ids": ["p"]}},
    }
    baseline = {
        "valid": {"paragraph_support_f1": 0.5, "expected_sources": {"paragraph_ids": ["p"]}},
        "sentinel": {"paragraph_support_f1": -1.0, "expected_sources": {"paragraph_ids": ["p"]}},
        "failed": {"paragraph_support_f1": 1.0, "expected_sources": {"paragraph_ids": ["p"]}},
    }

    assert _paired(prehop, baseline, "paragraph_support_f1").tolist() == [0.5]


def test_aggregates_exclude_runtime_errors_and_record_eligible_count():
    summary = {
        "details": [
            {"category": "2hop", "answer_em": 1.0},
            {"category": "2hop", "answer_em": 0.0, "error": "runtime failure"},
            {"category": "2hop", "answer_em": -1.0},
        ]
    }

    _recompute_aggregates(summary)

    assert summary["avg_answer_em"] == 1.0
    assert summary["eligible_answer_em_count"] == 1
    assert summary["category_summaries"]["2hop"]["eligible_answer_em_count"] == 1


def test_musique_identity_is_content_stable_and_title_sensitive():
    same = prepare_musique.paragraph_identity("Title", "Body")
    assert same == prepare_musique.paragraph_identity("Title", "Body")
    assert same != prepare_musique.paragraph_identity("Title", "Other body")
    assert same != prepare_musique.paragraph_identity("Other title", "Body")


def test_musique_query_preserves_local_idx_to_global_identity_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_musique, "QUERIES_PATH", tmp_path / "queries.json")
    rows = [
        {
            "id": "2hop__example",
            "answer": "answer",
            "question": "question",
            "answerable": True,
            "paragraphs": [
                {"idx": 3, "title": "Repeated", "paragraph_text": "first", "is_supporting": True},
                {"idx": 7, "title": "Repeated", "paragraph_text": "second", "is_supporting": True},
            ],
        }
    ]

    query = prepare_musique.build_queries(rows)[0]

    assert query["evidence_paragraph_indices"] == [3, 7]
    assert len(query["evidence_paragraph_ids"]) == 2
    assert query["evidence_paragraphs"] == [
        {"idx": 3, "paragraph_id": prepare_musique.paragraph_identity("Repeated", "first")},
        {"idx": 7, "paragraph_id": prepare_musique.paragraph_identity("Repeated", "second")},
    ]


def test_paragraph_identity_header_is_not_indexed_as_evidence_text():
    parsed = parse_pages_offline(
        "musique_aabbccddeeff0011.txt",
        "Title: Repeated\nParagraph-ID: musique:aabbccddeeff0011\n\nActual evidence.",
    )

    assert parsed["paragraph_id"] == "musique:aabbccddeeff0011"
    assert parsed["pages"] == [{"num": 1, "content": "Actual evidence."}]


def test_stale_musique_manifest_fails_before_benchmark_execution():
    stale = [{"_id": "musique_stale", "dataset": "musique", "query": "q", "ground_truth": "a"}]

    with pytest.raises(ValueError, match="evidence_paragraph_ids"):
        _validate_benchmark_data(stale, "stale.json")


def test_judge_disabled_does_not_block_deterministic_completion(tmp_path):
    summary = {
        "dataset": "MuSiQue",
        "judge_enabled": False,
        "total_queries": 1,
        "details": [
            {
                "answer": "Final Answer: alias",
                "question_type": "2hop",
                "answer_em": 1.0,
                "llm_judge_score": -1.0,
                "groundedness": -1.0,
                "hallucination": -1.0,
            }
        ],
    }
    _apply_judge_label(summary["details"][0])
    _recompute_aggregates(summary)
    _update_summary_status(summary)

    assert summary["status"] == "completed"
    assert summary["correct_rate"] == 1.0
    _assert_benchmark_complete(summary, tmp_path / "result.json")


def test_full_musique_requires_support_eligibility_and_gold_ids(tmp_path):
    summary = {
        "dataset": "MuSiQue",
        "judge_enabled": False,
        "evaluation_scope": "full_benchmark",
        "official_split_expected_queries": 1,
        "eligible_primary_answer_score_count": 1,
        "eligible_paragraph_support_f1_count": 0,
        "details": [{"expected_sources": {"paragraph_ids": []}}],
    }

    with pytest.raises(RuntimeError, match="MuSiQue"):
        _assert_benchmark_complete(summary, tmp_path / "result.json")


def test_multi_seed_aggregate_excludes_ineligible_seeds_and_all_ineligible_metrics():
    summaries = [
        {
            "avg_answer_em": 1.0,
            "eligible_answer_em_count": 2,
            "avg_llm_judge_score": 0.0,
            "eligible_llm_judge_score_count": 0,
            "category_summaries": {
                "2hop": {
                    "avg_answer_em": 1.0,
                    "eligible_answer_em_count": 2,
                    "avg_groundedness": 0.0,
                    "eligible_groundedness_count": 0,
                }
            },
        },
        {
            "avg_answer_em": 0.0,
            "eligible_answer_em_count": 0,
            "avg_llm_judge_score": 0.0,
            "eligible_llm_judge_score_count": 0,
            "category_summaries": {
                "2hop": {
                    "avg_answer_em": 0.0,
                    "eligible_answer_em_count": 0,
                    "avg_groundedness": 0.0,
                    "eligible_groundedness_count": 0,
                }
            },
        },
    ]

    aggregate = _aggregate_seed_summaries(summaries)

    assert aggregate["overall"]["avg_answer_em"] == {
        "mean": 1.0,
        "std": 0.0,
        "ci95_low": 1.0,
        "ci95_high": 1.0,
        "n": 1,
    }
    assert "avg_llm_judge_score" not in aggregate["overall"]
    assert aggregate["categories"]["2hop"]["avg_answer_em"]["n"] == 1
    assert "avg_groundedness" not in aggregate["categories"]["2hop"]


def test_self_judge_requires_explicit_debug_override():
    with pytest.raises(RuntimeError, match="must be independent"):
        _judge_independence("same-model", "same-model", "generation-model", False)

    assert _judge_independence("same-model", "same-model", "generation-model", True) == (False, True)
    assert _judge_independence("independent-judge", "generation-model", "generation-model", False) == (True, False)


def _artifact(strategy: str, *, query_id: str = "q-1") -> dict:
    return {
        "strategy": strategy,
        "dataset": "MuSiQue",
        "corpus_tag": "musique",
        "evaluation_scope": "full_benchmark",
        "status": "completed",
        "corpus_manifest_fingerprint": "corpus-fp",
        "index_manifest_fingerprint": "corpus-fp",
        "corpus_index_fingerprint_status": "matched",
        "details": [{"query_id": query_id, "query": "same text", "answer_em": 1.0}],
    }


def test_paired_bootstrap_loads_stable_ids_and_validates_artifact_identity(tmp_path):
    treatment_path = tmp_path / "prehop.json"
    baseline_path = tmp_path / "naive.json"
    treatment_path.write_text(json.dumps(_artifact("prehop")), encoding="utf-8")
    baseline_path.write_text(json.dumps(_artifact("naive")), encoding="utf-8")

    _, _, treatment, treatment_rows = _load(str(treatment_path))
    _, _, baseline, baseline_rows = _load(str(baseline_path))
    _validate_artifact_pair(treatment, baseline)

    assert list(treatment_rows) == ["q-1"]
    assert _paired(treatment_rows, baseline_rows, "answer_em").tolist() == [0.0]


def test_paired_bootstrap_rejects_incompatible_or_legacy_artifacts(tmp_path):
    legacy_path = tmp_path / "legacy.json"
    legacy = _artifact("naive")
    legacy["details"][0].pop("query_id")
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(ValueError, match="query_id"):
        _load(str(legacy_path))

    treatment = _artifact("prehop")
    baseline = _artifact("naive")
    baseline["corpus_manifest_fingerprint"] = "different"
    with pytest.raises(ValueError, match="fingerprint"):
        _validate_artifact_pair(treatment, baseline)

    baseline = _artifact("naive")
    baseline["evaluation_scope"] = "subset_exploratory"
    with pytest.raises(ValueError, match="allow-exploratory"):
        _validate_artifact_pair(treatment, baseline)

    baseline = _artifact("naive", query_id="different-query")
    with pytest.raises(ValueError, match="query ID sets differ"):
        _validate_artifact_pair(treatment, baseline)
