import asyncio
import hashlib
import json
import os

import pandas as pd
import pytest

from cli.benchmark import (
    OFFICIAL_QUERY_ID_DIGESTS,
    _aggregate_seed_summaries,
    _apply_judge_label,
    _benchmark_checkpoint_due,
    _evaluation_scope,
    _latest_index_manifest_metadata,
    _order_benchmark_rows,
    _recompute_aggregates,
    _resume_benchmark_rows,
    _update_summary_status,
)
from cli.index import _load_source_metadata
from models.hoprag import official_indexer as hop_official_indexer
from models.ms_graphrag import official_indexer as ms_official_indexer
from models.prehop.indexing.chunking import parse_pages_offline
from scripts.datasets import prepare_multihoprag, refresh_sample_records
from scripts.paired_bootstrap import (
    MULTIHOPRAG_METRICS,
    _dataset_marker,
    _load,
    _load_excluded_query_ids,
    _paired,
    _validate_artifact_pair,
)


def test_refresh_sample_records_preserves_ids_and_uses_current_annotations():
    full = [
        {"_id": "q1", "query": "new one", "evidence_docs": ["p1"]},
        {"_id": "q2", "query": "new two", "evidence_docs": ["p2"]},
    ]
    sample = [{"_id": "q2", "query": "old two"}, {"_id": "q1", "query": "old one"}]

    refreshed = refresh_sample_records.refresh_records(full, sample)

    assert [row["_id"] for row in refreshed] == ["q2", "q1"]
    assert refreshed[0]["query"] == "new two"
    assert refreshed[0]["evidence_docs"] == ["p2"]


def test_multihoprag_manifest_binds_corpus_and_query_content(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    source = corpus_dir / "Alpha.txt"
    source.write_text("Title: Alpha\n\nEvidence", encoding="utf-8")
    queries = [{"_id": "multihoprag_00000", "query": "Question", "dataset": "multihoprag"}]

    first = prepare_multihoprag.build_corpus_manifest(corpus_dir, queries)
    assert first == prepare_multihoprag.build_corpus_manifest(corpus_dir, queries)
    assert first["paragraph_count"] == 1
    assert first["query_ids_sha256"] == prepare_multihoprag.query_ids_sha256(queries)

    source.write_text("Title: Alpha\n\nChanged evidence", encoding="utf-8")
    assert prepare_multihoprag.build_corpus_manifest(corpus_dir, queries)["fingerprint"] != first["fingerprint"]
    changed_queries = [{**queries[0], "query": "Changed question"}]
    assert prepare_multihoprag.build_corpus_manifest(corpus_dir, changed_queries)["fingerprint"] != first["fingerprint"]


def test_multihoprag_source_metadata_round_trip(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    monkeypatch.setattr(prepare_multihoprag, "CORPUS_DIR", corpus_dir)
    payload = prepare_multihoprag.write_source_metadata(
        [
            {
                "title": "Article",
                "body": "Evidence.",
                "author": "Author",
                "source": "Publisher",
                "published_at": "2026-08-30T00:00:00Z",
                "category": "news",
                "url": "https://example.com/article",
            }
        ]
    )

    records, digest = _load_source_metadata(corpus_dir)

    assert payload["records"]["Article.txt"]["publisher"] == "Publisher"
    assert records["Article.txt"] == payload["records"]["Article.txt"]
    assert (
        digest == hashlib.sha256((corpus_dir / prepare_multihoprag.SOURCE_METADATA_FILENAME).read_bytes()).hexdigest()
    )


def test_benchmark_checkpoint_interval_is_bounded_and_always_writes_final_state():
    assert not _benchmark_checkpoint_due(1, 25, 10)
    assert _benchmark_checkpoint_due(10, 25, 10)
    assert _benchmark_checkpoint_due(25, 25, 10)


def _write_resume_fixture(tmp_path, rows, *, status="in_progress", strategy="hoprag"):
    result_file = tmp_path / "hoprag_multihoprag.json"
    result_file.write_text(
        json.dumps({"status": status, "strategy": strategy, "details": rows}),
        encoding="utf-8",
    )
    trace_rows = [
        {
            "idx": row.get("idx", idx),
            "query_id": row.get("query_id", ""),
            "query": row["query"],
            "interaction_trace": [{"step": f"trace-{idx}"}],
        }
        for idx, row in enumerate(rows, start=1)
    ]
    result_file.with_name("hoprag_multihoprag.traces.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trace_rows),
        encoding="utf-8",
    )
    return result_file


def test_benchmark_resume_retains_terminal_errors(tmp_path):
    benchmark_data = [
        {"_id": "q1", "query": "first"},
        {"_id": "q2", "query": "second"},
        {"_id": "q3", "query": "third"},
    ]
    result_file = _write_resume_fixture(
        tmp_path,
        [
            {"query_id": "q1", "query": "first", "answer": "ok"},
            {"query_id": "q2", "query": "second", "error": "interrupted"},
        ],
    )

    retained, metadata = _resume_benchmark_rows(
        result_file,
        benchmark_data,
        {"strategy": "hoprag"},
        judge_enabled=False,
    )

    assert [row["query_id"] for row in retained] == ["q1", "q2"]
    assert retained[0]["interaction_trace"] == [{"step": "trace-1"}]
    assert metadata["initial_rows"] == 2
    assert metadata["retained_rows"] == 2
    assert metadata["rerun_error_rows"] == 0


@pytest.mark.asyncio
async def test_concurrent_benchmark_rows_are_checkpointed_in_input_order():
    rows = []
    lock = asyncio.Lock()

    async def complete(idx, delay):
        await asyncio.sleep(delay)
        async with lock:
            rows.append({"idx": idx, "query_id": f"q{idx}"})
            _order_benchmark_rows(rows)

    await asyncio.gather(complete(1, 0.03), complete(2, 0.02), complete(3, 0.01))

    assert [row["query_id"] for row in rows] == ["q1", "q2", "q3"]


def test_benchmark_resume_reorders_partial_concurrent_checkpoint_by_manifest(tmp_path):
    benchmark_data = [
        {"_id": "q1", "query": "first"},
        {"_id": "q2", "query": "second"},
        {"_id": "q3", "query": "third"},
    ]
    result_file = _write_resume_fixture(
        tmp_path,
        [
            {"idx": 3, "query_id": "q3", "query": "third", "answer": "three"},
            {"idx": 1, "query_id": "q1", "query": "first", "answer": "one"},
        ],
    )

    retained, _ = _resume_benchmark_rows(
        result_file,
        benchmark_data,
        {"strategy": "hoprag"},
        judge_enabled=False,
    )

    assert [(row["idx"], row["query_id"]) for row in retained] == [(1, "q1"), (3, "q3")]


def test_benchmark_resume_migrates_behavior_equivalent_candidate_order_metadata(tmp_path):
    result_file = _write_resume_fixture(tmp_path, [{"query_id": "q1", "query": "first"}], strategy="prehop")
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    payload["ablation"] = {
        "graph_hop_depth": 0,
        "rerank_input_order": "search",
        "rerank_shuffle_seed": 0,
    }
    result_file.write_text(json.dumps(payload), encoding="utf-8")

    retained, _metadata = _resume_benchmark_rows(
        result_file,
        [{"_id": "q1", "query": "first"}],
        {
            "strategy": "prehop",
            "ablation": {
                "graph_hop_depth": 0,
                "graph_path_decay": 0.5,
                "candidate_order_input_order": "search",
                "candidate_order_shuffle_seed": 0,
                "final_rank_variant": "fused",
            },
        },
        judge_enabled=False,
    )

    assert [row["query_id"] for row in retained] == ["q1"]


def test_evaluation_scope_uses_actual_evaluated_count_before_filename():
    assert _evaluation_scope(
        "multihoprag", 2556, "custom_sample_name.json", OFFICIAL_QUERY_ID_DIGESTS["multihoprag"]
    ) == ("full_benchmark", 2556)
    assert _evaluation_scope("hotpotqa", 200, "hotpotqa_sample200_queries.json", "subset") == (
        "sample_exploratory",
        7405,
    )
    assert _evaluation_scope("hotpotqa", 200, "hotpotqa_queries.json", "subset") == ("subset_exploratory", 7405)


def test_index_artifact_selection_honors_explicit_path_and_run_identity(tmp_path, monkeypatch):
    selected_path = tmp_path / "chosen.json"
    selected_path.write_text(json.dumps({"run_id": "target", "status": "complete"}), encoding="utf-8")
    monkeypatch.setenv("RAG_RUN_ID", "target")
    monkeypatch.setenv("RAG_INDEX_STATS_PATH", str(selected_path))

    selected = _latest_index_manifest_metadata("prehop", "hotpotqa", tmp_path)
    assert selected["path"] == str(selected_path)

    monkeypatch.setenv("RAG_RUN_ID", "other")
    assert _latest_index_manifest_metadata("prehop", "hotpotqa", tmp_path)["status"] == "invalid"


def test_index_manifest_selection_does_not_cross_prefixing_corpus_tags(tmp_path):
    legacy = tmp_path / "prehop_multihoprag_legacy.json"
    grounded = tmp_path / "prehop_multihoprag_grounded_v1_newer.json"
    legacy.write_text(
        json.dumps({"strategy": "prehop", "corpus_tag": "multihoprag", "status": "complete"}),
        encoding="utf-8",
    )
    grounded.write_text(
        json.dumps({"strategy": "prehop", "corpus_tag": "multihoprag_grounded_v1", "status": "complete"}),
        encoding="utf-8",
    )
    os.utime(legacy, (1, 1))
    os.utime(grounded, (2, 2))

    selected = _latest_index_manifest_metadata("prehop", "multihoprag", tmp_path)

    assert selected["path"] == str(legacy)


def test_ms_snapshot_metadata_is_sidecar_and_requires_actual_document_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(ms_official_indexer, "_OUTPUT_ROOT", tmp_path)
    output_dir = ms_official_indexer.output_dir_for("hotpotqa")
    output_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "pandas.read_parquet",
        lambda _path: pd.DataFrame({"title": ["hotpotqa_alpha.txt", "hotpotqa_beta.txt"]}),
    )

    payload = ms_official_indexer._publish_snapshot(
        "hotpotqa",
        ["hotpotqa_alpha", "hotpotqa_beta"],
        {"fingerprint": "fp", "paragraph_count": 2},
        {"hotpotqa_alpha": "Alpha", "hotpotqa_beta": "Beta"},
    )

    assert payload["status"] == "complete"
    persisted = json.loads(ms_official_indexer.snapshot_metadata_path("hotpotqa").read_text(encoding="utf-8"))
    assert persisted["source_set_sha256"] == payload["source_set_sha256"]
    assert persisted["source_titles_sha256"] == payload["source_titles_sha256"]


def test_hoprag_snapshot_preserves_periods_in_stored_source_ids():
    class Result(list):
        def consume(self):
            return None

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def run(self, query, _parameters=None, **_kwargs):
            if "RETURN DISTINCT n.source AS source" in query:
                return Result([{"source": "Article_about_the_U.S._economy"}])
            return Result()

    class Driver:
        def session(self):
            return Session()

    class Builder:
        driver = Driver()
        label = "HO_multihoprag"

    payload = hop_official_indexer._publish_snapshot(
        Builder(),
        "multihoprag",
        ["Article_about_the_U.S._economy"],
        {"fingerprint": "fp", "paragraph_count": 1},
    )

    assert payload["source_count"] == 1
    assert payload["input_source_count"] == 1
    assert payload["omitted_source_count"] == 0


def test_paired_bootstrap_retains_runtime_failure_as_zero():
    prehop = {
        "valid": {"evidence_doc_f1": 1.0, "expected_sources": {"docs": ["p"]}},
        "sentinel": {"evidence_doc_f1": -1.0, "expected_sources": {"docs": ["p"]}},
        "failed": {"evidence_doc_f1": 0.0, "error": "boom", "expected_sources": {"docs": ["p"]}},
    }
    baseline = {
        "valid": {"evidence_doc_f1": 0.5, "expected_sources": {"docs": ["p"]}},
        "sentinel": {"evidence_doc_f1": -1.0, "expected_sources": {"docs": ["p"]}},
        "failed": {"evidence_doc_f1": 1.0, "expected_sources": {"docs": ["p"]}},
    }

    assert _paired(prehop, baseline, "evidence_doc_f1").tolist() == [0.5, -1.0]


def test_paired_bootstrap_reports_all_document_diagnostics():
    for metrics in (MULTIHOPRAG_METRICS,):
        assert "evidence_doc_precision" in metrics
        assert "evidence_doc_recall" in metrics
        assert "evidence_doc_f1" in metrics


def test_paired_bootstrap_uses_dataset_identity_with_custom_corpus_tag():
    assert _dataset_marker({"dataset": "MultiHop-RAG"}, "multihoprag_grounded_v1") == "multihoprag"


def test_paired_bootstrap_loads_fixed_development_ids(tmp_path):
    path = tmp_path / "development.json"
    path.write_text(json.dumps([{"_id": "q2"}, {"query_id": "q1"}]), encoding="utf-8")

    assert _load_excluded_query_ids(str(path)) == {"q1", "q2"}


def test_aggregates_include_runtime_errors_as_zero():
    summary = {
        "details": [
            {"category": "2hop", "answer_em": 1.0},
            {"category": "2hop", "answer_em": 0.0, "error": "runtime failure"},
            {"category": "2hop", "answer_em": -1.0},
        ]
    }

    _recompute_aggregates(summary)

    assert summary["avg_answer_em"] == 0.5
    assert summary["eligible_answer_em_count"] == 2
    assert summary["category_summaries"]["2hop"]["eligible_answer_em_count"] == 2


def test_paragraph_identity_header_is_not_indexed_as_evidence_text():
    parsed = parse_pages_offline(
        "hotpotqa_aabbccddeeff0011.txt",
        "Title: Repeated\nParagraph-ID: hotpotqa:aabbccddeeff0011\n\nActual evidence.",
    )

    assert parsed["paragraph_id"] == "hotpotqa:aabbccddeeff0011"
    assert parsed["pages"] == [{"num": 1, "content": "Actual evidence."}]


def test_judge_disabled_does_not_block_deterministic_completion(tmp_path):
    summary = {
        "dataset": "HotpotQA",
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

    assert summary["status"] == "completed_unadmitted"
    assert summary["correct_rate"] == 1.0


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


def _artifact(strategy: str, *, query_id: str = "q-1") -> dict:
    return {
        "strategy": strategy,
        "dataset": "HotpotQA",
        "corpus_tag": "hotpotqa",
        "evaluation_scope": "full_benchmark",
        "status": "completed_unadmitted",
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


def test_paired_bootstrap_enforces_query_only_ablation_contract():
    treatment = _artifact("prehop")
    baseline = _artifact("prehop")
    treatment["ablation"] = {
        "question_schema": "linked_v2",
        "continuation_edges_enabled": True,
        "default_top_k": 12,
    }
    baseline["ablation"] = {
        "question_schema": "linked_v2",
        "continuation_edges_enabled": False,
        "default_top_k": 12,
    }

    _validate_artifact_pair(
        treatment,
        baseline,
        expected_ablation_differences={"continuation_edges_enabled"},
    )

    baseline["ablation"]["default_top_k"] = 10
    with pytest.raises(ValueError, match="expected only"):
        _validate_artifact_pair(
            treatment,
            baseline,
            expected_ablation_differences={"continuation_edges_enabled"},
        )

    baseline["ablation"]["default_top_k"] = 12
    treatment["models"] = {"default": "same", "llm_seed": 42}
    baseline["models"] = {"default": "different", "llm_seed": 42}
    with pytest.raises(ValueError, match="controlled metadata"):
        _validate_artifact_pair(
            treatment,
            baseline,
            expected_ablation_differences={"continuation_edges_enabled"},
        )


def test_paired_bootstrap_requires_explicit_index_variant_override():
    treatment = _artifact("prehop")
    baseline = _artifact("prehop")
    baseline["corpus_tag"] = "hotpotqa_linked_v2"

    with pytest.raises(ValueError, match="corpus_tag"):
        _validate_artifact_pair(treatment, baseline)
    _validate_artifact_pair(treatment, baseline, allow_index_variant=True)

    baseline["corpus_manifest_fingerprint"] = "different"
    with pytest.raises(ValueError, match="fingerprint"):
        _validate_artifact_pair(treatment, baseline, allow_index_variant=True)
