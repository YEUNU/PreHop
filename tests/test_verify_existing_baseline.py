import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_existing_baseline import verify_existing_baseline
from utils.metrics import calculate_retrieval_ranking_metrics


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    corpus_dir = tmp_path / "multihoprag_corpus"
    corpus_dir.mkdir()
    source = corpus_dir / "source_one.txt"
    source.write_text("The target fact appears here.\n", encoding="utf-8")

    query = {
        "_id": "multihoprag_00000",
        "query": "Where is the target fact?",
        "ground_truth": "here",
        "evidence_docs": ["Source One"],
        "evidence_facts": ["The target fact appears here."],
        "evidence_doc": "Source One",
        "evidence_page": None,
        "evidence_text": "The target fact appears here.",
        "category": "inference_query",
        "question_type": "inference_query",
        "dataset": "multihoprag",
    }
    queries = [query]
    queries_path = tmp_path / "multihoprag_queries.json"
    queries_path.write_text(json.dumps(queries), encoding="utf-8")
    query_record = json.dumps(query, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest_payload = {
        "schema_version": 1,
        "paragraph_count": 1,
        "source_ids_sha256": _sha256_lines(["source_one"]),
        "corpus_files_sha256": _sha256_lines([f"source_one.txt\0{hashlib.sha256(source.read_bytes()).hexdigest()}"]),
        "query_ids_sha256": _sha256_lines(["multihoprag_00000"]),
        "query_records_sha256": _sha256_lines([query_record]),
    }
    manifest = {
        **manifest_payload,
        "fingerprint": hashlib.sha256(
            json.dumps(manifest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    manifest_path = corpus_dir / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    retrieved = [
        {
            "doc": "Source One",
            "source": "source_one",
            "page": 0,
            "text": "The target fact appears here.",
            "sent_id": 0,
        }
    ]
    metrics = calculate_retrieval_ranking_metrics(retrieved, query["evidence_facts"])
    detail = {
        "query_id": query["_id"],
        "query": query["query"],
        "category": query["category"],
        "question_type": query["question_type"],
        "ground_truth": query["ground_truth"],
        "expected_sources": {
            "docs": query["evidence_docs"],
            "facts": query["evidence_facts"],
            "paragraph_ids": [],
        },
        "retrieved_sources": retrieved,
        "error": "",
        **metrics,
    }
    stats_path = tmp_path / "index_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "strategy": "hoprag",
                "corpus_tag": "multihoprag",
                "dataset_path": str(corpus_dir),
                "status": "complete",
                "timing_seconds": {
                    "active_snapshot_verified": 1.0,
                    "active_snapshot_source_count": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    common = {
        "strategy": "hoprag",
        "corpus_tag": "multihoprag",
        "dataset": "MultiHop-RAG",
        "evaluation_scope": "full_benchmark",
        "official_split_expected_queries": 1,
        "evaluated_queries_count": 1,
        "evaluated_query_ids_sha256": manifest["query_ids_sha256"],
        "status": "completed",
    }
    raw_path = tmp_path / "baseline.json"
    raw_path.write_text(json.dumps({**common, "details": [detail]}), encoding="utf-8")
    summary = {
        **common,
        "index_manifest_stats_path": str(stats_path),
        "corpus_manifest_path": None,
        "corpus_manifest_fingerprint": None,
        "corpus_manifest_paragraph_count": None,
        "index_manifest_fingerprint": None,
        "index_manifest_status": "complete",
        "corpus_index_fingerprint_status": "manifest_absent",
        "active_index_snapshot": {"status": "manifest_absent"},
    }
    for field, value in metrics.items():
        if field in {
            "official_hits@4",
            "official_hits@10",
            "official_mrr@10",
            "official_map@10",
        }:
            summary[f"avg_{field}"] = value
            summary[f"eligible_{field}_count"] = 1
    summary_path = tmp_path / "baseline.summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path, queries_path, manifest_path


def test_verify_existing_baseline_writes_derived_matched_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary_path, queries_path, manifest_path = _write_fixture(tmp_path)
    original = summary_path.read_bytes()
    output_path = tmp_path / "verified" / "baseline.summary.json"

    verified = verify_existing_baseline(summary_path, queries_path, manifest_path, output_path)

    assert summary_path.read_bytes() == original
    assert verified["corpus_index_fingerprint_status"] == "matched"
    assert verified["compatibility_verification"]["checks"]["official_metrics_recomputed"] is True
    assert json.loads(output_path.read_text())["active_index_snapshot"]["source_count"] == 1


def test_verify_existing_baseline_rejects_retrieved_text_outside_current_corpus(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary_path, queries_path, manifest_path = _write_fixture(tmp_path)
    raw_path = tmp_path / "baseline.json"
    raw = json.loads(raw_path.read_text())
    raw["details"][0]["retrieved_sources"][0]["text"] = "tampered passage"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="retrieved text is absent"):
        verify_existing_baseline(
            summary_path,
            queries_path,
            manifest_path,
            tmp_path / "verified.summary.json",
        )
