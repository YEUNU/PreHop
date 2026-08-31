import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.combine_frozen_synthesis import combine_frozen_synthesis
from utils.metrics import calculate_answer_metrics, calculate_musique_support_metrics


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    query = {
        "_id": "musique_2hop__1_2",
        "query": "What is the answer?",
        "ground_truth": "target answer",
        "answer_aliases": [],
        "evidence_paragraph_ids": ["musique:abcdef123456"],
        "category": "2hop",
        "question_type": "2hop",
    }
    queries_path = tmp_path / "musique_queries.json"
    queries_path.write_text(json.dumps([query]), encoding="utf-8")
    query_digest = hashlib.sha256(query["_id"].encode()).hexdigest()
    source = {
        "doc": "Example",
        "source": "musique_abcdef123456.txt",
        "text": "Paragraph-ID: musique:abcdef123456\nThe target answer.",
    }
    support = calculate_musique_support_metrics([source], query["evidence_paragraph_ids"])
    detail = {
        "query_id": query["_id"],
        "query": query["query"],
        "ground_truth": query["ground_truth"],
        "category": query["category"],
        "question_type": query["question_type"],
        "expected_sources": {"paragraph_ids": query["evidence_paragraph_ids"]},
        "retrieved_sources": [source],
        "error": "",
        **support,
    }
    common = {
        "strategy": "hoprag",
        "corpus_tag": "musique",
        "dataset": "MuSiQue",
        "evaluation_scope": "full_benchmark",
        "official_split_expected_queries": 1,
        "manifest_queries_count": 1,
        "evaluated_queries_count": 1,
        "evaluated_query_ids_sha256": query_digest,
        "status": "completed",
    }
    raw_path = tmp_path / "baseline.json"
    raw_path.write_text(json.dumps({**common, "details": [detail]}), encoding="utf-8")
    summary = {
        **common,
        "corpus_manifest_path": "data/musique_corpus/corpus_manifest.json",
        "corpus_manifest_fingerprint": "corpus-fingerprint",
        "corpus_manifest_paragraph_count": 1,
        "index_manifest_stats_path": "index.json",
        "index_manifest_fingerprint": "corpus-fingerprint",
        "index_manifest_status": "complete",
        "corpus_index_fingerprint_status": "matched",
        "active_index_snapshot": {"status": "matched"},
        **{f"avg_{field}": value for field, value in support.items()},
        **{f"eligible_{field}_count": 1 for field in support},
    }
    summary_path = tmp_path / "baseline.summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    answer = "target answer"
    answer_metrics = calculate_answer_metrics(answer, query["ground_truth"], question_type=query["question_type"])
    synthesis_path = tmp_path / "answers.json"
    synthesis_path.write_text(
        json.dumps(
            {
                "hoprag": [
                    {
                        "query_id": query["_id"],
                        "answer": answer,
                        "synthesis_seconds": 1.5,
                        "official_answer_em": answer_metrics["official_answer_em"],
                        "official_answer_f1": answer_metrics["official_answer_f1"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prompt_path = tmp_path / "shared.py"
    prompt_path.write_text("prompt = 'current'\n", encoding="utf-8")
    os.utime(prompt_path, ns=(1_000_000_000, 1_000_000_000))
    os.utime(synthesis_path, ns=(2_000_000_000, 2_000_000_000))
    return summary_path, queries_path, synthesis_path, prompt_path


def test_combine_frozen_synthesis_recomputes_both_metric_groups(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary_path, queries_path, synthesis_path, prompt_path = _write_fixture(tmp_path)
    output_path = tmp_path / "combined" / "summary.json"

    combined = combine_frozen_synthesis(
        summary_path,
        queries_path,
        synthesis_path,
        "hoprag",
        prompt_path,
        output_path,
    )

    assert combined["avg_official_answer_em"] == 1.0
    assert combined["avg_paragraph_support_f1"] == 1.0
    assert combined["combined_result_provenance"]["checks"]["answer_metrics_recomputed"] is True
    assert json.loads(output_path.read_text())["status"] == "completed"


def test_combine_frozen_synthesis_rejects_answer_metric_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary_path, queries_path, synthesis_path, prompt_path = _write_fixture(tmp_path)
    payload = json.loads(synthesis_path.read_text())
    payload["hoprag"][0]["official_answer_em"] = 0.0
    synthesis_path.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(synthesis_path, ns=(2_000_000_000, 2_000_000_000))

    with pytest.raises(ValueError, match="stored official_answer_em differs"):
        combine_frozen_synthesis(
            summary_path,
            queries_path,
            synthesis_path,
            "hoprag",
            prompt_path,
            tmp_path / "combined.json",
        )
