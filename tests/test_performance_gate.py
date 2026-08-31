import pytest

from scripts.performance_gate import evaluate_gate


def _artifact(strategy: str, dataset: str, scope: str = "full_benchmark") -> dict:
    artifact = {
        "strategy": strategy,
        "dataset": dataset,
        "corpus_tag": dataset.casefold(),
        "status": "completed",
        "evaluation_scope": scope,
        "corpus_manifest_fingerprint": "corpus",
        "corpus_index_fingerprint_status": "matched",
        "evaluated_query_ids_sha256": "queries",
        "evaluated_queries_count": 10,
        "_path": f"{strategy}.json",
    }
    if dataset == "MultiHop-RAG":
        artifact.update(
            {
                "avg_official_hits@4": 0.5,
                "avg_official_hits@10": 0.6,
                "avg_official_mrr@10": 0.4,
                "avg_official_map@10": 0.3,
            }
        )
    else:
        artifact.update(
            {
                "avg_official_answer_em": 0.2,
                "avg_official_answer_f1": 0.3,
                "avg_paragraph_support_precision": 0.1,
                "avg_paragraph_support_recall": 0.6,
                "avg_paragraph_support_f1": 0.17,
            }
        )
    return artifact


def test_performance_gate_uses_strongest_baseline_per_metric():
    prehop = _artifact("prehop", "MultiHop-RAG")
    prehop.update(
        {
            "avg_official_hits@4": 0.66,
            "avg_official_hits@10": 0.77,
            "avg_official_mrr@10": 0.55,
            "avg_official_map@10": 0.44,
        }
    )
    naive = _artifact("naive", "MultiHop-RAG")
    hoprag = _artifact("hoprag", "MultiHop-RAG")
    hoprag["avg_official_hits@4"] = 0.6
    naive["avg_official_hits@10"] = 0.7

    report = evaluate_gate(prehop, [naive, hoprag])

    assert report["pass"] is True
    rows = {row["metric"]: row for row in report["metrics"]}
    assert rows["Hits@4"]["strongest_strategy"] == "hoprag"
    assert rows["Hits@10"]["strongest_strategy"] == "naive"
    assert rows["Hits@4"]["required"] == pytest.approx(0.66)


def test_performance_gate_fails_when_one_official_metric_misses_margin():
    prehop = _artifact("prehop", "MuSiQue")
    baseline = _artifact("hoprag", "MuSiQue")
    prehop.update(
        {
            "avg_official_answer_em": 0.22,
            "avg_official_answer_f1": 0.33,
            "avg_paragraph_support_precision": 0.11,
            "avg_paragraph_support_recall": 0.659,
            "avg_paragraph_support_f1": 0.187,
        }
    )

    report = evaluate_gate(prehop, [baseline])

    assert report["pass"] is False
    failed = [row["metric"] for row in report["metrics"] if not row["pass"]]
    assert failed == ["Support recall"]


def test_performance_gate_rejects_exploratory_or_incompatible_artifacts():
    prehop = _artifact("prehop", "MuSiQue", scope="sample_exploratory")
    baseline = _artifact("hoprag", "MuSiQue", scope="sample_exploratory")

    with pytest.raises(ValueError, match="full_benchmark"):
        evaluate_gate(prehop, [baseline])
    report = evaluate_gate(prehop, [baseline], allow_exploratory=True)
    assert report["paper_eligible"] is False

    baseline["evaluated_query_ids_sha256"] = "different"
    with pytest.raises(ValueError, match="incompatible"):
        evaluate_gate(prehop, [baseline], allow_exploratory=True)
