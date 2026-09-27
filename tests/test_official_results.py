import json
from pathlib import Path

import pytest

from scripts.export_official_results import export
from utils.hotpotqa import METRICS, score
from utils.official_results import build_reports
from utils.reporting import _write_model_report_artifacts


def mhr_result():
    return {
        "dataset": "MultiHopRAG", "status": "completed", "total_queries": 3,
        "details": [
            {"query_id": "q1", "question_type": "inference_query", "official_qa_accuracy": 1,
             "official_hits@4": 1, "official_hits@10": 1, "official_mrr@10": .5, "official_map@10": .25,
             "answer_em": 0, "answer_f1": .5},
            {"query_id": "q2", "question_type": "null_query", "official_qa_accuracy": 0, "null_refusal": 1},
            {"query_id": "q3", "question_type": "inference_query", "error": "timeout"},
        ],
    }


def test_mhr_official_report_separates_qa_population_and_auxiliary_scores():
    official, diagnostics = build_reports(mhr_result())

    assert official["qa"]["overall"]["accuracy"] == 1 / 3
    assert official["qa"]["overall"]["precision"] == official["qa"]["overall"]["f1"] == 1 / 3
    assert official["qa"]["non_null"]["accuracy"] == .5
    assert official["retrieval"]["rows"] == 2
    assert official["retrieval"]["metrics"]["MAP@10"] == .125
    assert official["failed_rows"] == 1
    assert official["details"][1]["metrics"]["retrieval"] is None
    assert "normalized_answer_em" not in json.dumps(official)
    assert diagnostics["details"][0]["metrics"]["normalized_answer_em"] == 0
    assert diagnostics["details"][1]["metrics"]["null_refusal"] == 1


def test_hotpot_official_report_has_exactly_twelve_official_metrics():
    actual = score("Alpha", [["A", 0], ["A", 1]], "Alpha", [["A", 0]])
    result = {"dataset": "HotpotQA", "details": [
        {"query_id": "release-0", "original_query_id": "original", **{"hotpot_" + k: v for k, v in actual.items()}},
        {"query_id": "release-1", "original_query_id": "original", "error": "timeout"},
    ]}
    official, _ = build_reports(result)

    assert set(official["metrics"]) == set(METRICS)
    assert official["metrics"]["em"] == .5
    assert official["metrics"]["joint_f1"] == pytest.approx(1 / 3)
    assert "MAP@10" not in json.dumps(official)
    assert len(official["details"]) == 2  # Repeated original IDs remain distinct occurrences.


def test_official_report_rejects_unmeasured_scores_and_duplicate_occurrences():
    result = mhr_result()
    result["details"][0]["official_map@10"] = -1
    with pytest.raises(ValueError, match="Missing official score"):
        build_reports(result)
    result = mhr_result()
    result["details"][1]["query_id"] = "q1"
    with pytest.raises(ValueError, match="unique"):
        build_reports(result)


def test_report_writer_and_offline_export_preserve_original_result(tmp_path):
    result = mhr_result()
    source = tmp_path / "prehop_multihoprag.json"
    source.write_text(json.dumps(result))
    before = source.read_bytes()
    _write_model_report_artifacts(result, source)
    first = json.loads((tmp_path / "prehop_multihoprag.official.json").read_text())
    assert first["qa"]["overall"]["accuracy"] == 1 / 3

    paths = export(source, tmp_path / "exported")
    final = json.loads(paths[0].read_text())
    assert final["qa"] == first["qa"]
    assert len(final["source_sha256"]) == 64
    assert source.read_bytes() == before
    assert all(Path(p).exists() for p in paths)
