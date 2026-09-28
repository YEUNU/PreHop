import json
from pathlib import Path

import pytest

from scripts.export_official_results import export, export_comparison
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


def save_complete(tmp_path, name, result):
    result = {**result, "strategy": name, "status": "completed",
              "evaluation_scope": "full_benchmark", "total_queries": len(result["details"])}
    path = tmp_path / name / "result.json"
    path.parent.mkdir()
    path.write_text(json.dumps(result))
    return path


def test_comparison_keeps_dataset_metrics_populations_and_sources_separate(tmp_path):
    import csv

    first = save_complete(tmp_path, "prehop", mhr_result())
    second = save_complete(tmp_path, "hoprag", mhr_result())
    hotpot = {"dataset": "HotpotQA", "details": [
        {"query_id": "release-0", **{"hotpot_" + k: v for k, v in
         score("Alpha", [["A", 0]], "Alpha", [["A", 0]]).items()}}]}
    third = save_complete(tmp_path, "hotpot", hotpot)
    before = {p: p.read_bytes() for p in [first, second, third]}
    out = tmp_path / "comparison"
    written = export_comparison([first, second, third], out)

    mhr = json.loads((out / "multihoprag/comparison.json").read_text())
    hp = json.loads((out / "hotpotqa/comparison.json").read_text())
    assert len(mhr["results"]) == 2 and len(hp["results"]) == 1
    assert mhr["results"][0]["qa"]["overall"]["rows"] == 3
    assert mhr["results"][0]["retrieval"]["rows"] == 2
    assert set(hp["results"][0]["metrics"]) == set(METRICS)
    assert "normalized_answer_em" not in json.dumps(mhr)
    assert "MAP@10" not in json.dumps(hp)
    with (out / "multihoprag/comparison.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [r["strategy"] for r in csv_rows] == ["prehop", "hoprag"]
    assert float(csv_rows[0]["qa.overall.accuracy"]) == 1 / 3
    assert len(csv_rows[0]["source_sha256"]) == 64
    assert all(p.exists() for p in written)
    assert len(written) == 7  # Same input filename cannot overwrite another method.
    assert not list(out.rglob("*.diagnostics.json"))
    assert all(p.read_bytes() == raw for p, raw in before.items())


@pytest.mark.parametrize("change, message", [
    ({"status": "in_progress"}, "complete result"),
    ({"total_queries": 4}, "complete result"),
    ({"evaluation_scope": "subset_exploratory"}, "complete result"),
    ({"strategy": "prehop"}, "one final result"),
])
def test_comparison_rejects_partial_or_duplicate_final_sources(tmp_path, change, message):
    first = save_complete(tmp_path, "prehop", mhr_result())
    second = save_complete(tmp_path, "hoprag", mhr_result())
    second.write_text(json.dumps({**json.loads(second.read_text()), **change}))
    with pytest.raises(ValueError, match=message):
        export_comparison([first, second], tmp_path / "comparison")
    assert not (tmp_path / "comparison").exists()


def test_comparison_rejects_different_questions_with_the_same_row_count(tmp_path):
    first = save_complete(tmp_path, "prehop", mhr_result())
    result = mhr_result()
    result["details"][0]["query_id"] = "different-question"
    second = save_complete(tmp_path, "hoprag", result)
    with pytest.raises(ValueError, match="Mismatched question population"):
        export_comparison([first, second], tmp_path / "comparison")


def test_comparison_matches_annotations_across_historical_storage_metadata(tmp_path):
    first_result = mhr_result()
    second_result = mhr_result()
    for row in first_result["details"]:
        row["original_query_id"] = row["query_id"]
        row["expected_sources"] = {"facts": ["gold fact"], "supporting_facts": []}
    for row in second_result["details"]:
        row["expected_sources"] = {"facts": ["gold fact"], "paragraph_ids": []}
    first = save_complete(tmp_path, "prehop", first_result)
    second = save_complete(tmp_path, "hoprag", second_result)
    assert export_comparison([first, second], tmp_path / "comparison")
    second_result["details"][0]["expected_sources"]["facts"] = ["different gold"]
    payload = json.loads(second.read_text())
    payload["details"] = second_result["details"]
    second.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Mismatched question population"):
        export_comparison([first, second], tmp_path / "other")
