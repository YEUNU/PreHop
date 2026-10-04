"""Boundary checks for retrospective candidate supply, without model services."""

import pytest

from scripts.analyze_evidence_accessibility import (
    direct_channel_evidence,
    evidence_witnesses,
    indexed,
    initial_coverage_summary,
    paired_scores,
    prefix_ids,
    recorded_channel_ranks,
    row_curves,
    summarize_direct_channels,
)


def test_prefix_preserves_low_ranked_starts_and_never_skips_extras():
    ordered = ["extra1", "start1", "extra2", "extra3", "start2"]
    starts = {"start1", "start2"}
    assert prefix_ids(ordered, starts, 0) == ["start1", "start2"]
    assert prefix_ids(ordered, starts, .5) == ["extra1", "start1", "start2"]
    assert prefix_ids(ordered, starts, 1) == ordered
    assert prefix_ids(["start1"], {"start1"}, .5) == ["start1"]
    with pytest.raises(ValueError, match="every initial start"):
        prefix_ids(["extra1"], starts, .5)
    with pytest.raises(ValueError, match="unique"):
        prefix_ids(["start1", "start1"], {"start1"}, 1)
    with pytest.raises(ValueError, match="fraction"):
        prefix_ids(ordered, starts, 1.1)
    with pytest.raises(ValueError, match="Repeated"):
        indexed([{"id": "a"}, {"id": "a"}])


def test_distinct_passages_can_supply_same_fact_without_duplicate_credit():
    hits = {"s": {0}, "g": {1, 2}, "d1": {1}, "d2": {1}, "d3": {3}}
    curves = row_curves({"s"}, ["g", "s"], ["d1", "d2", "s", "d3"], hits,
                        {0, 1, 2, 3}, [0, .5, 1])
    assert [p["new_graph_unavailable"] for p in curves] == [2, 1, 1]
    assert [p["direct_recall"] for p in curves] == [.25, .5, .75]
    assert all(p["graph_added_in_direct"] == 0 for p in curves)
    endpoint = curves[-1]
    assert endpoint["graph_exclusive_recall"] == .25
    assert endpoint["direct_exclusive_recall"] == .25
    # Union headroom is coverage gain; it is not an answer-quality estimate.
    assert endpoint["direct_recall"] + endpoint["graph_exclusive_recall"] == 1


def test_zero_additions_and_zero_graph_gain_keep_the_question():
    curves = row_curves({"s"}, ["s"], ["s"], {"s": {0}}, {0, 1}, [0, 1])
    assert len(curves) == 2
    assert all(p["new_graph_units"] == 0 and p["candidates"] == 1 for p in curves)
    with pytest.raises(ValueError, match="gold"):
        row_curves({"s"}, ["s"], ["s"], {"s": {0}}, set(), [1])
    with pytest.raises(ValueError, match="only evaluation gold"):
        row_curves({"s"}, ["s"], ["s"], {"s": {0, 99}}, {0}, [1])


def test_passage_novelty_does_not_imply_different_witnesses():
    hits = {"start": {0}, "common": {1}, "graph_noise": set(), "direct_noise": set()}
    result = evidence_witnesses({"start"}, {"start", "common", "graph_noise"},
                               {"start", "common", "direct_noise"}, hits)
    assert result["counts"]["shared_id"] == 1
    assert result["counts"]["distinct_only"] == 0
    assert result["counts"]["graph_only_passages"] == 1
    assert result["counts"]["graph_only_gold_bearing_passages"] == 0


def test_witness_partition_is_exhaustive_despite_duplicate_evidence():
    hits = {"s": {0}, "c": {1, 2}, "g": {2, 3, 4}, "d": {3}, "d2": {3}}
    result = evidence_witnesses({"s"}, {"s", "c", "g"}, {"s", "c", "d", "d2"}, hits)
    counts = result["counts"]
    assert [counts[k] for k in ("shared_id", "distinct_only", "graph_exclusive")] == [2, 1, 1]
    assert counts["shared_id_with_graph_only_witness"] == 1
    assert counts["graph_new_units"] == 4
    row = next(r for r in result["units"] if r["unit"] == 3)
    assert row["graph_witnesses"] == ["g"] and row["direct_witnesses"] == ["d", "d2"]
    assert row["common_witnesses"] == []
    with pytest.raises(ValueError, match="retain"):
        evidence_witnesses({"s"}, {"g"}, {"s", "d"}, hits)


def test_saved_score_sensitivity_retains_failed_rows_and_rejects_missing_metrics():
    queries = {"a": {"original_query_id": "q"}, "b": {"original_query_id": "q"}}
    left = {"a": {"answer_f1": 1}, "b": {"error": "terminal"}}
    right = {"a": {"answer_f1": 0}, "b": {"answer_f1": 1}}
    result = paired_scores(left, right, queries, "answer_f1", seed=42, repeats=50)
    assert result["graph"] == result["direct"] == .5
    assert result["graph_minus_direct"]["clusters"] == 1
    assert result["graph_minus_direct"]["rows"] == 2
    assert result["graph_minus_direct"]["ci95"] == [0, 0]
    left["b"] = {}
    with pytest.raises(ValueError, match="Unavailable required metric"):
        paired_scores(left, right, queries, "answer_f1", seed=42, repeats=50)


def coverage_fixture():
    arms = ("prehop_replay", "direct_tokens")
    rows, scores = [], {arm: {} for arm in arms}
    for qid, original, initial in [("a", "q0", 0), ("b", "q0", 0), ("c", "q1", 1), ("d", "q2", 2)]:
        # Identical final evidence must not affect the initial-coverage groups.
        rows.append({"query_id": qid, "original_query_id": original, "initial_covered": initial, "gold_units": 2,
                     "coverage": {stage: {arm: [0, 1] for arm in arms}
                                  for stage in ("pool", "selected10", "selected_all")},
                     "recall": {stage: {arm: 1 for arm in arms}
                                for stage in ("pool", "selected10", "selected_all")}})
        for arm in arms:
            scores[arm][qid] = {"original_query_id": original,
                                "answer_f1": int((qid != "b") == (arm == "prehop_replay"))}
    return rows, scores


def test_initial_coverage_groups_precede_outcomes_and_preserve_original_question_clusters():
    rows, scores = coverage_fixture()
    result = initial_coverage_summary(rows, scores, ["answer_f1"], seed=42, repeats=50)
    groups = result["groups"]
    assert [groups[k]["rows"] for k in ("none", "partial", "complete")] == [2, 1, 1]
    assert groups["none"]["original_questions"] == 1
    assert groups["partial"]["initial_recall"] == .5
    assert groups["none"]["metrics"]["answer_f1"]["graph_minus_direct"]["ci95"] == [0, 0]
    assert groups["all"]["metrics"]["answer_f1"]["graph_minus_direct"]["mean"] == .5
    empty = initial_coverage_summary(rows[2:], scores, ["answer_f1"], seed=42, repeats=50)["groups"]["none"]
    assert empty["rows"] == 0 and empty["metrics"]["answer_f1"]["graph_minus_direct"]["ci95"] is None


def test_initial_coverage_cannot_drop_missing_scores_or_duplicate_questions():
    rows, scores = coverage_fixture()
    with pytest.raises(ValueError, match="Duplicate coverage"):
        initial_coverage_summary(rows + rows[:1], scores, ["answer_f1"], seed=42, repeats=50)
    rows[0]["gold_units"] = 0
    with pytest.raises(ValueError, match="Invalid initial/gold"):
        initial_coverage_summary(rows, scores, ["answer_f1"], seed=42, repeats=50)
    rows[0]["gold_units"] = 2
    scores["prehop_replay"]["a"].pop("answer_f1")
    with pytest.raises(ValueError, match="Unavailable required metric"):
        initial_coverage_summary(rows, scores, ["answer_f1"], seed=42, repeats=50)
    scores["prehop_replay"].pop("a")
    with pytest.raises(ValueError, match="question identity mismatch"):
        initial_coverage_summary(rows, scores, ["answer_f1"], seed=42, repeats=50)


def test_direct_channel_categories_union_all_witnesses_and_keep_selection_separate():
    nodes = {"start": {}, "b": {"representation_scores": {"body": 1 / 25}},
             "qm": {"representation_scores": {"q_minus": 1 / 30}},
             "qp": {"representation_scores": {"q_plus": 1 / 256}}}
    hits = {"start": {0}, "b": {1, 3}, "qm": {2, 3}, "qp": {2, 3}}
    units = direct_channel_evidence({"start"}, nodes, hits, {2}, {2, 3}, depth_limit=256)
    by_unit = {u["unit"]: u for u in units}
    assert list(by_unit) == [1, 2, 3]  # Initial evidence is outside the analysis.
    assert by_unit[1]["category"] == "body_only" and not by_unit[1]["retained_context"]
    assert by_unit[2]["category"] == "questions_only" and by_unit[2]["retained_top10"]
    assert by_unit[3]["category"] == "both"  # Different passages supply the same unit.
    assert by_unit[3]["channels"] == ["body", "q_minus", "q_plus"]
    assert len(by_unit[3]["witnesses"]) == 3
    assert by_unit[3]["retained_context"] and not by_unit[3]["retained_top10"]


def test_direct_channel_summary_preserves_occurrences_and_empty_supply():
    units = [{"unit": ("Title", 0), "category": "both", "channels": ["body", "q_minus"],
              "witnesses": {}, "retained_top10": False, "retained_context": True}]
    rows = [{"query_id": qid, "original_query_id": "original", "direct_channel_units": units}
            for qid in ("a", "b")]
    rows.append({"query_id": "c", "original_query_id": "other", "direct_channel_units": []})
    result = summarize_direct_channels(rows)
    assert result["rows"] == 3 and result["original_questions"] == 2
    assert result["total"]["supplied"] == 2 and result["total"]["original_questions"] == 1
    assert result["categories"]["both"]["context_retention"] == 1
    assert result["categories"]["questions_only"]["context_retention"] is None
    assert result["categories"]["questions_only"]["share_of_added_units"] == 0
    assert sum(g["supplied"] for g in result["channel_combinations"].values()) == 2
    empty = summarize_direct_channels(rows[-1:])
    assert empty["total"]["supplied"] == 0 and empty["total"]["context_retention"] is None


@pytest.mark.parametrize("scores", [{}, {"unknown": 1}, {"body": 0}, {"body": .3}, {"q_plus": 1 / 257}])
def test_recorded_channels_reject_missing_or_invalid_provenance(scores):
    with pytest.raises(ValueError):
        recorded_channel_ranks({"representation_scores": scores}, 256)


def test_direct_channels_reject_selection_outside_the_pool():
    with pytest.raises(ValueError, match="Selected evidence"):
        direct_channel_evidence({"s"}, {"s": {}}, {"s": {0}}, {1}, {1}, depth_limit=256)
