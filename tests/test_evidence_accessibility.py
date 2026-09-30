"""Boundary checks for retrospective candidate supply, without model services."""

import pytest

from scripts.analyze_evidence_accessibility import evidence_witnesses, indexed, paired_scores, prefix_ids, row_curves


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
