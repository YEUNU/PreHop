import pytest

from scripts.evaluate_saved_retrieval import hotpot_metrics, multihop_metrics, rank_metrics


def test_distinct_gold_rank_credit_and_coverage():
    # Two facts in one passage receive the same rank; repeated facts add no MAP credit.
    result = rank_metrics([set(), {"a", "b"}, {"a"}, set(), {"c"}], 3)
    assert result == {"Hits@4": 1.0, "Hits@10": 1.0, "MRR@10": .5,
                      "MAP@10": pytest.approx(.4), "Recall@10": 1.0}


def test_empty_gold_is_excluded_but_empty_prediction_is_zero():
    assert rank_metrics([], 0) is None
    assert set(rank_metrics([], 2).values()) == {0.0}
    assert rank_metrics([set()] * 10 + [{"a"}], 1)["Recall@10"] == 0


def test_hotpot_identity_and_rank_cap():
    # The same sentence index in another title is not a match.
    sources = [[("Wrong title", 0)], [("Correct title", 0)], [("Correct title", 0)]]
    result = hotpot_metrics(sources, [["Correct title", 0], ["Correct title", 1]], lambda s: s)
    assert result["MAP@10"] == .25
    assert result["Recall@10"] == .5


def test_multihop_uses_official_case_sensitive_whitespace_matching():
    sources = [{"text": "A B\nC"}, {"text": "different"}]
    assert multihop_metrics(sources, ["ABC"])["MAP@10"] == 1
    assert multihop_metrics(sources, ["abc"])["MAP@10"] == 0


def test_official_map_denominator_cap():
    # Preserve the benchmark's min(gold_count, 10), not conventional AP.
    assert rank_metrics([{0}], 12)["MAP@10"] == .1
    assert rank_metrics([{0}], 12)["Recall@10"] == 1 / 12
