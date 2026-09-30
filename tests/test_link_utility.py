"""Retained HOP usefulness and failure-denominator checks."""
import pytest

from scripts.analyze_ablation_links import usefulness


def test_link_utility_separates_direct_and_next_and_zero_gain():
    payload = {"direct":[{"id":"d","text":"fact one"}],
               "expanded":[{"id":"d","text":"fact one","path_type":"hop"},
                           {"id":"h","text":"fact two","path_type":"hop"},
                           {"id":"h","text":"fact two","path_type":"next"},
                           {"id":"n","text":"fact three","path_type":"next"}],
               "selected":[{"id":"d"},{"id":"h"}]}
    result = usefulness(payload,["fact one","fact two","fact three"],"multihoprag")
    assert result["hop_destinations"] == 2
    assert result["hop_destination_relevance"] == 1
    assert result["added_gold_coverage"] == pytest.approx(1/3)
    assert result["retained_added_coverage"] == pytest.approx(1/3)
    assert result["retained_next_overlap_coverage"] == pytest.approx(1/3)
    empty = usefulness({"direct":[],"expanded":[],"selected":[]},["fact"],"multihoprag")
    assert empty["hop_destination_relevance"] is None
    assert empty["hop_destinations"] == 0 and empty["added_gold_coverage"] == 0


def test_actual_query_utility_uses_clusters_and_keeps_failures_separate():
    from scripts.analyze_ablation_links import summarize_utility
    rows=[{'query_id':'a','original_query_id':'same','added_gold_coverage':1},
          {'query_id':'b','original_query_id':'same','added_gold_coverage':1},
          {'query_id':'c','original_query_id':'other','added_gold_coverage':0},
          {'query_id':'d','original_query_id':'failure','failed':True}]
    result=summarize_utility(rows)['added_gold_coverage']
    assert result['release_row_mean']==pytest.approx(2/3)
    assert result['unique_question_macro_mean']==.5
    assert result['total_rows']==4 and result['eligible_queries']==3
    assert result['clusters']==2
