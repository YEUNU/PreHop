from utils.metrics import _resolve_judge_fields


def test_resolve_judge_fields_never_fabricates_missing_score():
    fields = _resolve_judge_fields(None, "answer", "gpt-test")
    assert fields["llm_judge_score"] == -1.0
    assert fields["hallucination"] == -1.0


def test_resolve_judge_fields_does_not_derive_hallucination_from_score():
    fields = _resolve_judge_fields({"score": 0.0, "reason": "incorrect"}, "substantive answer", "gpt-test")
    assert fields["llm_judge_score"] == 0.0
    assert fields["groundedness"] == -1.0
    assert fields["hallucination"] == -1.0
    assert fields["hallucination_source"] == "unjudged"
