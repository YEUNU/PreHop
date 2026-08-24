from utils.metrics import (
    UNJUDGED_SCORE,
    _resolve_judge_fields,
    calculate_answer_metrics,
    calculate_evidence_doc_metrics,
    calculate_retrieval_ranking_metrics,
)


def test_judge_correctness_and_groundedness_are_independent():
    fields = _resolve_judge_fields(
        {"score": 0, "groundedness": 1, "hallucination": 0, "reason": "supported but wrong"},
        "a substantive answer",
        "judge-test",
    )

    assert fields["llm_judge_score"] == 0.0
    assert fields["groundedness"] == 1.0
    assert fields["hallucination"] == 0.0


def test_judge_uses_final_answer_for_abstention_detection():
    fields = _resolve_judge_fields(
        {"score": 0, "groundedness": 0, "hallucination": 1, "reason": "unsupported"},
        "The intermediate branch had insufficient evidence.\n\nFinal Answer: Paris",
        "judge-test",
    )

    assert fields["groundedness"] == 0.0
    assert fields["hallucination"] == 1.0


def test_answer_metrics_use_final_answer_and_aliases():
    metrics = calculate_answer_metrics(
        "Reasoning text. Final Answer: Daniel Rozoum",
        "Daniel Darc",
        answer_aliases=["Daniel Rozoum"],
        question_type="2hop",
    )

    assert metrics["final_answer_extracted"] == "Daniel Rozoum"
    assert metrics["answer_em"] == 1.0
    assert metrics["answer_f1"] == 1.0
    assert metrics["null_refusal"] == UNJUDGED_SCORE


def test_null_queries_use_refusal_metric_not_answer_em():
    metrics = calculate_answer_metrics(
        "I do not know.",
        "Insufficient information",
        question_type="null_query",
    )

    assert metrics["answer_em"] == UNJUDGED_SCORE
    assert metrics["answer_f1"] == UNJUDGED_SCORE
    assert metrics["null_refusal"] == 1.0


def test_empty_gold_fact_ranking_is_excluded_not_zero():
    metrics = calculate_retrieval_ranking_metrics(
        [{"text": "some retrieved text", "doc": "source"}],
        [],
    )

    assert metrics["hits@10"] == UNJUDGED_SCORE
    assert metrics["mrr@10"] == UNJUDGED_SCORE
    assert metrics["map@10"] == UNJUDGED_SCORE


def test_evidence_doc_metrics_deduplicate_retrieved_chunks():
    metrics = calculate_evidence_doc_metrics(
        [
            {"doc": "Alpha", "text": "first"},
            {"doc": "Alpha", "text": "second"},
            {"doc": "Beta", "text": "third"},
        ],
        ["Alpha", "Gamma"],
    )

    assert metrics["evidence_doc_precision"] == 0.5
    assert metrics["evidence_doc_recall"] == 0.5
    assert metrics["evidence_doc_f1"] == 0.5
