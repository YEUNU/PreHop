from utils.metrics import (
    UNJUDGED_SCORE,
    calculate_answer_metrics,
    calculate_evidence_doc_metrics,
    calculate_retrieval_ranking_metrics,
    extract_final_answer,
)
from utils.prompts.shared import mark_answer_boundary


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
    assert metrics["official_answer_em"] == 1.0
    assert metrics["official_answer_f1"] == 1.0
    assert metrics["null_refusal"] == UNJUDGED_SCORE


def test_answer_extraction_preserves_unmarked_long_response():
    response = "Correct entity appears first. " + "supporting context " * 40

    assert len(response) > 300
    assert extract_final_answer(response) == response.strip()


def test_answer_extraction_does_not_match_answer_as_an_ordinary_word():
    response = "I do not know the answer, as the supplied passages omit it."

    assert extract_final_answer(response) == response


def test_answer_extraction_accepts_explicit_markers_only():
    assert extract_final_answer("Reasoning. Final Answer: Paris") == "Paris"
    assert extract_final_answer("Reasoning.\nAnswer: Paris") == "Paris"


def test_answer_extraction_does_not_truncate_marked_long_response():
    response = "@@ANSWER: " + "complete answer text " * 30

    assert len(response) > 400
    assert extract_final_answer(response) == response.removeprefix("@@ANSWER: ").strip()


def test_answer_boundary_marker_preserves_an_existing_explicit_marker():
    response = "Reasoning.\nFinal Answer: Paris"

    assert mark_answer_boundary(response) == response
    assert extract_final_answer(mark_answer_boundary(response)) == "Paris"


def test_null_queries_use_refusal_metric_not_answer_em():
    metrics = calculate_answer_metrics(
        "Insufficient information.",
        "Insufficient information.",
        question_type="null_query",
    )

    assert metrics["answer_em"] == UNJUDGED_SCORE
    assert metrics["answer_f1"] == UNJUDGED_SCORE
    assert metrics["null_refusal"] == 1.0
    assert metrics["official_qa_accuracy"] == 1.0


def test_multihoprag_official_null_qa_does_not_treat_any_refusal_as_gold():
    metrics = calculate_answer_metrics(
        "I do not know.",
        "Insufficient information.",
        question_type="null_query",
    )

    assert metrics["answer_em"] == UNJUDGED_SCORE
    assert metrics["null_refusal"] == 1.0
    assert metrics["official_qa_accuracy"] == 0.0


def test_empty_gold_fact_ranking_is_excluded_not_zero():
    metrics = calculate_retrieval_ranking_metrics(
        [{"text": "some retrieved text", "doc": "source"}],
        [],
    )

    assert metrics["official_hits@10"] == UNJUDGED_SCORE
    assert metrics["official_mrr@10"] == UNJUDGED_SCORE
    assert metrics["official_map@10"] == UNJUDGED_SCORE


def test_multihoprag_official_hit_is_query_level_but_fact_recall_is_fractional():
    metrics = calculate_retrieval_ranking_metrics(
        [{"text": "The first evidence is alpha.", "doc": "source"}],
        ["alpha", "beta"],
    )

    assert metrics["official_hits@4"] == 1.0
    assert metrics["evidence_fact_recall@4"] == 0.5


def test_multihoprag_official_map_counts_new_facts_at_their_rank():
    metrics = calculate_retrieval_ranking_metrics(
        [{"text": "alpha"}, {"text": "unrelated"}, {"text": "beta"}],
        ["alpha", "beta"],
    )

    assert metrics["official_mrr@10"] == 1.0
    assert metrics["official_map@10"] == (1 / 1 + 1 / 3) / 2


def test_multihoprag_official_map_caps_denominator_without_clipping_score():
    facts = [f"fact{i}!" for i in range(12)]
    metrics = calculate_retrieval_ranking_metrics([{"text": " ".join(facts)}], facts)

    assert metrics["official_map@10"] == 1.2
    assert metrics["exact_fact_recall@10"] == 1.0


def test_multihoprag_official_map_counts_duplicate_compact_fact_once():
    metrics = calculate_retrieval_ranking_metrics(
        [{"text": "same fact"}], ["same fact", "same\nfact"]
    )

    assert metrics["official_map@10"] == 0.5
    assert metrics["official_hits@4"] == 1.0
    assert metrics["exact_fact_recall@10"] == 1.0


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
