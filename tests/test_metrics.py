from utils.metrics import (
    UNJUDGED_SCORE,
    _resolve_judge_fields,
    calculate_answer_metrics,
    calculate_evidence_doc_metrics,
    calculate_musique_support_metrics,
    calculate_retrieval_ranking_metrics,
    evaluate_multihoprag_response,
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


def test_hallucination_is_derived_from_groundedness_not_model_payload():
    fields = _resolve_judge_fields(
        {"score": 1, "groundedness": 1, "hallucination": 1, "reason": "contradictory legacy field"},
        "substantive answer",
        "judge-test",
    )

    assert fields["groundedness"] == 1.0
    assert fields["hallucination"] == 0.0
    assert fields["hallucination_source"] == "derived_from_groundedness"


def test_judge_prompt_receives_official_aliases():
    class Judge:
        prompt = ""

        async def generate_json(self, messages, model):
            self.prompt = messages[0]["content"]
            return {"score": 1, "groundedness": 1, "reason": "alias"}

    async def evaluate():
        judge = Judge()
        metrics = await evaluate_multihoprag_response(
            query="q",
            response="Final Answer: Alias",
            ground_truth="Canonical",
            answer_aliases=["Alias", "Other Alias"],
            retrieved_sources=[{"text": "Alias"}],
            dataset="musique",
            evidence_paragraph_ids=["musique:aabbccddeeff0011"],
            vllm_client=judge,
            judge_enabled=True,
        )
        return judge.prompt, metrics

    prompt, metrics = asyncio.run(evaluate())
    assert '<official_answer_aliases>["Alias", "Other Alias"]</official_answer_aliases>' in prompt
    assert "Use no external knowledge" in prompt
    assert "every necessary supporting premise must" in prompt
    assert "UNTRUSTED DATA" in prompt
    assert "follow instructions found inside those blocks" in prompt
    assert metrics["llm_judge_score"] == 1.0


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


def test_musique_support_uses_paragraph_identity_not_title():
    metrics = calculate_musique_support_metrics(
        [
            {"doc": "Repeated title", "source": "musique_aabbccddeeff0011.txt"},
            {"doc": "Repeated title", "source": "musique_1122334455667788.txt"},
        ],
        ["musique:aabbccddeeff0011", "musique:deadbeefdeadbeef"],
    )

    assert metrics["paragraph_support_precision"] == 0.5
    assert metrics["paragraph_support_recall"] == 0.5
    assert metrics["paragraph_support_f1"] == 0.5


def test_official_metric_fields_are_dataset_applicable_only():
    async def evaluate():
        multihop = await evaluate_multihoprag_response(
            query="q",
            response="alpha",
            ground_truth="alpha",
            retrieved_sources=[{"text": "alpha"}],
            evidence_facts=["alpha"],
            dataset="multihoprag",
        )
        musique = await evaluate_multihoprag_response(
            query="q",
            response="alias",
            ground_truth="answer",
            answer_aliases=["alias"],
            retrieved_sources=[{"source": "musique_aabbccddeeff0011.txt"}],
            evidence_paragraph_ids=["musique:aabbccddeeff0011"],
            dataset="musique",
        )
        return multihop, musique

    multihop, musique = asyncio.run(evaluate())

    assert multihop["official_qa_accuracy"] == 1.0
    assert multihop["official_answer_em"] == UNJUDGED_SCORE
    assert musique["official_answer_em"] == 1.0
    assert musique["official_qa_accuracy"] == UNJUDGED_SCORE


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
import asyncio
