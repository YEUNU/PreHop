MULTIHOPRAG_JUDGE_PROMPT = """
### Task: Evaluate a model prediction for a multi-hop question-answering query
in two independent directions during a SINGLE LLM call:

(a) answer correctness, directed from the prediction to the Ground Truth; and
(b) evidence groundedness, directed from the prediction to the Retrieved
Context. Do not use one judgement as a substitute for the other.

Answers are short factual spans such as an entity, person, organization,
source, date, number, or yes/no comparison result. Judge factual equivalence
and accept harmless formatting differences that do not change the answer.

**Question Type:** {question_type}
**Question:** {query}
**Ground Truth Answer:** {ground_truth}
**Model Prediction:** {response}
**Retrieved Context:**
{retrieved_context}

### Instructions
1. Locate the FINAL answer in the Model Prediction (typically after
   "Final Answer:", the last "@@ANSWER:" marker, or inside \\boxed{{...}}).
   Judge on that final answer only — ignore intermediate hop-by-hop reasoning.
2. Apply the criterion for the question type:
   - inference_query / comparison_query / comparison / bridge / Nhop
     (e.g. 2hop, 3hop, 4hop): the predicted entity / comparison outcome must
     match the Ground Truth entity (surface aliases and reorderings are
     acceptable; e.g. "The Verge" == "the verge").
   - temporal_query: the chronological fact / ordering / date must match.
   - null_query: the Ground Truth indicates the corpus has NO answer
     ("insufficient information"). Here an honest abstention ("the context
     does not contain ...", "insufficient information", "cannot be
     determined") is the CORRECT answer (score 1.0); fabricating a concrete
     answer is wrong (score 0.0).
   - any other question type: judge factual equivalence to the Ground Truth
     directly, using the same score/hallucination rules below.
3. `score` is answer correctness against the Ground Truth:
   - 1.0 if the final answer is factually equivalent to the Ground Truth
     (minor wording / alias / casing differences ok). Treat an answer alias
     as semantic equivalence, not as permission to add unsupported facts.
   - 0.0 if it names the wrong entity/date/outcome, or — for a non-null
     question — abstains when a substantive Ground Truth exists.
4. `groundedness` is support by the Retrieved Context:
   - 1.0 if the context entails or directly supports the final substantive
     answer, allowing a short multi-hop synthesis from statements present in
     the context.
   - 0.0 if the final substantive answer is unsupported, contradicted, or
     requires information absent from the context.
   - For an honest abstention or empty prediction, return 0.0; the evaluator
     records abstention as not applicable for groundedness.
   - If the Retrieved Context is empty, every substantive answer is
     unsupported.
5. `hallucination` is a context-groundedness label, not another correctness
   label:
   - 1.0 for a substantive answer when groundedness=0.0.
   - 0.0 for a substantive answer when groundedness=1.0, and for an honest
     abstention.
   - For `null_query`, an honest abstention has score=1.0 and
     hallucination=0.0; a concrete answer is a hallucination even if it is
     plausible from outside the retrieved context.
6. Keep the axes independent. A wrong answer supported by the retrieved
   context has score=0.0, groundedness=1.0, hallucination=0.0. A correct
   answer not supported by the retrieved context has score=1.0,
   groundedness=0.0, hallucination=1.0.

Respond ONLY in JSON format:
{{"score": 1.0 or 0.0, "groundedness": 1.0 or 0.0, "hallucination": 1.0 or 0.0, "reason": "brief explanation covering correctness and context support"}}
"""
