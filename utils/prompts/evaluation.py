MULTIHOPRAG_JUDGE_PROMPT = """
### Task: Score a Model Prediction for a multi-hop question-answering query
on (a) correctness vs Ground Truth and (b) hallucination — in a SINGLE LLM
call so the two judgements stay internally consistent.

Answers are short, factual spans (an entity/person/organization name, a
publisher, a date or time period, or a yes/no comparison result), NOT
financial figures. There is no unit scaling or currency formatting to
reconcile here — judge on factual identity, not numeric tolerance.

**Question Type:** {question_type}
**Question:** {query}
**Ground Truth Answer:** {ground_truth}
**Model Prediction:** {response}

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
3. score:
   - 1.0 if the final answer is factually equivalent to the Ground Truth
     (minor wording / alias / casing differences ok).
   - 0.0 if it names the wrong entity/date/outcome, or — for a non-null
     question — abstains when a substantive Ground Truth exists.
4. hallucination:
   - 1.0 if the final answer asserts a concrete but factually wrong entity,
     date, or comparison outcome (including a fabricated answer to a
     null_query).
   - 0.0 if it is factually consistent with the Ground Truth, or is an
     honest abstention.
5. Internal consistency: hallucination=1.0 implies score=0.0; score=1.0
   implies hallucination=0.0; score=0.0 with hallucination=0.0 is the
   honest-abstain case.

Respond ONLY in JSON format:
{{"score": 1.0 or 0.0, "hallucination": 1.0 or 0.0, "reason": "brief explanation covering both judgements"}}
"""
