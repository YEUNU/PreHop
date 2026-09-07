"""Observe the pinned agent entrypoint without reimplementing its reasoning loop."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


def run_native_agent(module, graphq, retriever, question: str, schema_path: str, audit_path: Path):
    merged = []
    final_evidence = []
    calls = []
    answers = []
    initial_results = []
    triples = []
    native_merge = module.merge_chunk_contents
    native_initial = module.initial_question_decomposition
    native_dedup = module.deduplicate_triples

    def observe_initial(*args, **kwargs):
        result = native_initial(*args, **kwargs)
        initial_results.append(result)
        return result

    def observe_triples(values):
        result = native_dedup(values)
        triples[:] = result
        return result

    def observe_merge(ids, contents):
        result = native_merge(ids, contents)
        merged[:] = list(zip(ids, result))
        return result

    class ObservedRetriever:
        def __getattr__(self, name):
            return getattr(retriever, name)

        def generate_prompt(self, *args, **kwargs):
            result = retriever.generate_prompt(*args, **kwargs)
            final_evidence[:] = merged
            return result

        def generate_answer(self, prompt):
            answer = retriever.generate_answer(prompt)
            calls.append({'prompt': prompt, 'answer': answer})
            return answer

    class CaptureEvaluation:
        def eval(self, observed_question, _gold, answer):
            if observed_question != question:
                raise RuntimeError('Native Youtu agent changed query identity')
            answers.append(answer)
            # This is the post-answer evaluator, outside retrieval/generation.
            # Paper metrics are computed against the real gold in the benchmark.
            return 'not_evaluated'

    try:
        with (patch.object(module, 'Eval', CaptureEvaluation),
              patch.object(module, 'merge_chunk_contents', observe_merge),
              patch.object(module, 'initial_question_decomposition', observe_initial),
              patch.object(module, 'deduplicate_triples', observe_triples)):
            module.agent_retrieval(graphq, ObservedRetriever(), [{'question': question, 'answer': ''}], schema_path)
        if len(answers) != 1 or not isinstance(answers[0], str) or not answers[0].strip() or answers[0].startswith('Error:'):
            raise RuntimeError('Native Youtu agent did not produce one valid final answer')
        if len(initial_results) != 1:
            raise RuntimeError('Native Youtu agent omitted its initial retrieval result')
        return {**initial_results[0], 'triples': list(triples), 'initial_answer': answers[0], 'chunk_ids': [x[0] for x in final_evidence],
                'chunk_contents': [x[1] for x in final_evidence]}
    finally:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'query': question, 'native_mode': 'agent', 'calls': calls,
                                     'answers': answers, 'final_evidence': final_evidence,
                                     'native_posthoc_evaluator': 'replaced_by_paper_metrics'}, ensure_ascii=False) + '\n')
