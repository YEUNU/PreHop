"""Saved retrieval inputs for answer-only replay, without retrieval or text fitting."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.prompts.prehop_answer import build_answer_messages


def context_digest(context: str) -> str:
    return hashlib.sha256(context.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class SynthesisInput:
    dataset: str
    method: str
    query_id: str
    query: str
    context: str
    provenance: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def key(self) -> tuple[str, str, str]:
        return self.dataset, self.method, self.query_id

    @property
    def context_sha256(self) -> str:
        return context_digest(self.context)

    def messages(self) -> list[dict[str, str]]:
        """The hook replaces answer instructions and preserves the entire evidence string."""
        return build_answer_messages(self.context, self.query)

    def as_record(self) -> dict[str, Any]:
        return {**self.metadata, 'dataset': self.dataset, 'method': self.method,
                'query_id': self.query_id, 'query': self.query, 'context': self.context,
                'context_sha256': self.context_sha256, 'trace_provenance': self.provenance}

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> SynthesisInput:
        context = row['context']
        if not isinstance(context, str):
            raise TypeError('Recorded synthesis context must be text')
        fields = {'dataset', 'method', 'query_id', 'query', 'context', 'context_sha256', 'trace_provenance', 'messages'}
        return cls(row['dataset'], row['method'], row['query_id'], row['query'], context,
                   row['trace_provenance'], {k: v for k, v in row.items() if k not in fields})


class TraceContextProvider:
    """Default synthesis source: recorded inputs only, with no live-retriever fallback."""
    def __init__(self, path: Path):
        self.path = Path(path)

    def __iter__(self) -> Iterator[SynthesisInput]:
        with self.path.open() as stream:
            for line in stream:
                if line.strip():
                    yield SynthesisInput.from_record(json.loads(line))

    def groups(self) -> Iterator[tuple[tuple[str, str], list[SynthesisInput]]]:
        # First-seen dataset/system order is explicit in the saved input file.
        groups: dict[tuple[str, str], list[SynthesisInput]] = {}
        for row in self:
            groups.setdefault((row.dataset, row.method), []).append(row)
        yield from groups.items()


def returned_passage_context(sources: Iterable[dict[str, Any]]) -> str:
    """Format saved passage outputs, retaining every text in order."""
    return '\n\n'.join(
        f"[[{p.get('doc', p.get('title', 'Unknown'))}, Page {p.get('page', 0)}, Chunk {p.get('sent_id', 0)}]]\n{p.get('text', '')}"
        for p in sources
    )


async def evaluate_responses(
    result_path: Path, responses: list[dict[str, Any]], response_reference: dict[str, str],
    *, sentence_store: Path,
) -> dict[str, Any]:
    """Score a complete reader replay without altering the original retrieval result."""
    from core.benchmark_evaluation import _apply_judge_label, _recompute_aggregates, _update_summary_status
    from utils.metrics import evaluate_multihoprag_response
    from utils.official_results import dataset_key
    from utils.provenance import code_provenance

    raw = result_path.read_bytes()
    source = json.loads(raw)
    dataset = dataset_key(source.get('corpus_tag') or source['dataset'])
    method = source['strategy']
    source_sha256 = hashlib.sha256(raw).hexdigest()
    selected = {}
    for response in responses:
        if (response['dataset'], response['method']) != (dataset, method):
            continue
        qid = response['query_id']
        if qid in selected:
            raise ValueError(f'Duplicate reader response: {dataset}/{method}/{qid}')
        selected[qid] = response
    original_rows = source['details']
    query_ids = [row['query_id'] for row in original_rows]
    if len(set(query_ids)) != len(query_ids) or set(selected) != set(query_ids):
        raise ValueError(f'Reader response population differs from {result_path}')
    reader_models = {row.get('model') for row in selected.values()}
    if len(reader_models) != 1 or not next(iter(reader_models), None):
        raise ValueError(f'Reader model must be recorded consistently for {result_path}')
    rows = []
    for original in original_rows:
        qid = original['query_id']
        response = selected[qid]
        if (response.get('status') != 'completed'
                or response.get('trace_provenance', {}).get('source_sha256') != source_sha256
                or response.get('context_sha256') != context_digest(returned_passage_context(original['retrieved_sources']))
                or response.get('query') != original['query']):
            raise ValueError(f'Incomplete or mismatched reader evidence: {dataset}/{method}/{qid}')
        row = {key: original[key] for key in (
            'idx', 'query_id', 'original_query_id', 'query', 'category', 'question_type',
            'ground_truth', 'answer_aliases', 'expected_sources', 'retrieved_sources', 'error', 'failure_scope',
        ) if key in original}
        row['answer'] = response['answer']
        expected = row.get('expected_sources') or {}
        row.update(await evaluate_multihoprag_response(
            query=row['query'], response=row['answer'], ground_truth=row['ground_truth'],
            retrieved_sources=row['retrieved_sources'], evidence_facts=expected.get('facts', []),
            evidence_docs=expected.get('docs', []), question_type=row.get('question_type', ''),
            dataset=dataset, answer_aliases=row.get('answer_aliases', []), judge_enabled=False,
            supporting_facts=expected.get('supporting_facts', []), hotpot_sentence_store=str(sentence_store),
        ))
        if row.get('error'):
            from core.benchmark_failures import QUALITY_METRICS
            row.update({key: 0.0 for key in QUALITY_METRICS})
        row['reader_seconds'] = response.get('reader_seconds')
        row['reader_request_sha256'] = response.get('request_sha256')
        row['reader_transport'] = response.get('reader_transport')
        _apply_judge_label(row)
        rows.append(row)
    result = {key: source[key] for key in (
        'dataset', 'corpus_tag', 'dataset_protocol', 'strategy', 'evaluation_scope',
        'official_split_expected_queries', 'total_queries', 'index_manifest_stats_path',
        'corpus_manifest', 'query_provenance',
    ) if key in source}
    result.update(
        details=rows, queries_count=len(rows), judge_enabled=False,
        evaluation_provenance=code_provenance(),
        common_reader={
            'source_result': str(result_path.resolve()), 'source_sha256': source_sha256,
            'responses': response_reference, 'model': next(iter(reader_models)),
            'context_policy': 'all returned passages in recorded order; no truncation',
            'timing_scope': 'reader-only; no new retrieval or end-to-end latency',
        },
    )
    _recompute_aggregates(result)
    _update_summary_status(result)
    return result
