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
