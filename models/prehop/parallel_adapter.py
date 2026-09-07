"""Bounded lookahead around unchanged native Prehop chunk extraction.

Native parsing, question generation, failure scopes and output assembly are
retained. Futures are consumed in source order, including duplicate chunk text.
"""
from __future__ import annotations

import asyncio
import os
from collections import deque
from contextvars import ContextVar

from core.execution_profile import execution_profile
from core.structured_diagnostics import indexing_failure_scope
from models.prehop.graphrag import GraphRAG
from models.prehop.indexing.chunking import parse_pages_offline, split_fixed_sentence_windows
from models.prehop.tracing import traced

_active_chunks: ContextVar = ContextVar('prehop_adapter_chunk_window', default=None)


class ParallelChunkGraphRAG(GraphRAG):
    @traced
    async def extract_knowledge(self, content, source='', prepared_pages=None):
        window = execution_profile()['settings'].get('prehop_chunk_concurrency', 1)
        if window == 1 or os.environ.get('RAG_CHUNK_CACHE', '').lower() != 'off':
            return await super().extract_knowledge(content, source, prepared_pages)
        parsed = prepared_pages if prepared_pages is not None else parse_pages_offline(source, content)
        title = str(parsed.get('title') or source)
        specs = iter((page['num'], i, text)
                     for page in parsed.get('pages', []) if page['content']
                     for i, text in enumerate(split_fixed_sentence_windows(page['content'])))
        pending = deque()

        async def extract(page, i, text):
            with indexing_failure_scope(source, content, title, text, page, i):
                return await super(ParallelChunkGraphRAG, self).extract_hoprag_queries(text, title)

        def fill():
            while len(pending) < window:
                spec = next(specs, None)
                if spec is None:
                    break
                pending.append((spec[2], asyncio.create_task(extract(*spec))))

        token = _active_chunks.set((self, title, pending, fill))
        try:
            fill()
            result = await super().extract_knowledge(content, source, parsed)
            if pending or next(specs, None) is not None:
                raise RuntimeError('Native Prehop did not consume the adapter chunk plan')
            return result
        finally:
            _active_chunks.reset(token)
            tasks = [task for _, task in pending]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def extract_hoprag_queries(self, chunk, title=''):
        frame = _active_chunks.get()
        if frame is None or frame[0] is not self:
            return await super().extract_hoprag_queries(chunk, title)
        _, expected_title, pending, fill = frame
        if not pending or pending[0][0] != chunk or expected_title != title:
            raise RuntimeError('Native Prehop chunk order differs from adapter lookahead')
        _, task = pending.popleft()
        result = await task
        fill()
        return result
