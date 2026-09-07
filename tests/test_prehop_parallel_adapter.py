import asyncio
import json

import pytest

from models.prehop import parallel_adapter
from models.prehop.graphrag import GraphRAG
from models.prehop.indexing import chunking
from tests.test_adapter_producer_concurrency import profile


def setup(tmp_path, monkeypatch):
    value = profile(); value['settings']['prehop_chunk_concurrency'] = 4
    path = tmp_path / 'profile.json'; path.write_text(json.dumps(value))
    monkeypatch.setenv('RAG_EXECUTION_PROFILE', str(path))
    monkeypatch.setenv('RAG_CHUNK_CACHE', 'off')
    for module in (chunking, parallel_adapter):
        monkeypatch.setattr(module, 'split_fixed_sentence_windows', lambda value: value.split('|'))
    engine = object.__new__(parallel_adapter.ParallelChunkGraphRAG)
    engine.corpus_tag = 'multihoprag'
    engine.indexing_model_id = 'default'
    engine.save_intermediate = False
    return engine


def test_prefetch_keeps_native_assembly_duplicates_and_document_isolation(tmp_path, monkeypatch):
    engine = setup(tmp_path, monkeypatch)
    state = {'active': 0, 'peak': 0, 'calls': []}
    async def extract(self, text, title):
        state['active'] += 1; state['peak'] = max(state['peak'], state['active'])
        state['calls'].append((text, title))
        try:
            await asyncio.sleep(.01 if text == 'A' else 0)
            return {'q_minus': [title + text], 'q_plus': []}
        finally:
            state['active'] -= 1
    monkeypatch.setattr(GraphRAG, 'extract_hoprag_queries', extract)
    pages = lambda title: {'title': title, 'pages': [{'num': 1, 'content': 'A|A|B'}, {'num': 2, 'content': 'C'}]}
    async def run():
        parallel = await asyncio.gather(*(engine.extract_knowledge('content', title, pages(title)) for title in ('X', 'Y')))
        assert state['peak'] == 8 and len(state['calls']) == 8
        monkeypatch.delenv('RAG_EXECUTION_PROFILE')
        serial = [await GraphRAG.extract_knowledge(engine, 'content', title, pages(title)) for title in ('X', 'Y')]
        assert parallel == serial
        assert [c['text'] for c in parallel[0]['chunks']] == ['A', 'A', 'B', 'C']
        assert [c['sent_id'] for c in parallel[0]['chunks']] == [0, 1, 2, 3]
    asyncio.run(run())


def test_failed_prefetch_cancels_and_joins_remaining_requests(tmp_path, monkeypatch):
    engine = setup(tmp_path, monkeypatch)
    finished = []
    async def extract(self, text, title):
        try:
            if text == 'A': raise ValueError('native extraction failure')
            await asyncio.Event().wait()
        finally: finished.append(text)
    monkeypatch.setattr(GraphRAG, 'extract_hoprag_queries', extract)
    async def run():
        with pytest.raises(ValueError, match='native extraction failure'):
            await engine.extract_knowledge('content', 'source', {'title': 'T', 'pages': [{'num': 1, 'content': 'A|B|C|D|E'}]})
        assert sorted(finished) == ['A', 'B', 'C', 'D']
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    asyncio.run(run())
