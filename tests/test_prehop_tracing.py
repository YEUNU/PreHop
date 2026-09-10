import asyncio
import gzip
import json
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI

from core.config import RAGConfig
from core.structured_outputs import StructuredOutputError, question_contract
from core.vllm_client import VLLMClient
from models.prehop.tracing import TraceRecorder, attach_client, trace_identity, traced
from scripts.inspect_prehop_trace import inspect_trace


def recorder(tmp_path):
    return TraceRecorder(tmp_path / 'session', secrets=('test-secret-key', '4096'))


def payloads(trace):
    return inspect_trace(trace.directory / 'events.jsonl', payloads=True)


def client(monkeypatch, trace, handler, sdk_retries=0):
    monkeypatch.setenv('RAG_PAPER_MODE', 'false')
    monkeypatch.delenv('RAG_QUEUE_PROXY_URL', raising=False)
    monkeypatch.setattr(RAGConfig, 'VLLM_URL', 'http://test/v1')
    monkeypatch.setattr(RAGConfig, 'VLLM_EMBED_URL', 'http://test/v1')
    monkeypatch.setattr(RAGConfig, 'LLM_MAX_RETRIES', 2)
    monkeypatch.setattr(RAGConfig, 'LLM_RETRY_DELAY', 0)
    sdk = AsyncOpenAI(base_url='http://test/v1', api_key='test-secret-key',
                     max_retries=sdk_retries, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    value = VLLMClient('generation-model')
    value._get_cached_client = lambda _: sdk
    return attach_client(value, trace), sdk


def completion(content, finish='stop'):
    return {'id': 'test', 'object': 'chat.completion', 'created': 1, 'model': 'generation-model',
            'choices': [{'index': 0, 'finish_reason': finish,
                         'message': {'role': 'assistant', 'content': content}}],
            'usage': {'prompt_tokens': 455, 'completion_tokens': 4096, 'total_tokens': 4551}}


@pytest.mark.asyncio
async def test_truncation_raw_response_survives_validation_failure(tmp_path, monkeypatch):
    trace = recorder(tmp_path)
    raw = '{"q_minus":["unfinished question'
    value, sdk = client(monkeypatch, trace, lambda _: httpx.Response(200, json=completion(raw, 'length')))
    try:
        with trace_identity(source='source.txt', query_id='q1'), pytest.raises(StructuredOutputError):
            await value.generate_json([{'role': 'user', 'content': 'original input'}],
                                      structured_contract=question_contract('index'))
        events = payloads(trace)
        response = next(e['data'] for e in events if e['event'] == 'http.response')
        assert json.loads(response['body'])['choices'][0]['message']['content'] == raw
        sdk_response = next(e['data'] for e in events if e['event'] == 'sdk_attempt.result')
        assert sdk_response['usage']['completion_tokens'] == 4096
        assert any(e['event'] == 'generate_json.error' for e in events)
        assert all(e['identity'].get('source') == 'source.txt' for e in events if e['event'] != 'session_start')
        for event in events:
            assert 'test-secret-key' not in json.dumps(event)
    finally:
        await sdk.close()


@pytest.mark.asyncio
async def test_format_retry_keeps_both_original_responses(tmp_path, monkeypatch):
    trace = recorder(tmp_path)
    bodies = ['{"q_minus":', '{"q_minus":["Who?"],"q_plus":[]}']
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=completion(bodies[len(calls)-1]))
    value, sdk = client(monkeypatch, trace, handler)
    try:
        result = await value.generate_json([{'role': 'user', 'content': 'source'}],
                                            structured_contract=question_contract('index'))
        assert result == {'q_minus': ['Who?'], 'q_plus': []}
        assert calls[0] == calls[1]
        events = payloads(trace)
        assert [e['data']['valid'] for e in events if e['event'] == 'structured.attempt'] == [False, True]
        assert [json.loads(e['data']['body'])['choices'][0]['message']['content']
                for e in events if e['event'] == 'http.response'] == bodies
    finally:
        await sdk.close()


@pytest.mark.asyncio
async def test_sdk_internal_http_retry_is_also_visible(tmp_path, monkeypatch):
    trace = recorder(tmp_path)
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, json={'error': {'message': 'busy'}})
        return httpx.Response(200, json=completion('answer'))
    value, sdk = client(monkeypatch, trace, handler, sdk_retries=1)
    try:
        assert await value.generate_response([{'role': 'user', 'content': 'question'}]) == 'answer'
        events = payloads(trace)
        assert [e['data']['status_code'] for e in events if e['event'] == 'http.response'] == [503, 200]
        # Other strategy using the same HTTP pool has no Prehop context.
        count = len(events)
        await sdk.chat.completions.create(model='generation-model', messages=[{'role': 'user', 'content': 'other'}])
        assert len(payloads(trace)) == count
    finally:
        await sdk.close()


@pytest.mark.asyncio
async def test_parallel_document_scopes_and_cancelled_spans(tmp_path):
    trace = recorder(tmp_path)
    class Engine:
        trace_recorder = trace
        @traced
        async def document(self, source):
            await asyncio.sleep(0)
            if source == 'cancel':
                raise asyncio.CancelledError
            return {'source': source, 'chunks': [source]}
    result = await asyncio.gather(*(Engine().document(str(i)) for i in range(60)))
    assert len(result) == 60
    with pytest.raises(asyncio.CancelledError):
        await Engine().document('cancel')
    events = payloads(trace)
    for event in events:
        if event['event'].endswith('.result'):
            assert event['identity']['source'] == event['data']['source']
    assert len({e['span_id'] for e in events if e['event'].endswith('.start')}) == 61
    assert any(e['event'].endswith('.error') and e['data']['error_type'] == 'CancelledError' for e in events)


def test_inspector_rejects_corruption_and_keeps_numeric_fields(tmp_path):
    trace = recorder(tmp_path)
    trace.emit('test', {'count': 4096, 'api_key': 'test-secret-key', 'text': 'secret test-secret-key'})
    rows = payloads(trace)
    assert rows[-1]['data'] == {'count': 4096, 'api_key': '[REDACTED]', 'text': 'secret [REDACTED]'}
    target = trace.directory / rows[-1]['payload']
    assert target.stat().st_mode & 0o777 == 0o600
    target.write_bytes(gzip.compress(b'{}'))
    with pytest.raises(ValueError, match='hash mismatch'):
        payloads(trace)


def test_prehop_default_trace_and_other_strategy_boundary(tmp_path, monkeypatch):
    monkeypatch.delenv('RAG_PREHOP_TRACE', raising=False)
    monkeypatch.setenv('RAG_PREHOP_TRACE_DIR', str(tmp_path))
    monkeypatch.setenv('RAG_PAPER_MODE', 'false')
    monkeypatch.setattr(RAGConfig, 'VLLM_URL', 'http://test/v1')
    monkeypatch.setattr(RAGConfig, 'VLLM_EMBED_URL', 'http://test/v1')
    from models.prehop.graphrag import GraphRAG
    engine = GraphRAG(corpus_tag='test')
    assert engine.save_intermediate
    assert Path(engine.trace_recorder.reference['events_path']).is_file()
    other = GraphRAG(strategy='other', corpus_tag='test')
    assert other.trace_recorder is None


@pytest.mark.asyncio
async def test_batched_graph_write_is_findable_from_each_document(tmp_path):
    from models.prehop.tracing import TracedNeo4j
    trace = recorder(tmp_path)
    class Service:
        async def execute_query(self, query, parameters):
            return [{'written': len(parameters['documents'])}]
    service = TracedNeo4j(Service(), trace)
    result = await service.execute_query('UNWIND $documents AS doc RETURN doc',
        {'documents': [{'doc_id': 'one.txt'}, {'doc_id': 'two.txt'}]})
    assert result == [{'written': 2}]
    for source in ('one.txt', 'two.txt'):
        rows = inspect_trace(trace.directory / 'events.jsonl', source=source)
        assert len(rows) == 2
        assert rows[0]['span_id'] == rows[1]['span_id']


@pytest.mark.asyncio
async def test_traced_embeddings_progress_when_default_executor_is_saturated(tmp_path, monkeypatch):
    """Permit waiters must not block the HTTP hooks of permit holders."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    trace = recorder(tmp_path)
    gate = asyncio.Event()
    entered = asyncio.Event()
    requests = 0

    async def handler(request):
        nonlocal requests
        requests += 1
        if requests == 2:
            entered.set()
        await gate.wait()
        return httpx.Response(200, json={'object': 'list', 'model': 'embedding-model',
            'data': [{'object': 'embedding', 'index': 0, 'embedding': [1.0, 0.0]}],
            'usage': {'prompt_tokens': 1, 'total_tokens': 1}})

    value, sdk = client(monkeypatch, trace, handler)
    monkeypatch.setattr(VLLMClient, '_embed_semaphores',
                        {value.embed_url.rstrip('/'): threading.BoundedSemaphore(2)})
    loop = asyncio.get_running_loop()
    previous = loop._default_executor
    pool = ThreadPoolExecutor(max_workers=2)
    loop.set_default_executor(pool)
    tasks = []
    try:
        tasks = [asyncio.create_task(value._create_embedding_request(['doc'])) for _ in range(2)]
        await asyncio.wait_for(entered.wait(), timeout=3)
        tasks.extend(asyncio.create_task(value._create_embedding_request(['doc'])) for _ in range(58))
        await asyncio.sleep(0.1)
        gate.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
        assert len(results) == 60
        events = payloads(trace)
        assert sum(e['event'] == 'http.request' for e in events) == 60
        assert sum(e['event'] == 'http.response' for e in events) == 60
    finally:
        gate.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Cancelled semaphore waiters release their permits on the event loop.
        while any(thread.is_alive() for thread in pool._threads):
            pool.shutdown(wait=False)
            await asyncio.sleep(0.01)
        loop._default_executor = previous
        await sdk.close()
