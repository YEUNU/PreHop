"""Actual SDK transport proves one bounded format budget and complete response accounting."""
import json
import logging

import httpx
import pytest
from openai import AsyncOpenAI, InternalServerError

from core.config import RAGConfig
from core.inference_telemetry import begin, finish
from core.structured_outputs import StructuredOutputError, question_contract
from core.vllm_client import VLLMClient


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', ['{bad'])
@pytest.mark.parametrize('exhausted', [False, True])
async def test_actual_transport_format_attempts_are_identical_and_all_usage_is_counted(monkeypatch, bad, exhausted):
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        content = bad if exhausted or len(requests) == 1 else '{"q_minus":[],"q_plus":[]}'
        return httpx.Response(200, json={'id': 'fixture', 'object': 'chat.completion', 'created': 0,
            'model': 'gemma-4-31b-it', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': content}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}, 'response_cost': 0.25})
    sdk = AsyncOpenAI(base_url='http://retry.test/v1', api_key='synthetic',
                      http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    client = configured_client(monkeypatch, sdk)
    token = begin()
    try:
        if exhausted:
            with pytest.raises(StructuredOutputError, match='metadata='):
                await client.generate_json([{'role': 'user', 'content': 'unchanged'}],
                                           structured_contract=question_contract('index'), max_tokens=4096, temperature=0)
        else:
            assert await client.generate_json([{'role': 'user', 'content': 'unchanged'}],
                structured_contract=question_contract('index'), max_tokens=4096, temperature=0) == {'q_minus': [], 'q_plus': []}
        totals = finish(token)
        expected = 5 if exhausted else 2
        assert len(requests) == expected and all(row == requests[0] for row in requests)
        assert totals['generation_calls'] == totals['structured_attempt_count'] == expected
        assert totals['prompt_tokens'] == 10 * expected and totals['completion_tokens'] == 5 * expected
        assert totals['reported_cost'] == .25 * expected and totals['cost_complete']
        assert totals['structured_elapsed_seconds'] > 0
        failures = totals['structured_invalid_attempts']
        assert len(failures) == (5 if exhausted else 1)
        assert len({row['request_sha256'] for row in failures}) == 1
        assert [row['wire_attempts_used'] for row in failures] == list(range(1, len(failures) + 1))
    finally:
        await sdk.close()


def configured_client(monkeypatch, sdk):
    monkeypatch.setenv('RAG_PAPER_MODE', 'false')
    monkeypatch.setattr(RAGConfig, 'LLM_MAX_RETRIES', 5)
    monkeypatch.setattr(RAGConfig, 'LLM_RETRY_DELAY', 0)
    client = VLLMClient.__new__(VLLMClient)
    client.model_name = 'gemma-4-31b-it'
    client.vllm_url = 'http://retry.test/v1'
    client._retry_attempts = 5
    client._generation_concurrency = 1
    client.logger = logging.getLogger('format_retry')
    monkeypatch.setattr(client, '_get_cached_client', lambda url: sdk)
    monkeypatch.setattr(client, '_truncate_messages', lambda messages: messages)
    monkeypatch.setattr(client, '_resolve_output_token_limit', lambda limit: limit)
    return client


@pytest.mark.asyncio
async def test_transport_and_format_share_five_wire_attempts_and_disable_sdk_retries(monkeypatch):
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) in {1, 3, 5}:
            return httpx.Response(503, json={'error': {'message': 'fixture unavailable'}})
        return httpx.Response(200, json={'id': 'fixture', 'object': 'chat.completion', 'created': 0,
            'model': 'gemma-4-31b-it', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': '{invalid'}}]})
    sdk = AsyncOpenAI(base_url='http://retry.test/v1', api_key='synthetic',
                      http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    client = configured_client(monkeypatch, sdk)
    token = begin()
    try:
        with pytest.raises(InternalServerError):
            await client.generate_json([{'role': 'user', 'content': 'unchanged'}],
                                       structured_contract=question_contract('index'), max_tokens=512)
        totals = finish(token)
        assert len(requests) == totals['generation_calls'] == 5
        assert all(row == requests[0] for row in requests)
        assert totals['generation_transport_failures'] == 3
        assert totals['structured_attempt_count'] == 2
        assert not totals['token_usage_complete'] and not totals['cost_complete']
    finally:
        await sdk.close()


def test_retry_changes_only_prehop_identity_and_its_chunk_cache(monkeypatch):
    from core.generation_profiles import generation_profiles
    from core.paper_compatibility import method_identity
    from models.prehop.indexing.chunking import _generation_signature
    naive = method_identity('naive')
    prehop = method_identity('prehop')
    cache = _generation_signature('gemma-4-31b-it')
    assert 'structured_format_retry' not in generation_profiles('naive')
    monkeypatch.setattr(RAGConfig, 'LLM_MAX_RETRIES', RAGConfig.LLM_MAX_RETRIES + 1)
    assert method_identity('naive') == naive
    assert method_identity('prehop') != prehop
    assert _generation_signature('gemma-4-31b-it') != cache
