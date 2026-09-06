"""Malformed structured responses retain only safe, request-local replay identities."""
import asyncio
import json
import logging

import httpx
import pytest
from openai import AsyncOpenAI

from core.structured_diagnostics import caller_metadata, indexing_failure_scope, json_sha256, text_sha256
from core.structured_outputs import StructuredOutputError, question_contract
from core.vllm_client import VLLMClient


@pytest.mark.asyncio
@pytest.mark.parametrize(('raw', 'category', 'reason'), [
    ('{"q_minus":[],"q_plus":[]} TRAILING_SECRET', 'json_syntax', 'extra_data'),
    ('{"q_minus":["UNTERMINATED_SECRET', 'json_syntax', 'unterminated_string'),
    ('{"SECRET_PROPERTY":1,"SECRET_PROPERTY":2}', 'duplicate_property', None),
    ('{"q_minus":NaN,"q_plus":[]}', 'nonfinite_constant', None),
    ('{"q_minus":Infinity,"q_plus":[]}', 'nonfinite_constant', None),
    ('{"q_minus":["문서\u0001SECRET_CONTROL"],"q_plus":[]}', 'json_syntax', 'invalid_control_character'),
])
async def test_actual_sdk_parse_failure_has_only_safe_metadata_and_bounded_identical_requests(monkeypatch, raw, category, reason):
    requests = []
    sdk_parameters = []
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'id': 'SECRET_RESPONSE_ID', 'object': 'chat.completion',
            'created': 1, 'model': 'gemma-4-31b-it', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': raw}}],
            'usage': {'prompt_tokens': 123, 'completion_tokens': 27, 'total_tokens': 150}})
    sdk = AsyncOpenAI(api_key='SECRET_API_KEY', base_url='http://fixture.test/v1',
                      http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    client = VLLMClient.__new__(VLLMClient)
    client.model_name = 'gemma-4-31b-it'
    client.vllm_url = 'http://fixture.test/v1'
    client.logger = logging.getLogger('parse_failure_test')
    monkeypatch.setattr(client, '_get_cached_client', lambda url: sdk)
    monkeypatch.setattr(client, '_truncate_messages', lambda messages: messages[-1:])
    monkeypatch.setattr(client, '_resolve_output_token_limit', lambda value: value)
    monkeypatch.setattr(client, '_is_openai_model', lambda model: False)
    async def create(request_client, params):
        sdk_parameters.append(params)
        return await request_client.chat.completions.create(**params)
    monkeypatch.setattr(client, '_create_generation_request', create)
    messages = [{'role': 'system', 'content': 'SECRET_TRUNCATED_CONTEXT'},
                {'role': 'user', 'content': 'SECRET_NATIVE_PROMPT'}]
    contract = question_contract('index')
    try:
        with (
            indexing_failure_scope('SECRET_FILENAME', 'SECRET_DOCUMENT', 'SECRET_TITLE', 'SECRET_CHUNK', 3, 7),
            pytest.raises(StructuredOutputError) as caught,
        ):
            await client.generate_json(messages, structured_contract=contract, max_tokens=4096,
                                       temperature=0, json_debug_label='SECRET_STAGE')
        diagnostic = str(caught.value)
        assert 'SECRET' not in diagnostic and 'fixture.test' not in diagnostic
        metadata = json.loads(diagnostic.split('metadata=', 1)[1])
        failure = metadata['parse_failure']
        assert failure['category'] == category
        if reason is not None:
            assert failure['reason'] == reason
            assert all(type(failure[key]) is int for key in ('offset', 'line', 'column'))
        assert failure['content']['characters'] == len(raw)
        assert failure['content']['sha256'] == text_sha256(raw)
        assert metadata['prompt_sha256'] == json_sha256(messages[-1:])
        assert metadata['caller_prompt_sha256'] == json_sha256(messages)
        assert metadata['request_sha256'] == json_sha256(sdk_parameters[0])
        assert metadata['schema_sha256'] == contract.provenance()['schema_sha256']
        assert metadata['caller'] == {'source_sha256': text_sha256('SECRET_FILENAME'),
            'document_sha256': text_sha256('SECRET_DOCUMENT'), 'title_sha256': text_sha256('SECRET_TITLE'),
            'chunk_sha256': text_sha256('SECRET_CHUNK'), 'page': 3, 'page_chunk_index': 7}
        assert metadata['choice_count'] == 1 and metadata['finish_reasons'] == ['stop']
        assert metadata['usage']['prompt_tokens'] == 123 and metadata['usage']['completion_tokens'] == 27
        assert metadata['effective_max_tokens'] == metadata['requested_max_tokens'] == 4096
        assert len(requests) == 5
        assert all(request == requests[0] for request in requests)
        assert requests[0]['messages'] == messages[-1:] and requests[0]['temperature'] == 0
        assert requests[0]['response_format'] == contract.response_format()
        assert not any('diagnostic' in key or 'debug' in key or 'metadata' in key for key in requests[0])
        assert caller_metadata() == {}
    finally:
        await sdk.close()


@pytest.mark.asyncio
async def test_input_hash_scopes_are_isolated_between_concurrent_documents():
    entered = asyncio.Event()
    async def read(name):
        with indexing_failure_scope(name, name, name, name, 1, 0):
            entered.set()
            await entered.wait()
            await asyncio.sleep(0)
            return caller_metadata()
    left, right = await asyncio.gather(read('left'), read('right'))
    assert left['document_sha256'] == text_sha256('left')
    assert right['document_sha256'] == text_sha256('right')
    assert caller_metadata() == {}
