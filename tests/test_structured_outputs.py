"""Strict request schemas, raw response boundaries and semantic identity."""
import json
import logging

import httpx
import pytest
from openai import AsyncOpenAI

from core.inference_telemetry import begin, finish
from core.paper_policy import canonical_query_policy, canonical_semantic_index_policy
from core.structured_outputs import StructuredOutputError, question_contract, ranking_contract, structured_bundle_sha256
from core.vllm_client import VLLMClient


@pytest.mark.asyncio
@pytest.mark.parametrize(('content', 'finish_reason', 'refusal', 'valid'), [
    ('{"q_minus":["Who?"],"q_plus":[]}', 'stop', None, True),
    ('```json\n{"q_minus":[],"q_plus":[]}\n```', 'stop', None, False),
    ('prefix {"q_minus":[],"q_plus":[]}', 'stop', None, False),
    ('{"q_minus":[],"q_plus":[]}', 'length', None, False),
    ('{"q_minus":[],"q_plus":[]}', 'stop', 'refused', False),
    (None, 'stop', None, False),
    ('{"q_minus":[3],"q_plus":[]}', 'stop', None, False),
    ('{"q_minus":[],"q_minus":[],"q_plus":[]}', 'stop', None, False),
])
async def test_real_sdk_strict_raw_response_has_no_repair_or_downgrade(monkeypatch, content, finish_reason, refusal, valid):
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0,
            'model': 'gemma-4-31b-it', 'usage': {'prompt_tokens': 123, 'completion_tokens': 512,
                'total_tokens': 635, 'completion_tokens_details': {'reasoning_tokens': 7}},
            'choices': [{'index': 0, 'finish_reason': finish_reason,
            'message': {'role': 'assistant', 'content': content, 'refusal': refusal,
                        'reasoning_content': '{"q_minus":[],"q_plus":[]}'}}]})
    sdk = AsyncOpenAI(base_url='http://gateway.test/v1', api_key='synthetic',
                      http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    client = VLLMClient.__new__(VLLMClient)
    client.model_name = 'gemma-4-31b-it'
    client.vllm_url = "http://gateway.test/v1"
    monkeypatch.setattr(client, "_get_cached_client", lambda url: sdk)
    client.logger = logging.getLogger('structured_test')
    monkeypatch.setattr(client, '_truncate_messages', lambda messages: messages)
    monkeypatch.setattr(client, '_resolve_output_token_limit', lambda value: value or 512)
    monkeypatch.setattr(client, '_is_openai_model', lambda model: False)
    async def create(sdk, params):
        return await sdk.chat.completions.create(**params)
    monkeypatch.setattr(client, '_create_generation_request', create)
    token = begin()
    contract = question_contract('index')
    try:
        try:
            expected = json.loads(content)
        except (TypeError, ValueError):
            with pytest.raises((TypeError, ValueError, StructuredOutputError)):
                await client.generate_json([{'role': 'user', 'content': 'rewrite'}], structured_contract=contract)
            assert len(requests) == 5
        else:
            assert await client.generate_json([{'role': 'user', 'content': 'rewrite'}], structured_contract=contract) == expected
            assert len(requests) == 1
        assert requests[0]['response_format'] == contract.response_format()
        assert finish(token)['structured_output_contracts'] == [contract.provenance()]
    finally:
        await sdk.close()


def test_policy_and_cache_bind_structured_factory(monkeypatch):
    from models.prehop.indexing.chunking import _generation_signature

    digest = structured_bundle_sha256()
    for strategy in ('prehop', 'naive'):
        assert canonical_query_policy(strategy)['structured_schema_bundle_sha256'] == digest
        from core.structured_outputs import structured_index_bundle_sha256
        assert canonical_semantic_index_policy(strategy, 'hotpotqa')['structured_schema_bundle_sha256'] == structured_index_bundle_sha256()
    before = _generation_signature('gemma-4-31b-it')
    monkeypatch.setattr('core.structured_outputs.structured_index_bundle_sha256', lambda: 'different schema')
    assert _generation_signature('gemma-4-31b-it') != before


@pytest.mark.asyncio
@pytest.mark.parametrize('index_schema', ['legacy', 'grounded_v1', 'linked_v2'])
async def test_actual_two_paths_through_typed_transport_and_sdk(monkeypatch, index_schema):
    import os

    from core.config import RAGConfig
    from core.strategy_registry import paper_environment_defaults
    from models.prehop.graphrag import GraphRAG

    for name in tuple(os.environ):
        if name.startswith(('RAG_', 'VLLM_', 'OPENAI_', 'AZURE_', 'EMBEDDING_', 'MAX_EMBEDDING', 'NEO4J_VECTOR')):
            monkeypatch.delenv(name, raising=False)
    for key, value in paper_environment_defaults().items():
        monkeypatch.setenv(key, value)
    for key, value in {'RAG_INFERENCE_BASE_URL': 'http://litellm.test/v1', 'RAG_INFERENCE_API_KEY': 'synthetic',
                       'RAG_GENERATION_MODEL': 'gemma-4-31b-it', 'RAG_EMBEDDING_MODEL': 'qwen3-embedding-4b',
                       'RAG_PAPER_MODE': 'true', 'RAG_SKIP_PROJECT_ENV': 'true', 'RAG_LLM_SEED': '42'}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(RAGConfig, 'LLM_SEED', None)  # module import preceded late environment resolution
    monkeypatch.setattr(RAGConfig, 'QUESTION_SCHEMA', index_schema)
    requests = []
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        schema_name = payload.get('response_format', {}).get('json_schema', {}).get('name', '')
        content = {'ranking': ['C000']} if 'ranking' in schema_name else {'q_minus': [], 'q_plus': []}
        return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0,
             'model': 'gemma-4-31b-it', 'choices': [{'index': 0, 'finish_reason': 'stop',
             'message': {'role': 'assistant', 'content': json.dumps(content) if schema_name else 'Plain answer'}}]})
    sdk = AsyncOpenAI(base_url='http://litellm.test/v1', api_key='synthetic',
                      http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    client = VLLMClient()
    monkeypatch.setattr(client, '_get_cached_client', lambda url: sdk)
    rag = GraphRAG.__new__(GraphRAG)
    rag.llm = rag.indexing_llm = client
    try:
        await rag.extract_hoprag_queries('Ada visited London.', 'Ada')
        await rag._role_body_list_ranking('Who visited?', [{'id': 'Ada', 'text': 'Ada visited.'}], 1)
        assert [p['response_format']['json_schema']['name'] for p in requests] == [
            f'prehop_index_{index_schema}_v1', 'prehop_ranking_v1']
        assert all('seed' not in p and p['model'] == 'gemma-4-31b-it' for p in requests)
        from core.generation_profiles import request_settings
        for request, consumer in zip(requests, ('question_index', 'ranking'), strict=True):
            assert all(request[key] == value for key, value in request_settings(consumer).items())
        assert all(p['response_format']['json_schema']['strict'] is True for p in requests)
        assert await client.generate_response([{'role': 'user', 'content': 'answer'}]) == 'Plain answer'
        assert 'response_format' not in requests[-1]
    finally:
        await sdk.close()


def test_structured_diagnostics_drop_unknown_strings_and_invalid_usage():
    from types import SimpleNamespace
    sentinel = 'SECRET_UNKNOWN_PROVIDER_METADATA'
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason=sentinel)],
        usage=SimpleNamespace(prompt_tokens=sentinel, completion_tokens=-1, total_tokens=True,
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=sentinel)))
    raw = VLLMClient._structured_response_diagnostics(response, {'max_tokens': 4096, 'api_key': sentinel})
    assert sentinel not in raw
    metadata = json.loads(raw)
    assert metadata['finish_reasons'] == ['unknown']
    assert set(metadata['usage'].values()) == {None}


def test_portable_nonblank_profile_changes_schema_index_query_and_cache_identity(monkeypatch):
    from core import structured_outputs
    from models.prehop.indexing.chunking import _generation_signature
    assert structured_outputs.PREHOP_STRUCTURED_PROFILE == 'prehop-json-schema-v3'
    assert canonical_semantic_index_policy('prehop', 'hotpotqa')['method_contract'] == 'paper-method-v5'
    current = structured_bundle_sha256()
    cache = _generation_signature('gemma-4-31b-it')
    monkeypatch.setattr(structured_outputs, 'PREHOP_STRUCTURED_PROFILE', 'prehop-json-schema-v1')
    assert structured_bundle_sha256() != current
    assert _generation_signature('gemma-4-31b-it') != cache


def test_every_materialized_schema_uses_reviewed_wire_keywords():
    contracts = [question_contract('index', mode) for mode in ('legacy', 'grounded_v1', 'linked_v2')]
    contracts += [ranking_contract(['A'], 1), ranking_contract(['A', 'B', 'C'], 2)]
    for contract in contracts:
        schema = contract.response_format()['json_schema']['schema']
        assert 'uniqueItems' not in json.dumps(schema)
        assert 'minLength' not in json.dumps(schema)
    assert contracts[-1].schema()['properties']['ranking']['minItems'] == 2
    assert contracts[-1].schema()['properties']['ranking']['maxItems'] == 2
    assert contracts[-2].schema()['properties']['ranking']['items']['const'] == 'A'
