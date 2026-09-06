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


@pytest.mark.parametrize('stage', ['index', 'rewrite', 'refine'])
def test_question_contract_is_recursive_strict_and_preserves_empty_directions(stage):
    contract = question_contract(stage)
    assert contract.validate({'q_minus': [], 'q_plus': []}) == {'q_minus': [], 'q_plus': []}
    for invalid in ({'q_minus': [3], 'q_plus': []}, {'q_minus': [' '], 'q_plus': []},
                    {'q_minus': [], 'q_plus': [], 'extra': True}, {'q_minus': []},
                    {'q_minus': ['a'] * 4, 'q_plus': []}):
        with pytest.raises(StructuredOutputError):
            contract.validate(invalid)


@pytest.mark.parametrize('schema', ['grounded_v1', 'linked_v2'])
def test_nested_grounded_schema_preserves_native_linked_empty_anchor(schema):
    entry = {'question': 'Who?', 'answer': 'Ada', 'grounding_quote': 'Ada visited.', 'anchor_entities': ['Ada']}
    if schema == 'linked_v2':
        entry.update(continuation_anchor='', anchor_entities=[])
    contract = question_contract('index', schema)
    assert contract.validate({'q_minus': [entry], 'q_plus': []})['q_minus'] == [entry]
    for change in ({'answer': 1}, {'unexpected': 'value'}, {'anchor_entities': [{}]}):
        with pytest.raises(StructuredOutputError):
            contract.validate({'q_minus': [{**entry, **change}], 'q_plus': []})


def test_ranking_exact_ids_count_and_uniqueness():
    contract = ranking_contract(['A', 'B', 'C'], 2)
    assert contract.validate({'ranking': ['C', 'A']}) == {'ranking': ['C', 'A']}
    for value in (['A'], ['A', 'A'], ['A', 'Z'], [1, 'A'], ['A', 'B', 'C']):
        with pytest.raises(StructuredOutputError):
            contract.validate({'ranking': value})
    assert contract.schema()['properties']['ranking']['items']['enum'] == ['A', 'B', 'C']


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
            'model': 'gemma-4-31b-it', 'choices': [{'index': 0, 'finish_reason': finish_reason,
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
    contract = question_contract('rewrite')
    try:
        if valid:
            assert await client.generate_json([{'role': 'user', 'content': 'rewrite'}], structured_contract=contract) == {'q_minus': ['Who?'], 'q_plus': []}
        else:
            with pytest.raises(StructuredOutputError):
                await client.generate_json([{'role': 'user', 'content': 'rewrite'}], structured_contract=contract)
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
        assert canonical_semantic_index_policy(strategy, 'musique')['structured_schema_bundle_sha256'] == digest
    before = _generation_signature('gemma-4-31b-it')
    monkeypatch.setattr('core.structured_outputs.structured_bundle_sha256', lambda: 'different schema')
    assert _generation_signature('gemma-4-31b-it') != before


@pytest.mark.asyncio
async def test_four_production_paths_supply_their_contract_and_valid_json_examples(monkeypatch):
    from unittest.mock import AsyncMock

    from core.config import RAGConfig
    from models.prehop.graphrag import GraphRAG

    monkeypatch.setattr(RAGConfig, 'QUESTION_SCHEMA', 'legacy')
    monkeypatch.setattr(RAGConfig, 'QUERY_REWRITE_VARIANT', 'role_aligned_evidence_iterative')
    monkeypatch.setattr(RAGConfig, 'QUERY_REWRITE_MAX_WORDS', 0)
    rag = GraphRAG(strategy='prehop')
    rag.llm = AsyncMock()
    rag.indexing_llm = AsyncMock()
    rag.indexing_llm.generate_json.return_value = {'q_minus': [], 'q_plus': []}
    await rag.extract_hoprag_queries('Ada visited London.', 'Ada')
    call = rag.indexing_llm.generate_json.await_args
    assert call.kwargs['structured_contract'].name == 'prehop_index_legacy_v1'
    # Native format example is a JSON object, not its escaped template spelling.
    assert '{{' not in call.args[0][-1]['content']
    rag.llm.generate_json.return_value = {'q_minus': ['Who visited?'], 'q_plus': []}
    await rag._rewrite_query_roles('Who visited?')
    assert rag.llm.generate_json.await_args.kwargs['structured_contract'].name == 'prehop_rewrite_legacy_v1'
    await rag._refine_query_roles('Who visited?', 'Ada visited.', [])
    assert rag.llm.generate_json.await_args.kwargs['structured_contract'].name == 'prehop_refine_legacy_v1'
    rag.llm.generate_json.return_value = {'ranking': ['C000']}
    await rag._role_body_list_ranking('Who visited?', [{'id': 'Ada', 'text': 'Ada visited.'}], 1)
    contract = rag.llm.generate_json.await_args.kwargs['structured_contract']
    assert contract.name == 'prehop_ranking_v1'
    assert contract.validate({'ranking': ['C000']}) == {'ranking': ['C000']}


@pytest.mark.asyncio
@pytest.mark.parametrize('index_schema', ['legacy', 'grounded_v1', 'linked_v2'])
async def test_actual_four_paths_through_typed_transport_and_sdk(monkeypatch, index_schema):
    import hashlib
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
                       'RAG_GENERATION_MODEL': 'gemma-4-31b-it', 'RAG_EMBEDDING_MODEL': 'qwen3-embedding-8b',
                       'RAG_PAPER_MODE': 'true', 'RAG_SKIP_PROJECT_ENV': 'true', 'RAG_LLM_SEED': '42'}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr('core.inference_transport._approved_gateway_identity',
                        lambda: hashlib.sha256(b'http://litellm.test/v1').hexdigest())
    monkeypatch.setattr(RAGConfig, 'LLM_SEED', None)  # module import preceded late environment resolution
    monkeypatch.setattr(RAGConfig, 'QUESTION_SCHEMA', index_schema)
    monkeypatch.setattr(RAGConfig, 'QUERY_REWRITE_VARIANT', 'role_aligned_evidence_iterative')
    monkeypatch.setattr(RAGConfig, 'QUERY_REWRITE_MAX_WORDS', 0)
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
        await rag._rewrite_query_roles('Who visited?')
        await rag._refine_query_roles('Who visited?', 'Ada visited.', [])
        await rag._role_body_list_ranking('Who visited?', [{'id': 'Ada', 'text': 'Ada visited.'}], 1)
        assert [p['response_format']['json_schema']['name'] for p in requests] == [
            f'prehop_index_{index_schema}_v1', 'prehop_rewrite_legacy_v1', 'prehop_refine_legacy_v1', 'prehop_ranking_v1']
        assert all(p['seed'] == 42 and p['model'] == 'gemma-4-31b-it' for p in requests)
        from core.generation_profiles import request_settings
        for request, consumer in zip(requests, ('question_index', 'rewrite', 'refine', 'ranking'), strict=True):
            assert all(request[key] == value for key, value in request_settings(consumer).items())
        assert all(p['response_format']['json_schema']['strict'] is True for p in requests)
        assert await client.generate_response([{'role': 'user', 'content': 'answer'}]) == 'Plain answer'
        assert 'response_format' not in requests[-1]
    finally:
        await sdk.close()
