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
    contract = question_contract('rewrite')
    try:
        if valid:
            assert await client.generate_json([{'role': 'user', 'content': 'rewrite'}], structured_contract=contract) == {'q_minus': ['Who?'], 'q_plus': []}
        else:
            with pytest.raises(StructuredOutputError) as caught:
                await client.generate_json([{'role': 'user', 'content': 'rewrite'}], structured_contract=contract)
            if finish_reason != 'stop' or refusal or content is None:
                metadata = json.loads(str(caught.value).split('metadata=', 1)[1])
                assert metadata['choice_count'] == 1
                assert metadata['finish_reasons'] == [finish_reason]
                assert metadata['effective_max_tokens'] == 512
                assert metadata['requested_max_tokens'] is None
                assert metadata['schema_name'] == 'prehop_rewrite_legacy_v1'
                assert len(metadata['schema_sha256']) == 64
                assert metadata['usage'] == {'prompt_tokens': 123, 'completion_tokens': 512,
                                             'total_tokens': 635, 'reasoning_tokens': 7}
                assert 'reasoning_content' not in str(caught.value)
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


@pytest.mark.asyncio
@pytest.mark.parametrize('count', [0, 2])
async def test_structured_choice_count_failure_is_distinct_and_metadata_only(monkeypatch, count):
    from types import SimpleNamespace
    sentinel = 'SECRET_RESPONSE_OR_REQUEST_BODY'
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
        message=SimpleNamespace(content=sentinel, reasoning_content=sentinel)) for _ in range(count)],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=11, total_tokens=20,
                              completion_tokens_details=None), id=sentinel)
    client = VLLMClient.__new__(VLLMClient)
    client.model_name = 'gemma-4-31b-it'
    client.vllm_url = 'http://gateway.test/v1'
    client.logger = logging.getLogger('structured_count_test')
    monkeypatch.setattr(client, '_get_cached_client', lambda url: None)
    monkeypatch.setattr(client, '_truncate_messages', lambda messages: messages)
    monkeypatch.setattr(client, '_resolve_output_token_limit', lambda value: 4096)
    monkeypatch.setattr(client, '_is_openai_model', lambda model: False)
    async def create(*args):
        return response
    monkeypatch.setattr(client, '_create_generation_request', create)
    with pytest.raises(StructuredOutputError, match='choice-count mismatch') as caught:
        await client.generate_response([{'role': 'user', 'content': sentinel}],
            response_format=question_contract('index').response_format())
    assert sentinel not in str(caught.value)
    metadata = json.loads(str(caught.value).split('metadata=', 1)[1])
    assert metadata['choice_count'] == count
    assert metadata['finish_reasons'] == ['stop'] * count
    assert metadata['effective_max_tokens'] == 4096
    assert metadata['requested_max_tokens'] is None


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


@pytest.mark.parametrize('text', ['', ' ', '\t\n', 'A', 'Who visited London?', '\nWho visited London?\n', '한글 질문인가요?', 'one\ntwo'])
def test_nonblank_schema_has_equivalent_search_fullmatch_and_local_semantics(text):
    import re

    from core.structured_outputs import NONBLANK_PATTERN
    expected = re.search(r'\S', text) is not None
    assert (re.search(NONBLANK_PATTERN, text) is not None) == expected
    assert (re.fullmatch(NONBLANK_PATTERN, text) is not None) == expected
    contract = question_contract('index')
    assert contract.schema()['properties']['q_minus']['items']['pattern'] == NONBLANK_PATTERN
    if expected:
        assert contract.validate({'q_minus': [text], 'q_plus': []})['q_minus'] == [text]
    else:
        with pytest.raises(StructuredOutputError):
            contract.validate({'q_minus': [text], 'q_plus': []})


def test_portable_nonblank_profile_changes_schema_index_query_and_cache_identity(monkeypatch):
    from core import structured_outputs
    from models.prehop.indexing.chunking import _generation_signature
    assert structured_outputs.PREHOP_STRUCTURED_PROFILE == 'prehop-json-schema-v3'
    assert canonical_semantic_index_policy('prehop', 'musique')['method_contract'] == 'paper-method-v3'
    current = structured_bundle_sha256()
    cache = _generation_signature('gemma-4-31b-it')
    monkeypatch.setattr(structured_outputs, 'PREHOP_STRUCTURED_PROFILE', 'prehop-json-schema-v1')
    assert structured_bundle_sha256() != current
    assert _generation_signature('gemma-4-31b-it') != cache


def test_every_materialized_schema_uses_reviewed_wire_keywords():
    from core.structured_outputs import validate_wire_schema
    contracts = [question_contract('index', mode) for mode in ('legacy', 'grounded_v1', 'linked_v2')]
    contracts += [question_contract(stage) for stage in ('rewrite', 'refine')]
    contracts += [ranking_contract(['A'], 1), ranking_contract(['A', 'B', 'C'], 2)]
    for contract in contracts:
        schema = contract.response_format()['json_schema']['schema']
        validate_wire_schema(schema)
        assert 'uniqueItems' not in json.dumps(schema)
        assert 'minLength' not in json.dumps(schema)
    assert contracts[-1].schema()['properties']['ranking']['minItems'] == 2
    assert contracts[-1].schema()['properties']['ranking']['maxItems'] == 2
    assert contracts[-2].schema()['properties']['ranking']['items']['const'] == 'A'


@pytest.mark.parametrize('key', ['uniqueItems', 'contains', 'minContains', 'maxContains',
                                  'multipleOf', 'patternProperties', 'propertyNames', 'minLength', 'maxLength'])
def test_unreviewed_schema_features_fail_before_transmission(key):
    from core.structured_outputs import validate_wire_schema
    schema = {'type': 'object', 'properties': {'ranking': {'type': 'array', 'items': {'type': 'string'}, key: True}}}
    with pytest.raises(StructuredOutputError, match='unsupported registered wire-schema keys'):
        validate_wire_schema(schema)
    # Field names and literal values are data, not schema keywords.
    validate_wire_schema({'type': 'object', 'properties': {key: {'type': 'string', 'enum': [key]}}})


@pytest.mark.asyncio
async def test_sdk_ranking_duplicate_is_rejected_locally_without_wire_unique_items(monkeypatch):
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'id': 'fixture', 'object': 'chat.completion', 'created': 0,
            'model': 'gemma-4-31b-it', 'choices': [{'index': 0, 'finish_reason': 'stop',
            'message': {'role': 'assistant', 'content': '{"ranking":["A","A"]}'}}]})
    sdk = AsyncOpenAI(base_url='http://gateway.test/v1', api_key='synthetic',
                      http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    client = VLLMClient.__new__(VLLMClient)
    client.model_name = 'gemma-4-31b-it'
    client.vllm_url = 'http://gateway.test/v1'
    client.logger = logging.getLogger('ranking_duplicates')
    monkeypatch.setattr(client, '_get_cached_client', lambda url: sdk)
    monkeypatch.setattr(client, '_truncate_messages', lambda messages: messages)
    monkeypatch.setattr(client, '_resolve_output_token_limit', lambda value: 1024)
    monkeypatch.setattr(client, '_is_openai_model', lambda model: False)
    async def create(sdk, params):
        return await sdk.chat.completions.create(**params)
    monkeypatch.setattr(client, '_create_generation_request', create)
    try:
        with pytest.raises(StructuredOutputError, match='duplicate candidate IDs'):
            await client.generate_json([{'role': 'user', 'content': 'Rank the candidates'}],
                                       structured_contract=ranking_contract(['A', 'B'], 2))
        assert len(requests) == 1
        array = requests[0]['response_format']['json_schema']['schema']['properties']['ranking']
        assert 'uniqueItems' not in array
        assert array['minItems'] == array['maxItems'] == 2
        assert array['items']['enum'] == ['A', 'B']
    finally:
        await sdk.close()
