import json
from types import SimpleNamespace

import pytest

from models.external_research.extraction_contract import (
    ExtractionFormatError,
    normalize_entities,
    normalize_triples,
    validate_extraction,
)
from models.ms_graphrag.extraction_guard import validate_records


@pytest.mark.parametrize('values', [[1], [None], [True], [''], ['  '], [['A']], [{'name': 'A'}],
                                    [{'entity': 'A', 'text': 'B'}], [{'entity': 'A', 'type': None}]])
def test_bad_entities_never_pass_as_native_success(values):
    with pytest.raises(ExtractionFormatError):
        normalize_entities(values)


@pytest.mark.parametrize('values', [['ABC'], [1], [[1, True, None]], [['A', 'B']],
                                    [['A', 'B', 'C', 'D']], [[{'name': 'A'}, 'B', 'C']]])
def test_bad_triples_never_become_key_or_character_edges(values):
    with pytest.raises(ExtractionFormatError):
        normalize_triples(values)


def test_explicit_triple_objects_map_values_in_contract_order():
    assert normalize_triples([{'object': 'B', 'subject': 'A', 'predicate': 'knows'},
                              {'tail': 'B', 'relation': 'knows', 'head': 'A'}]) == [['A', 'knows', 'B']]


@pytest.mark.parametrize('raw', ['{"named_entities": ["A"], "named_entities": []}',
                               '{"named_entities": [NaN]}', '{"named_entities": ["A"',
                               '{"named_entities": []} {}', '["A"]'])
def test_incomplete_duplicate_nonfinite_and_ambiguous_json_rejected(raw):
    with pytest.raises(ExtractionFormatError):
        validate_extraction(raw, 'named_entities', 'stop')


def test_truncation_is_not_silently_repaired_even_if_prefix_is_valid_json():
    with pytest.raises(ExtractionFormatError):
        validate_extraction('{"named_entities":[]}', 'named_entities', 'length')
    assert json.loads(validate_extraction('{"named_entities":[]}', 'named_entities', 'stop')) == {'named_entities': []}


@pytest.mark.parametrize('raw', ['("entity"<|>A<|>Person)', '("relationship"<|>A<|>B<|>knows<|>nan)',
                               '("entity"<|><|>Person<|>description)', 'prose'])
def test_ms_malformed_records_fail_before_native_drop(raw):
    with pytest.raises(ExtractionFormatError):
        validate_records(raw + '<|COMPLETE|>')


def test_ms_valid_records_keep_native_content():
    raw = '("entity"<|>A<|>Person<|>description)##("relationship"<|>A<|>B<|>knows<|>1)<|COMPLETE|>'
    assert validate_records(raw) == ({'A'}, [('A', 'B')])


def test_hippo_retry_bypasses_invalid_cache_and_accounts_all_attempts(tmp_path):
    from models.external_research.drivers.hippo_extraction import install_extraction_adapter

    class Native:
        calls = 0

        def infer(self, **kwargs):
            self.calls += 1
            self.kwargs = kwargs
            return '{}', {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12,
                          'finish_reason': 'stop'}, False

    def uncached(self, **kwargs):
        self.calls += 1
        assert kwargs == self.kwargs
        return '{"named_entities":[{"entity":"A","label":"Person"}]}', {
            'prompt_tokens': 10, 'completion_tokens': 7, 'total_tokens': 17, 'finish_reason': 'stop'}

    Native.infer.__wrapped__ = uncached
    llm = Native()
    def ner():
        raw, metadata, cache = openie.llm_model.infer(messages=[], max_new_tokens=512)
        assert json.loads(raw) == {'named_entities': ['A']}
        assert not cache and metadata['total_tokens'] == 29
        return SimpleNamespace(response=raw, metadata=metadata)
    openie = SimpleNamespace(llm_model=llm, ner_max_tokens=512, triple_max_tokens=2048,
                             ner=ner, triple_extraction=lambda: None)
    audit = install_extraction_adapter(openie, tmp_path / 'audit.jsonl', 2)
    result = openie.ner()
    assert json.loads(result.response)['named_entities'][0]['entity'] == 'A'
    assert llm.calls == 2
    assert llm.kwargs['response_format']['type'] == 'json_schema'
    rows = [json.loads(line) for line in audit.path.read_text().splitlines()]
    assert [row['status'] for row in rows] == ['retry', 'accepted']
    audit.assert_healthy()


def test_hippo_exhaustion_is_durable_and_not_admissible(tmp_path):
    from models.external_research.drivers.hippo_extraction import install_extraction_adapter

    class Native:
        def infer(self, **kwargs):
            return '{"triples":[]}', {'finish_reason': 'length'}, False

    Native.infer.__wrapped__ = lambda self, **kwargs: ('{"triples":[]}', {'finish_reason': 'length'})
    native = Native()
    openie = SimpleNamespace(llm_model=native, ner_max_tokens=512, triple_max_tokens=2048,
                             ner=lambda: None, triple_extraction=lambda: None)
    audit = install_extraction_adapter(openie, tmp_path / 'audit.jsonl', 2)
    with pytest.raises(ExtractionFormatError):
        openie.llm_model.infer(messages=[], max_new_tokens=2048)
    with pytest.raises(RuntimeError, match='exhausted'):
        audit.assert_healthy()
    rows = [json.loads(line) for line in audit.path.read_text().splitlines()]
    assert [r['status'] for r in rows] == ['retry', 'exhausted']


def test_audit_concurrent_calls_are_individually_readable(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from models.external_research.extraction_contract import ExtractionAudit

    audit = ExtractionAudit(tmp_path / 'audit.jsonl')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: audit.write(messages=[str(i)], response=str(i), status='accepted'), range(100)))
    rows = [json.loads(line) for line in audit.path.read_text().splitlines()]
    assert len(rows) == 100 and {r['response'] for r in rows} == {str(i) for i in range(100)}


def test_index_audit_binding_survives_query_append_but_detects_tampering(tmp_path):
    from models.external_research.extraction_contract import ExtractionAudit, audit_evidence, validate_audit_evidence

    audit = ExtractionAudit(tmp_path / 'audit.jsonl')
    audit.write(messages=['index'], response='original', status='accepted')
    evidence = audit_evidence(audit)
    audit.write(messages=['query'], response='answer', status='accepted')
    validate_audit_evidence(evidence)
    audit.path.write_text(audit.path.read_text().replace('original', 'modified'))
    with pytest.raises(ValueError, match='changed'):
        validate_audit_evidence(evidence)


def test_missing_and_exhausted_audit_cannot_become_index_evidence(tmp_path):
    from models.external_research.extraction_contract import ExtractionAudit, audit_evidence, validate_audit_evidence

    with pytest.raises(ValueError):
        validate_audit_evidence(None)
    audit = ExtractionAudit(tmp_path / 'audit.jsonl')
    audit.write(messages=[], status='exhausted', error='bad format')
    with pytest.raises(RuntimeError):
        audit_evidence(audit)


def test_gfm_guard_retries_before_native_eval_and_preserves_raw_text(tmp_path):
    from pydantic import BaseModel

    from models.external_research.drivers.gfm_extraction import guarded_chat_type
    from models.external_research.extraction_contract import ExtractionAudit

    class Reply(BaseModel):
        content: str
        response_metadata: dict = {'finish_reason': 'stop'}

    class FakeChat(BaseModel):
        replies: list
        calls: list = []

        def invoke(self, input, config=None, **kwargs):
            self.calls.append(kwargs)
            return self.replies.pop(0)

    raw = '{"triples":[{"subject":"Ada","predicate":"visited","object":"London"}]}'
    audit = ExtractionAudit(tmp_path / 'audit.jsonl')
    model = guarded_chat_type(FakeChat)(replies=[Reply(content='__import__("os").system("false")'), Reply(content=raw)])
    model.configure_validation(audit, 2, {1024}, 4096)
    result = model.invoke([], max_tokens=4096)
    assert json.loads(result.content) == {'triples': [['Ada', 'visited', 'London']]}
    assert model.calls[0] == model.calls[1]
    assert model.calls[0]['response_format']['type'] == 'json_schema'
    rows = [json.loads(line) for line in audit.path.read_text().splitlines()]
    assert rows[0]['status'] == 'retry' and rows[1]['response'] == raw
    audit.assert_healthy()


def test_gfm_guard_records_exhaustion_even_if_native_caller_swallows_it(tmp_path):
    from pydantic import BaseModel

    from models.external_research.drivers.gfm_extraction import guarded_chat_type
    from models.external_research.extraction_contract import ExtractionAudit

    class FakeChat(BaseModel):
        def invoke(self, input, config=None, **kwargs):
            return SimpleNamespace(content='{}', response_metadata={'finish_reason': 'stop'})

    audit = ExtractionAudit(tmp_path / 'audit.jsonl')
    model = guarded_chat_type(FakeChat)().configure_validation(audit, 1, {1024}, 4096)
    try:
        model.invoke([], max_tokens=1024)
    except ValueError:
        pass
    with pytest.raises(RuntimeError, match='exhausted'):
        audit.assert_healthy()


@pytest.mark.asyncio
@pytest.mark.parametrize('finish', ['stop', 'length'])
async def test_ms_stream_preserves_chunks_and_rejects_incomplete_completion(monkeypatch, tmp_path, finish):
    from openai.types.chat import ChatCompletionChunk

    factory = pytest.importorskip('graphrag_llm.completion.completion_factory')
    native_module = pytest.importorskip('graphrag_llm.completion.lite_llm_completion')
    from models.external_research.extraction_contract import ExtractionAudit
    from models.ms_graphrag.extraction_guard import register_guard

    chunks = [ChatCompletionChunk(id='fixture', object='chat.completion.chunk', created=0, model='fixture',
              choices=[{'index': 0, 'delta': {'content': 'Answer'}, 'finish_reason': None}]),
              ChatCompletionChunk(id='fixture', object='chat.completion.chunk', created=0, model='fixture',
              choices=[{'index': 0, 'delta': {}, 'finish_reason': finish}])]
    async def stream():
        for chunk in chunks:
            yield chunk
    class Base:
        def __init__(self, **kwargs):
            self._completion = lambda **kwargs: None
            async def completion(**kwargs):
                return stream()
            self._completion_async = completion
    captured = []
    monkeypatch.setattr(native_module, 'LiteLLMCompletion', Base)
    monkeypatch.setattr(factory, 'register_completion', lambda name, implementation: captured.append(implementation))
    audit = ExtractionAudit(tmp_path / 'audit.jsonl')
    register_guard(audit, 2)
    provider = captured[0]()
    iterator = await provider._completion_async(messages=[{'role': 'user', 'content': 'question'}], stream=True)
    observed = []
    if finish == 'stop':
        async for chunk in iterator:
            observed.append(chunk)
        audit.assert_healthy()
    else:
        with pytest.raises(ExtractionFormatError):
            async for chunk in iterator:
                observed.append(chunk)
        with pytest.raises(RuntimeError):
            audit.assert_healthy()
    assert all(a is b for a, b in zip(chunks, observed)) and len(observed) == len(chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize('asynchronous', [False, True])
async def test_ms_length_failure_is_audited_without_identical_retries(monkeypatch, tmp_path, asynchronous):
    from types import SimpleNamespace

    from models.external_research.extraction_contract import ExtractionAudit
    from models.ms_graphrag.extraction_guard import OutputTruncatedError, register_guard

    factory = pytest.importorskip('graphrag_llm.completion.completion_factory')
    native_module = pytest.importorskip('graphrag_llm.completion.lite_llm_completion')
    calls = []
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='length')],
                               model_dump=lambda: {'choices': [{'finish_reason': 'length'}]})
    class Base:
        def __init__(self, **kwargs):
            def sync(**kwargs):
                calls.append(kwargs)
                return response
            async def async_call(**kwargs):
                return sync(**kwargs)
            self._completion, self._completion_async = sync, async_call
    captured = []
    monkeypatch.setattr(native_module, 'LiteLLMCompletion', Base)
    monkeypatch.setattr(factory, 'register_completion', lambda name, implementation: captured.append(implementation))
    audit = ExtractionAudit(tmp_path / 'audit.jsonl')
    register_guard(audit, 5)
    provider = captured[0]()
    with pytest.raises(OutputTruncatedError):
        if asynchronous:
            await provider._completion_async(messages=[{'role': 'user', 'content': 'extract'}])
        else:
            provider._completion(messages=[{'role': 'user', 'content': 'extract'}])
    assert len(calls) == 1 and audit.exhausted == 1
    rows = [json.loads(line) for line in audit.path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['status'] == 'exhausted'
