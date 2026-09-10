import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from models.external_research.extraction_contract import ExtractionAudit, audit_evidence, validate_audit_evidence
from models.external_research.native_observation import PROFILE, observed_chat_type


@pytest.mark.parametrize('content', ['{"named_entities":[{"entity":"A"}]}', '{"named_entities":', '', 'explanation\n("entity"<|>A)'])
def test_gfm_passes_native_response_and_options_unchanged(tmp_path, content):
    calls = []
    reply = SimpleNamespace(content=content, response_metadata={'finish_reason': 'length'})
    class Native(BaseModel):
        def invoke(self, input, config=None, **kwargs):
            calls.append(kwargs)
            return reply
    audit = ExtractionAudit(tmp_path/'audit.jsonl', profile=PROFILE)
    client = observed_chat_type(Native)().configure_validation(audit, 5, {300}, 1024)
    assert client.invoke([], max_tokens=300, response_format={'type':'json_object'}) is reply
    assert calls == [{'max_tokens':300, 'response_format':{'type':'json_object'}}]
    validate_audit_evidence(audit_evidence(audit))


def test_native_exception_is_rethrown_once_and_does_not_override_native_fallback(tmp_path):
    calls = []
    error = RuntimeError('length limit reached')
    class Native(BaseModel):
        def invoke(self, input, config=None, **kwargs):
            calls.append(kwargs)
            raise error
    audit = ExtractionAudit(tmp_path/'audit.jsonl', profile=PROFILE)
    client = observed_chat_type(Native)().configure_validation(audit, 5, {300}, 1024)
    try:
        client.invoke([], max_tokens=300)
    except RuntimeError as exc:
        assert exc is error
        native_result = []
    assert native_result == [] and len(calls) == 1
    audit.assert_healthy()
    evidence = audit_evidence(audit)
    assert evidence['counts'] == {'native_exception':1}
    validate_audit_evidence(evidence)
    audit.path.write_text(audit.path.read_text().replace('length limit', 'changed text'))
    with pytest.raises(ValueError): validate_audit_evidence(evidence)


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.asyncio
async def test_ms_observer_keeps_preamble_truncation_and_cache(monkeypatch, tmp_path, asynchronous):
    factory = pytest.importorskip('graphrag_llm.completion.completion_factory')
    native_module = pytest.importorskip('graphrag_llm.completion.lite_llm_completion')
    from models.external_research.native_observation import register_ms_observer
    from models.ms_graphrag.extraction_guard import PROVIDER
    calls = []
    class Response:
        content = 'Preamble\n("entity"<|>A<|>person<|>description)'
        def model_dump(self): return {'content':self.content, 'finish_reason':'length'}
    response = Response()
    class Base:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            def call(**args): calls.append(args); return response
            async def acall(**args): return call(**args)
            self._completion, self._completion_async = call, acall
    registered = {}
    monkeypatch.setattr(native_module, 'LiteLLMCompletion', Base)
    monkeypatch.setattr(factory, 'register_completion', lambda name, cls: registered.update({name:cls}))
    audit = ExtractionAudit(tmp_path/'audit.jsonl', profile=PROFILE)
    register_ms_observer(audit, 5)
    provider = registered[PROVIDER](cache='native-cache')
    result = await provider._completion_async(messages=[]) if asynchronous else provider._completion(messages=[])
    assert result is response and len(calls) == 1
    assert provider.kwargs['cache'] == 'native-cache'
    assert json.loads(audit.path.read_text())['response']['content'] == response.content
