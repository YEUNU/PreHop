import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError

from models.external_research.drivers import youtu_format_retry as retry


def response(content):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=content))])


VALID = json.dumps({'attributes': {}, 'triples': [['A', 'knows', 'B']], 'entity_types': {},
                    'new_schema_types': {'nodes': [], 'relations': [], 'attributes': []}})


def client(outcomes, seen):
    def create(**kwargs):
        seen.append(kwargs)
        value = outcomes[len(seen) - 1]
        if isinstance(value, Exception): raise value
        return value
    native = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    def options(**kwargs):
        assert kwargs == {'max_retries': 0}
        return native
    native.with_options = options
    return native


def test_format_and_transport_share_one_budget_and_keep_request_unchanged(monkeypatch):
    monkeypatch.setattr(retry.time, 'sleep', lambda _: None)
    seen = []
    rate_limit = APIStatusError('busy', response=httpx.Response(429, request=httpx.Request('POST', 'http://test')), body=None)
    good = response(VALID)
    facade = retry.RetryingYoutuConstructionClient(client([response('{"a":1,"a":2}'), rate_limit, good], seen), 3)
    assert facade.chat.completions.create(model='model', messages=[{'role': 'user', 'content': 'unchanged'}], temperature=.3) is good
    assert seen[0] == seen[1] == seen[2]
    assert facade.retry_stats() == {'attempts': 3, 'format_errors': 1, 'transport_errors': 1, 'exhausted': 0}


def test_invalid_responses_are_never_repaired_or_retried_past_budget(monkeypatch):
    monkeypatch.setattr(retry.time, 'sleep', lambda _: None)
    seen = []
    facade = retry.RetryingYoutuConstructionClient(client([response('bad')] * 3, seen), 3)
    with pytest.raises(json.JSONDecodeError): facade.chat.completions.create(model='model', messages=[])
    assert len(seen) == 3 and facade.retry_stats()['exhausted'] == 1


def test_refusal_and_caller_format_override_do_not_retry(monkeypatch):
    monkeypatch.setattr(retry.time, 'sleep', lambda _: None)
    seen = []; refused = response(VALID); refused.choices[0].message.refusal = 'refused'
    facade = retry.RetryingYoutuConstructionClient(client([refused], seen), 3)
    with pytest.raises(ValueError, match='cannot be overridden'):
        facade.chat.completions.create(response_format={})
    assert not seen
    with pytest.raises(ValueError, match='refusal'):
        facade.chat.completions.create(model='model', messages=[])
    assert len(seen) == 1
