import json
from types import SimpleNamespace

import pytest

from models.hoprag.response_recovery import install


def test_valid_json_is_not_damaged_by_native_cleaner(tmp_path):
    tool = SimpleNamespace(txt2obj=lambda raw: None, get_chat_completion=lambda *a, **k: None)
    install(tool, tmp_path/'audit.jsonl')
    value = {'Question List': ['What does {x} mean?', 'Who said "hello"?']}
    assert tool.txt2obj(json.dumps(value)) == value
    assert tool.txt2obj('```json\n'+json.dumps(value)+'\n```') == value


def test_native_bad_arity_retries_without_dropping_document(tmp_path):
    calls = iter([(None, None, None), (['Who?'], [])])
    tool = SimpleNamespace(txt2obj=lambda raw: None, get_chat_completion=lambda *a, **k: next(calls))
    install(tool, tmp_path/'audit.jsonl', attempts=2)
    assert tool.get_chat_completion([], keys=['Question List']) == (['Who?'], [])
    rows = [json.loads(x) for x in (tmp_path/'audit.jsonl').read_text().splitlines()]
    assert [x['status'] for x in rows] == ['retry', 'accepted']


def test_exhausted_native_response_remains_failure(tmp_path):
    tool = SimpleNamespace(txt2obj=lambda raw: None, get_chat_completion=lambda *a, **k: (None,None,None))
    install(tool, tmp_path/'audit.jsonl', attempts=2)
    with pytest.raises(RuntimeError, match='exhausted'):
        tool.get_chat_completion([], keys=['Question List'])
    assert len((tmp_path/'audit.jsonl').read_text().splitlines()) == 2


@pytest.mark.parametrize(('key', 'value'), [
    ('Decision', 'Completely Irrelevant'),
    ('Decision', 'Indirectly Relevant'),
    ('Decision', 'Relevant and Necessary'),
    ('Decision', 'Not Needed'),
    ('answer', 'native answer'),
    ('Answer', 'native answer'),
    ('Title', 'native title'),
    ('Question List', ['Who?']),
    ('Subqueries', ['Where?']),
])
def test_native_caller_values_are_preserved_without_retry(tmp_path, key, value):
    chat = [{'role': 'assistant', 'content': json.dumps({key: value})}]
    result = (value, chat)
    calls = []
    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return result
    tool = SimpleNamespace(txt2obj=lambda raw: None, get_chat_completion=native)
    install(tool, tmp_path/'audit.jsonl')
    assert tool.get_chat_completion([], keys=[key]) is result
    assert len(calls) == 1
    assert json.loads((tmp_path/'audit.jsonl').read_text())['status'] == 'accepted'


def test_positional_keys_preserve_native_return_contract(tmp_path):
    result = ('Relevant and Necessary', 'answer', [])
    tool = SimpleNamespace(txt2obj=lambda raw: None, get_chat_completion=lambda *a, **k: result)
    install(tool, tmp_path/'audit.jsonl')
    assert tool.get_chat_completion([], True, 'native', 4096, ['Decision', 'answer']) is result


@pytest.mark.parametrize(('key', 'bad'), [('Decision', ['Relevant']), ('answer', None), ('Subqueries', 'question')])
def test_invalid_known_field_still_exhausts_recovery(tmp_path, key, bad):
    tool = SimpleNamespace(txt2obj=lambda raw: None, get_chat_completion=lambda *a, **k: (bad, []))
    install(tool, tmp_path/'audit.jsonl', attempts=2)
    with pytest.raises(RuntimeError, match='exhausted'):
        tool.get_chat_completion([], keys=[key])
    rows = [json.loads(line) for line in (tmp_path/'audit.jsonl').read_text().splitlines()]
    assert [r['status'] for r in rows] == ['retry', 'exhausted']
