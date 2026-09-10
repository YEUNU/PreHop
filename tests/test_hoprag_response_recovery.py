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
    assert json.loads((tmp_path/'audit.jsonl').read_text())['status'] == 'observed'


def test_positional_keys_preserve_native_return_contract(tmp_path):
    result = ('Relevant and Necessary', 'answer', [])
    tool = SimpleNamespace(txt2obj=lambda raw: None, get_chat_completion=lambda *a, **k: result)
    install(tool, tmp_path/'audit.jsonl')
    assert tool.get_chat_completion([], True, 'native', 4096, ['Decision', 'answer']) is result




@pytest.mark.parametrize('result', [(None, None, None), (['unexpected decision'], []), ('text', [])])
def test_native_result_is_observed_without_shape_rejection(tmp_path, result):
    calls = []
    def native(*args, **kwargs):
        calls.append(1)
        return result
    tool = SimpleNamespace(txt2obj=lambda raw: None, get_chat_completion=native)
    install(tool, tmp_path/'audit.jsonl')
    assert tool.get_chat_completion([], keys=['Decision']) is result
    assert calls == [1]
    assert json.loads((tmp_path/'audit.jsonl').read_text())['status'] == 'observed'
