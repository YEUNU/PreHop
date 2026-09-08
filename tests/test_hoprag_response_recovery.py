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
