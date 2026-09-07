import pytest

from models.external_research.extraction_contract import normalize_entities as normalize_ner_entities


def test_observed_objects_preserve_names_order_and_deduplicate():
    assert normalize_ner_entities([
        'Amazon', {'entity': 'Amazon', 'type': 'Organization'}, {'entity': 'Amazon'}, {'text': 'EU'},
        {'text': 'EU', 'type': 'Organization'}, {'text': 'August 2022', 'label': 'Date'},
    ]) == ['Amazon', 'EU', 'August 2022']


@pytest.mark.parametrize('value', [
    {'entity': 'A', 'text': 'B'}, {'name': 'A'}, {'entity': {'text': 'A'}, 'type': 'Org'},
    {'entity': '', 'type': 'Org'}, ['A'], None, 1,
])
def test_ambiguous_or_invalid_entities_fail_closed(value):
    with pytest.raises(ValueError):
        normalize_ner_entities([value])


def test_native_empty_and_string_lists_preserved():
    assert normalize_ner_entities([]) == []
    assert normalize_ner_entities([' A ', 'A', ' A ']) == [' A ', 'A']


def test_native_recovery_preserves_raw_response_and_audits_rejection(tmp_path):
    import os
    import subprocess
    from pathlib import Path

    home = os.environ.get('RAG_TEST_PINNED_RUNTIME_HOME')
    if not home:
        pytest.skip('actual pinned Hippo runtime not selected')
    root = Path(__file__).resolve().parents[1]
    runtime = Path(home).resolve() / 'hipporag2'
    code = '''
import json,sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from hipporag.information_extraction.openie_openai import OpenIE
from models.external_research.drivers.hippo_extraction import install_extraction_adapter
class LLM:
    raw = '{"named_entities":[{"entity":"Ada","type":"Person"},"London"]}'
    calls = 0
    def infer(self, **kwargs):
        self.calls += 1
        return self.raw, {"finish_reason":"stop"}, False
llm = LLM()
openie = OpenIE(llm)
audit = Path(sys.argv[3])
install_extraction_adapter(openie, audit, 1)
result = openie.ner('doc', 'Ada visited London')
assert result.unique_entities == ['Ada', 'London']
assert result.response == llm.raw and llm.calls == 1
assert not result.metadata.get('error')
assert result.metadata['adapter_attempts'] == 1
llm.raw = '{"named_entities":["Ada"]}'
assert openie.ner('native', 'Ada').unique_entities == ['Ada']
assert len(audit.read_text().splitlines()) == 2
llm.raw = '{"named_entities":[{"entity":"A","text":"B"}]}'
assert openie.ner('bad', 'A B').metadata.get('error')
rows = [json.loads(line) for line in audit.read_text().splitlines()]
assert rows[-1]['status'] == 'exhausted' and rows[-1]['response'] == llm.raw
'''
    subprocess.run([str(runtime / 'venv/bin/python'), '-B', '-c', code, str(root),
                    str(runtime / 'source/src'), str(tmp_path / 'audit.jsonl')], check=True)


@pytest.mark.parametrize("key", ["entity", "text"])
@pytest.mark.parametrize("annotation", [{}, {"type": "Organization"}, {"label": "Organization"},
                                         {"type": "Organization", "label": "Organization"}])
def test_explicit_name_with_optional_type_and_label(key, annotation):
    assert normalize_ner_entities([{key: "Amazon", **annotation}]) == ["Amazon"]
