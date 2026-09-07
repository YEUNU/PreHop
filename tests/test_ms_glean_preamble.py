import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.external_research.extraction_contract import ExtractionFormatError
from models.ms_graphrag.extraction_guard import normalize_glean_preamble, validate_records, validate_response

ENTITY = '("entity"<|>MUSEUM<|>organization<|>A museum (navigation))'


def test_all_ten_observed_failed_responses_preserve_every_tuple():
    rows = json.loads((Path(__file__).parent / 'fixtures/ms_glean_preambles.json').read_text())
    assert len(rows) == 10
    for raw in rows:
        normalized = normalize_glean_preamble(raw)
        assert raw.endswith(normalized)
        assert normalized.count('<|>') == raw.count('<|>')
        entities, _ = validate_records(normalized)
        assert 'MERIDIAN MUSEUM OF NAVIGATION' in entities


@pytest.mark.parametrize('prefix', ['Explanation.\n\n', 'The objects do not match the entity types.\n'])
def test_descriptions_parentheses_and_all_fields_remain_verbatim(prefix):
    raw = ENTITY + '<|COMPLETE|>'
    assert normalize_glean_preamble(prefix + raw) == raw
    assert normalize_glean_preamble(raw) == raw


@pytest.mark.parametrize('prefix', ['("entity"<|>BROKEN\n', 'unknown<|>value\n', '<|COMPLETE|>\n', '##\n'])
def test_never_discards_structured_prefix(prefix):
    raw = prefix + ENTITY + '<|COMPLETE|>'
    assert normalize_glean_preamble(raw) == raw


@pytest.mark.parametrize('suffix', ['', '<|COMPLETE|> trailing prose'])
def test_incomplete_response_cannot_be_repaired(suffix):
    with pytest.raises(ExtractionFormatError):
        normalize_glean_preamble('Explanation.\n' + ENTITY + suffix)


def test_normalization_is_scoped_to_native_glean():
    prompts = pytest.importorskip('graphrag.prompts.index.extract_graph')
    raw = 'Explanation.\n' + ENTITY + '<|COMPLETE|>'
    response = SimpleNamespace(content=raw, choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace())])
    history = [{'role': 'user', 'content': prompts.GRAPH_EXTRACTION_PROMPT}]
    with pytest.raises(ExtractionFormatError):
        validate_response(response, history)
    observation = validate_response(response, history + [{'role': 'user', 'content': prompts.CONTINUE_PROMPT}])
    assert observation['glean_preamble_removed']
    assert observation['normalized_extraction'] == ENTITY + '<|COMPLETE|>'
    assert response.content == raw


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.asyncio
async def test_provider_preserves_raw_audit_and_native_parser_recovers_first_tuple(monkeypatch, tmp_path, asynchronous):
    factory = pytest.importorskip('graphrag_llm.completion.completion_factory')
    native_module = pytest.importorskip('graphrag_llm.completion.lite_llm_completion')
    from graphrag.index.operations.extract_graph.graph_extractor import GraphExtractor
    from graphrag.prompts.index.extract_graph import CONTINUE_PROMPT, GRAPH_EXTRACTION_PROMPT

    from models.external_research.extraction_contract import ExtractionAudit
    from models.ms_graphrag.extraction_guard import register_guard
    raw = 'Explanation.\n' + ENTITY + '<|COMPLETE|>'
    class Response:
        def __init__(self):
            self.choices = [SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=raw))]
        @property
        def content(self):
            return self.choices[0].message.content
        def model_dump(self):
            return {'content': self.content}
    class Base:
        def __init__(self, **kwargs):
            self._completion = lambda **kwargs: Response()
            async def call(**kwargs):
                return Response()
            self._completion_async = call
    captured = []
    monkeypatch.setattr(native_module, 'LiteLLMCompletion', Base)
    monkeypatch.setattr(factory, 'register_completion', lambda name, implementation: captured.append(implementation))
    audit = ExtractionAudit(tmp_path / 'audit.jsonl')
    register_guard(audit, 5)
    provider = captured[0]()
    args = {'messages': [{'role': 'user', 'content': GRAPH_EXTRACTION_PROMPT},
                         {'role': 'user', 'content': CONTINUE_PROMPT}]}
    response = await provider._completion_async(**args) if asynchronous else provider._completion(**args)
    row = json.loads(audit.path.read_text())
    assert row['response']['content'] == raw
    assert row['normalized_response'] == '##' + ENTITY + '##<|COMPLETE|>'
    assert row['native_observation']['glean_preamble_removed']
    native = object.__new__(GraphExtractor)
    entities, edges = native._process_result(response.content, 'fixture', '<|>', '##')
    assert entities.iloc[0]['title'] == 'MUSEUM'
    assert entities.iloc[0]['description'] == 'A museum (navigation)'
    assert edges.empty
    audit.assert_healthy()
