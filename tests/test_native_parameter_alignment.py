"""Compare paper settings and adapter boundaries against the pinned native code."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.generation_profiles import request_settings


def _native(strategy, relative):
    home = os.environ.get('RAG_TEST_PINNED_RUNTIME_HOME')
    if not home:
        pytest.skip('actual pinned native sources not selected')
    return Path(home) / strategy / 'source' / relative


def test_ms_native_models_do_not_set_an_output_cap():
    from graphrag_llm.config.model_config import ModelConfig
    assert ModelConfig.model_fields['call_args'].default_factory() == {}
    for consumer in ('ms_completion', 'ms_extract', 'ms_report'):
        assert 'max_tokens' not in request_settings(consumer)
        assert 'max_completion_tokens' not in request_settings(consumer)
        assert 'temperature' not in request_settings(consumer)


def test_gfm_driver_uses_native_prompt_and_answer_without_shared_synthesis():
    from unittest.mock import Mock

    from models.external_research.drivers.gfm_rag import GFMRAGDriver

    driver = GFMRAGDriver.__new__(GFMRAGDriver)
    candidates = [{'id':'a','type':'document','attributes':{'content':'alpha'},'score':0.7}]
    driver.retriever = SimpleNamespace(retrieve=lambda q, top_k: {'document': candidates})
    driver.extraction_audit = Mock(spec=['assert_healthy'])
    driver.qa_audit = Mock(spec=['assert_healthy'])
    driver.top_k = 5
    driver.rows = {'a':{'source_id':'a','title':'A','text':'alpha'}}
    calls = []
    def build(question, sources):
        calls.append((question, sources))
        return [{'role':'user','content':'native prompt'}]
    driver.qa_prompt = SimpleNamespace(build_input_prompt=build)
    def generate(messages):
        assert messages == [{'role':'user','content':'native prompt'}]
        return 'native answer'
    driver.qa_llm = SimpleNamespace(generate_sentence=generate)
    result = driver.query('question')
    assert calls == [('question', {'document':[{'name':'a','content':'alpha'}]})]
    assert result['answer'] == 'native answer' and result['documents'][0]['source_id'] == 'a'
