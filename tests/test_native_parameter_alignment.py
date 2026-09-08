"""Compare paper settings and adapter boundaries against the pinned native code."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.generation_profiles import request_settings
from core.strategy_registry import get_strategy


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


def test_hippo_retrieval_candidates_are_not_qa_context_count():
    tree = ast.parse(_native('hipporag2', 'src/hipporag/utils/config_utils.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BaseConfig')
    defaults = {}
    for n in cls.body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.value, ast.Call):
            for keyword in n.value.keywords:
                if keyword.arg == 'default' and n.target.id in ('retrieval_top_k', 'qa_top_k'):
                    defaults[n.target.id] = ast.literal_eval(keyword.value)
    policy = dict(get_strategy('hipporag2').paper_index_policy)
    assert policy['retrieval_top_k'] == defaults['retrieval_top_k'] == 200
    assert policy['qa_top_k'] == defaults['qa_top_k'] == 5


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
