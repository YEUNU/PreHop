"""Compare paper settings and adapter boundaries against the pinned native code."""
import ast
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from core.generation_profiles import request_settings
from core.strategy_registry import get_strategy
from models.external_research.drivers.youtu_agent import run_native_agent


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


def test_youtu_native_agent_loop_runs_and_preserves_final_evidence(tmp_path):
    tree = ast.parse(_native('youtu_graphrag', 'main.py').read_text())
    names = {'agent_retrieval', 'deduplicate_triples', 'merge_chunk_contents'}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ModuleType('pinned_fixture')
    log = SimpleNamespace(info=lambda *a: None, error=lambda *a: None)
    initial = {'triples': ['A->B'], 'chunk_ids': ['a'], 'chunk_contents': ['alpha'],
               'initial_answer': 'initial', 'total_time': 0, 'sub_question_results': []}
    namespace = module.__dict__
    namespace.update({'List': list, 'logger': log, 'time': SimpleNamespace(sleep=lambda _: None),
                 'config': SimpleNamespace(retrieval=SimpleNamespace(agent=SimpleNamespace(max_steps=5), top_k_filter=20)),
                 'initial_question_decomposition': lambda *a: initial,
                 'Eval': lambda: (_ for _ in ()).throw(AssertionError('native LLM judge must not run'))})
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<pinned-youtu-agent>', 'exec'), namespace)  # noqa: S102 - execute exact pinned function in an isolated fixture
    calls = []
    class Retriever:
        def generate_answer(self, prompt):
            calls.append(prompt)
            return ['The new query is: followup', 'So the answer is: beta', 'beta'][len(calls)-1]
        def process_retrieval_results(self, query, top_k):
            assert query == 'followup' and top_k == 20
            return {'triples': ['B->C'], 'chunk_ids': ['b'], 'chunk_contents': ['beta']}, 0
        def generate_prompt(self, question, context):
            assert 'alpha' in context and 'beta' in context
            return question + context
    result = run_native_agent(module, None, Retriever(), 'question', 'schema', tmp_path/'audit.jsonl')
    assert result['initial_answer'] == 'beta'
    assert dict(zip(result['chunk_ids'],result['chunk_contents'])) == {'a':'alpha','b':'beta'}
    assert len(calls) == 3
    audit = json.loads((tmp_path/'audit.jsonl').read_text())
    assert audit['answers'] == ['beta'] and audit['native_mode'] == 'agent'


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
