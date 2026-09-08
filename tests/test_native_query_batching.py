import asyncio
from types import SimpleNamespace

import pytest

from core.benchmark_failures import BenchmarkIntegrityError
from models.external_research.query_batching import NativeQueryBatcher
from models.external_research.drivers.linear_rag import LinearRAGDriver


def test_coalesces_duplicate_questions_and_flushes_partial_batch(monkeypatch):
    monkeypatch.setenv('RAG_BENCHMARK_CONCURRENCY', '8')
    calls = []
    def request(payload):
        calls.append(payload)
        return {'results': [{'answer': str(i), 'documents': []} for i, _ in enumerate(payload['queries'])]}
    async def run():
        batcher = NativeQueryBatcher(SimpleNamespace(request=request))
        results = await asyncio.gather(*(batcher.request('same') for _ in range(11)))
        assert [r['answer'] for r in results] == list(map(str, range(8))) + ['0', '1', '2']
        assert [r['native_query_batch_size'] for r in results] == [8]*8 + [3]*3
        assert all(r['worker_queue_seconds'] >= 0 for r in results)
    asyncio.run(run())
    assert [len(c['queries']) for c in calls] == [8, 3]


@pytest.mark.parametrize('mode', ['exception', 'cardinality'])
def test_batch_failures_propagate_without_retry(mode):
    calls = []
    def request(payload):
        calls.append(payload)
        if mode == 'exception':
            raise RuntimeError('native failed')
        return {'results': []}
    async def run():
        batcher = NativeQueryBatcher(SimpleNamespace(request=request))
        results = await asyncio.gather(batcher.request('a'), batcher.request('b'), return_exceptions=True)
        kind = RuntimeError if mode == 'exception' else BenchmarkIntegrityError
        assert all(isinstance(r, kind) for r in results)
    asyncio.run(run())
    assert len(calls) == 1


def test_driver_uses_native_batch_api_and_preserves_order():
    driver = LinearRAGDriver.__new__(LinearRAGDriver)
    calls = []
    def qa(questions):
        calls.append(questions)
        return [{'pred_answer': q['question'], 'sorted_passage': ['0:doc'], 'sorted_passage_scores': [0.7]} for q in questions]
    driver.engine = SimpleNamespace(graph=SimpleNamespace(vcount=lambda: 1), qa=qa)
    driver.passages = ['0:doc']
    driver.by_index = {0: {'source_id': 'd', 'title': 'title', 'text': 'doc'}}
    results = driver.query_batch(['a', 'b'])
    assert [r['answer'] for r in results] == ['a', 'b']
    assert calls == [[{'question': 'a', 'answer': ''}, {'question': 'b', 'answer': ''}]]


def test_pinned_native_qa_runs_eight_inferences_concurrently():
    """Execute the pinned method itself with deterministic transport/retrieval doubles."""
    import ast
    import threading
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    path = Path('data/official_baselines/linear_rag/source/src/LinearRAG.py')
    if not path.exists():
        pytest.skip('pinned native checkout not installed')
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LinearRAG')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'qa')
    scope = {'ThreadPoolExecutor': ThreadPoolExecutor, 'tqdm': lambda iterable, **kwargs: iterable}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    barrier = threading.Barrier(8)
    def infer(messages):
        barrier.wait(timeout=5)
        return 'Thought: native. Answer: ' + messages[1]['content'].split('Question: ')[1].split('\n')[0]
    engine = SimpleNamespace(config=SimpleNamespace(max_workers=16), llm_model=SimpleNamespace(infer=infer),
        retrieve=lambda questions: [{'question': q['question'], 'sorted_passage': ['doc']} for q in questions])
    results = scope['qa'](engine, [{'question': str(i)} for i in range(8)])
    assert [row['pred_answer'] for row in results] == list(map(str, range(8)))


def test_driver_keeps_pinned_native_worker_default():
    import ast
    from pathlib import Path
    tree = ast.parse(Path("models/external_research/drivers/linear_rag.py").read_text())
    config_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "LinearRAGConfig"]
    assert len(config_calls) == 1
    assert "max_workers" not in {kw.arg for kw in config_calls[0].keywords}


def test_lightrag_native_async_queries_overlap_and_keep_individual_failure():
    from models.external_research.drivers.lightrag import LightRAGDriver
    driver = LightRAGDriver.__new__(LightRAGDriver)
    driver.loop = asyncio.new_event_loop()
    driver.param = object()
    driver.by_id = {}
    active = 0
    peak = 0
    async def aquery(question, param):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        if question == 'bad':
            raise RuntimeError('native failure')
        return {'status': 'success', 'data': {'chunks': []}, 'llm_response': {'content': question}}
    driver.engine = SimpleNamespace(aquery_llm=aquery)
    try:
        rows = driver.query_batch(['a', 'bad', 'c'])
        assert peak == 3
        assert rows[0]['answer'] == 'a' and rows[2]['answer'] == 'c'
        assert isinstance(rows[1], RuntimeError)
    finally:
        driver.loop.close()


def test_gfm_native_answer_pool_preserves_retrieval_seriality_and_order():
    import threading
    from models.external_research.drivers.gfm_rag import GFMRAGDriver
    driver = GFMRAGDriver.__new__(GFMRAGDriver)
    driver.native_qa_max_workers = 5
    prepared = []
    driver._prepare_query = lambda q: (prepared.append(q) or q)
    barrier = threading.Barrier(5)
    def answer(q):
        assert prepared == list(range(5))
        barrier.wait(timeout=5)
        if q == 2:
            raise RuntimeError('native failure')
        return {'answer': str(q)}
    driver._answer_prepared = answer
    rows = driver.query_batch(list(range(5)))
    assert [r['answer'] for r in rows if isinstance(r, dict)] == ['0','1','3','4']
    assert isinstance(rows[2], RuntimeError)


def test_batcher_keeps_successful_siblings_when_one_native_query_fails():
    async def run():
        worker = SimpleNamespace(request=lambda _: {'results': [{'answer': 'ok'}, {'error': 'native failure'}]})
        batcher = NativeQueryBatcher(worker)
        results = await asyncio.gather(batcher.request('a'), batcher.request('b'), return_exceptions=True)
        assert results[0]['answer'] == 'ok'
        assert isinstance(results[1], RuntimeError)
    asyncio.run(run())


def test_handoff_waits_for_cleanup_and_skips_terminal_attempts():
    import runpy
    from pathlib import Path
    path=Path('data/results/native-query-concurrency-20260908/wait_and_continue.py')
    if not path.exists():
        pytest.skip('campaign-specific handoff not present')
    code=runpy.run_path(str(path), run_name='handoff_test')
    state={'state':'failed','owned_cleanup_complete':True,'remaining_owned_processes':[]}
    assert not code['ready'](state,True)
    assert code['ready'](state,False)
    assert not code['ready']({**state,'owned_cleanup_complete':False},False)
    assert not code['ready']({**state,'remaining_owned_processes':[123]},False)
    rows=[{'strategy':s} for s in ['linear_rag','lightrag','gfm_rag']]
    assert code['remaining'](rows, {'linear_rag':{'state':'admitted'},'lightrag':{'state':'failed'},'gfm_rag':{'state':'planned'}})==[{'strategy':'gfm_rag'}]
