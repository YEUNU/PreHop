"""Adapter ownership survives cancellation and result-publication failures."""
import asyncio
import json

import pytest

from cli import benchmark


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['cancel', 'checkpoint'])
@pytest.mark.parametrize('close_fails', [False, True])
async def test_failed_benchmark_drains_queries_before_closing(monkeypatch, tmp_path, failure, close_fails):
    both_started = asyncio.Event()
    calls = []
    active = set()
    closed = []

    class Engine:
        async def run_workflow(self, query, history):
            active.add(query)
            calls.append(query)
            if len(calls) == 2:
                both_started.set()
            try:
                await both_started.wait()
                if failure == 'checkpoint' and query == 'one':
                    return 'answer', [], []
                await asyncio.Future()
            finally:
                active.remove(query)

        def close(self):
            closed.append(set(active))
            if close_fails:
                raise RuntimeError('cleanup failed')

    async def evaluate(**kwargs):
        return {'answer_em': 1., 'llm_judge_score': -1., 'doc_match': 0.}

    def fail_write(*args):
        raise OSError('checkpoint disk failure')

    monkeypatch.delenv('RAG_EXECUTION_PROFILE', raising=False)
    monkeypatch.delenv('RAG_BENCHMARK_RESUME', raising=False)
    monkeypatch.delenv('RAG_INDEX_REUSE_LINK', raising=False)
    monkeypatch.setenv('RAG_BENCHMARK_CONCURRENCY', '2')
    monkeypatch.setenv('RAG_BENCHMARK_CHECKPOINT_EVERY', '1')
    monkeypatch.setattr(benchmark.RAGConfig, 'JUDGE_ENABLED', False)
    monkeypatch.setattr(benchmark, 'NaiveRAG', lambda **kwargs: Engine())
    monkeypatch.setattr(benchmark, '_latest_index_manifest_metadata', lambda *args: None)
    monkeypatch.setattr(benchmark, 'evaluate_multihoprag_response', evaluate)
    monkeypatch.setattr(benchmark, 'write_checkpoint', fail_write)
    queries = tmp_path / 'queries.json'
    queries.write_text(json.dumps([{'_id': q, 'query': q, 'dataset': 'multihoprag'} for q in ['one', 'two']]))
    task = asyncio.create_task(benchmark.run_benchmark(str(queries), 'naive', 'default', output_dir=tmp_path / 'out'))
    await asyncio.wait_for(both_started.wait(), timeout=3)
    if failure == 'cancel':
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(OSError, match='checkpoint disk failure'):
            await asyncio.wait_for(task, timeout=3)
    assert closed == [set()]
    assert not active


@pytest.mark.asyncio
async def test_adapter_closes_when_later_initialization_fails(monkeypatch):
    closed = []

    class Engine:
        def close(self):
            closed.append(True)

    def fail_client(*args):
        raise ValueError('invalid judge client')

    monkeypatch.setattr(benchmark, 'NaiveRAG', lambda **kwargs: Engine())
    monkeypatch.setattr(benchmark, 'get_llm_client', fail_client)
    with pytest.raises(RuntimeError, match='invalid judge client'):
        async with benchmark._benchmark_engine('naive', 'default', 'test', True):
            pytest.fail('Initialization should not complete')
    assert closed == [True]
