"""Owned stage orchestration: prerequisites, process barriers and immutable evidence."""
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import paper_stage_runner as runner
from scripts import recovery_checkpoint as recovery


@pytest.fixture(autouse=True)
def preserve_environment():
    original = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original)


def test_stage_runner_guarded_help_has_no_native_imports():
    completed = subprocess.run([sys.executable, 'scripts/paper_stage_runner.py', '--help'], capture_output=True, text=True, check=False)
    assert completed.returncode == 0
    assert 'one-query' in completed.stdout and 'full-target' in completed.stdout and 'reattest' in completed.stdout


def test_recovery_hook_default_is_inert(monkeypatch):
    monkeypatch.delenv('RAG_RECOVERY_TEST_CONTROL', raising=False)
    asyncio.run(recovery.checkpoint_barrier(Path('/does/not/exist'), {}))




def test_actual_owned_child_atomic_checkpoint_then_sigterm(tmp_path):
    """Exercise only the test barrier with a new subprocess; no service/native action."""
    run_id = 'owned-test'
    base = tmp_path/'data/results'/run_id/'recovery'
    base.mkdir(parents=True)
    control = {'run_id': run_id, 'owner_pid': os.getpid(), 'owner_start': recovery.process_start(os.getpid()), 'nonce': 'test-nonce'}
    (base/'control.json').write_text(json.dumps(control))
    script = f'''
import asyncio,json,os
from pathlib import Path
from scripts import recovery_checkpoint as r
r.ROOT=Path({str(tmp_path)!r})
base=Path({str(base)!r})
result=base/'benchmark/result.json'
result.parent.mkdir()
summary={{'status':'in_progress','total_queries':2,'details':[{{'query_id':'one'}}]}}
result.write_text(json.dumps(summary))
result.with_name('result.traces.jsonl').write_text('{{"query_id":"one"}}\\n')
asyncio.run(r.checkpoint_barrier(result,summary))
'''
    env = os.environ.copy()
    env.update(RAG_RUN_ID=run_id, RAG_RECOVERY_TEST_PROFILE='owned-checkpoint-v1', RAG_RECOVERY_TEST_CONTROL=str(base/'control.json'))
    child = subprocess.Popen([sys.executable, '-c', script], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    start = recovery.process_start(child.pid)
    deadline = time.monotonic() + 10
    while not (base/'checkpoint_ready.json').exists() and child.poll() is None and time.monotonic() < deadline:
        time.sleep(.02)
    if not (base/'checkpoint_ready.json').exists():
        # Natural bounded hook timeout; do not terminate any unverified process.
        stdout, stderr = child.communicate(timeout=130)
        pytest.fail(f'owned hook did not notify: {stdout!r} {stderr!r}')
    notice = json.loads((base/'checkpoint_ready.json').read_text())
    assert notice['pid'] == child.pid and notice['process_start'] == start
    assert recovery.process_start(child.pid) == start and child.poll() is None
    child.send_signal(signal.SIGTERM)
    assert child.wait(timeout=5) == -signal.SIGTERM
    assert notice['completed'] == 1 and notice['total'] == 2
    assert (base/'checkpoint_ready.pending').read_bytes() == (base/'checkpoint_ready.json').read_bytes()






def test_shell_handoff_uses_selected_main_prefix_and_rejects_drift(monkeypatch):
    monkeypatch.delenv('PYTHON_BIN', raising=False)
    monkeypatch.delenv('UV_PROJECT_ENVIRONMENT', raising=False)
    env = runner.selected_python_environment()
    assert env['PYTHON_BIN'] == str(Path(sys.prefix).resolve()/'bin/python')
    assert env['UV_PROJECT_ENVIRONMENT'] == str(Path(sys.prefix).resolve())
    monkeypatch.setenv('PYTHON_BIN', '/some/other/venv/bin/python')
    assert runner.selected_python_environment()['PYTHON_BIN'] == str(Path(sys.prefix).resolve()/'bin/python')




def test_production_benchmark_checkpoint_resume_retains_rows_and_schema_metadata(tmp_path, monkeypatch):
    from cli import benchmark
    from core.config import RAGConfig
    from core.paper_policy import structured_query_identity

    monkeypatch.setenv('RAG_PAPER_MODE', 'false')
    monkeypatch.setenv('RAG_BENCHMARK_CONCURRENCY', '1')
    monkeypatch.setenv('RAG_BENCHMARK_CHECKPOINT_EVERY', '1')
    monkeypatch.delenv('RAG_BENCHMARK_RESUME', raising=False)
    monkeypatch.setattr(RAGConfig, 'JUDGE_ENABLED', False)
    monkeypatch.setattr(RAGConfig, 'LLM_SEED', RAGConfig.LLM_SEED)
    calls = []
    class NativeFixture:
        def __init__(self, **kwargs):
            pass
        async def run_workflow(self, query, history):
            calls.append(query)
            return '@@ANSWER: Ada', [{'doc': 'Ada', 'source': 'Ada.txt', 'text': 'Ada visited.'}], [{'step': 'answer', 'output': 'Ada'}]
    monkeypatch.setattr(benchmark, 'NaiveRAG', NativeFixture)
    monkeypatch.setattr(benchmark, '_latest_index_manifest_metadata', lambda *args: None)
    query_path = tmp_path/'multihoprag_queries.json'
    query_path.write_text(json.dumps([{'_id': str(i), 'dataset': 'multihoprag', 'query': f'Who visited {i}?',
                                      'ground_truth': 'Ada', 'evidence_facts': ['Ada visited.'], 'evidence_docs': ['Ada']} for i in range(2)]))
    ready = asyncio.Event()
    async def interrupt(path, summary):
        if len(summary['details']) == 1:
            ready.set()
            await asyncio.Future()
    monkeypatch.setattr(recovery, 'checkpoint_barrier', interrupt)
    async def first_attempt():
        task = asyncio.create_task(benchmark.run_benchmark(str(query_path), 'naive', 'default',
            corpus_tag='multihoprag', output_dir=tmp_path/'result', limit=2, seed=42))
        await asyncio.wait_for(ready.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(first_attempt())
    result = tmp_path/'result/naive/multihoprag/seed_42/naive_multihoprag.json'
    prior = json.loads(result.read_text())
    trace_path = result.with_name('naive_multihoprag.traces.jsonl')
    prior_trace = json.loads(trace_path.read_text())
    assert prior['status'] == 'in_progress' and len(calls) == 1
    assert all(prior['ablation'][key] == value for key, value in structured_query_identity('naive').items())
    async def no_barrier(*args):
        return None
    monkeypatch.setattr(recovery, 'checkpoint_barrier', no_barrier)
    monkeypatch.setenv('RAG_BENCHMARK_RESUME', 'true')
    asyncio.run(benchmark.run_benchmark(str(query_path), 'naive', 'default', corpus_tag='multihoprag',
                                       output_dir=tmp_path/'result', limit=2, seed=42))
    final = json.loads(result.read_text())
    assert final['status'] == 'completed_unadmitted' and len(calls) == 2
    assert final['details'][0] == prior['details'][0]
    assert json.loads(trace_path.read_text().splitlines()[0]) == prior_trace
    assert final['resume']['retained_rows'] == 1 and final['resume']['resumed_rows'] == 1
    assert all(final['ablation'][key] == value for key, value in structured_query_identity('naive').items())


def test_all_gate_verifier_sources_are_existing_path_objects():
    from core.admission import verifier_sources
    from scripts.paper_gate_ledger import _provenance

    assert all(isinstance(path, Path) and path.is_file() for path in verifier_sources())
    context = _provenance()
    assert context['verifier_sha256'] and context['code']['source_tree_sha256']
