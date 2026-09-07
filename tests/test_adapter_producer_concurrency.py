import asyncio
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from core.execution_profile import apply_execution_profile, execution_profile
from models.external_research.drivers.youtu_concurrency import SynchronizedYoutuSchema


def profile():
    return {'version': 2, 'name': 'producer-test', 'settings': {
        'generation_concurrency': 60, 'embedding_batch_size': 16,
        'embedding_concurrency': 2, 'benchmark_concurrency': 8,
        'index_document_concurrency': 60, 'index_prefetch_documents': 120,
        'lightrag_document_concurrency': 32, 'youtu_document_concurrency': 32}}


def test_profile_binds_producers_and_semantic_youtu_policy(tmp_path, monkeypatch):
    p = tmp_path / 'profile.json'
    p.write_text(json.dumps(profile()))
    monkeypatch.setenv('RAG_EXECUTION_PROFILE', str(p))
    env = {}
    apply_execution_profile(env)
    assert env['RAG_MAX_PARALLEL_FILES'] == '60'
    assert env['RAG_FILE_SCHEDULE_BATCH'] == '120'
    from models.external_research.drivers.lightrag import LightRAGDriver
    assert LightRAGDriver._producer_options() == {'max_parallel_insert': 32}
    result = subprocess.run([sys.executable, '-c', '''
from core.strategy_registry import get_strategy, PAPER_TRANSPORT
p = dict(get_strategy('youtu_graphrag').paper_index_policy)
assert p['construction_concurrency'] == 32
assert p['schema_update_policy'] == 'locked-native-v1'
assert PAPER_TRANSPORT.generation_concurrency == 60
'''], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    old = execution_profile()['sha256']
    value = profile(); value['settings']['youtu_document_concurrency'] = 16
    p.write_text(json.dumps(value))
    assert execution_profile()['sha256'] != old
    value['settings']['index_prefetch_documents'] = 1
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='prefetch'):
        execution_profile()


def test_schema_updates_do_not_lose_types_and_llm_work_remains_parallel(tmp_path):
    path = tmp_path / 'schema.json'
    path.write_text(json.dumps({'Nodes': []}))
    class Native:
        def __init__(self):
            self.schema = {'Nodes': []}
        def _get_construction_prompt(self, chunk):
            return json.dumps(self.schema)
        def _update_schema_with_new_types(self, update):
            value = json.loads(path.read_text())
            time.sleep(.002)  # expose native read/modify/write race
            value['Nodes'].extend(update['nodes'])
            path.write_text(json.dumps(value))
            self.schema = value
    class Builder(SynchronizedYoutuSchema, Native):
        pass
    builder = Builder()
    barrier = threading.Barrier(8)
    def document(i):
        json.loads(builder._get_construction_prompt(str(i)))
        barrier.wait(timeout=5)  # stand-in LLM call, outside schema critical section
        builder._update_schema_with_new_types({'nodes': [str(i)]})
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(document, range(8)))
    assert set(json.loads(path.read_text())['Nodes']) == set(map(str, range(8)))
    assert builder.schema == json.loads(path.read_text())


def test_swallowed_native_schema_write_failure_is_rejected():
    class Native:
        def __init__(self): self.schema = {'Nodes': []}
        def _update_schema_with_new_types(self, _): pass
    class Builder(SynchronizedYoutuSchema, Native): pass
    with pytest.raises(RuntimeError, match='did not persist'):
        Builder()._update_schema_with_new_types({'nodes': ['new']})


def test_lightrag_adapter_joins_native_background_workers():
    from models.external_research.drivers.lightrag import LightRAGDriver
    closed = []
    class Engine:
        async def finalize_storages(self): closed.append('storage')
    driver = object.__new__(LightRAGDriver)
    driver.engine = Engine()
    driver.loop = asyncio.new_event_loop()
    async def worker():
        try: await asyncio.Event().wait()
        finally: closed.append('worker')
    driver.loop.create_task(worker())
    driver.loop.run_until_complete(asyncio.sleep(0))
    driver.close()
    driver.close()
    assert closed == ['storage', 'worker']
    assert driver.loop.is_closed()


def test_lightrag_close_in_pinned_python_runtime():
    runtime = Path(os.environ.get('RAG_TEST_PINNED_RUNTIME_HOME', 'data/official_baselines'))
    python = runtime / 'lightrag/venv/bin/python'
    if not python.exists():
        pytest.skip('pinned LightRAG runtime is not installed')
    result = subprocess.run([str(python), '-c', '''
import asyncio
from models.external_research.drivers.lightrag import LightRAGDriver
closed = []
class Engine:
    async def finalize_storages(self): closed.append('storage')
driver = object.__new__(LightRAGDriver)
driver.engine = Engine()
driver.loop = asyncio.new_event_loop()
async def worker():
    try: await asyncio.wait_for(asyncio.Queue().get(), timeout=100)
    finally: closed.append('worker')
driver.loop.create_task(worker())
driver.loop.run_until_complete(asyncio.sleep(0.01))
driver.close()
driver.close()
assert closed == ['storage', 'worker'], closed
assert driver.loop.is_closed()
'''], capture_output=True, text=True, timeout=20, check=False,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode == 0, result.stderr
    assert 'Task was destroyed' not in result.stderr


def test_youtu_adapter_uses_io_workers_and_joins_before_native_graph_stages(monkeypatch):
    from types import SimpleNamespace

    from models.external_research.drivers.youtu_concurrency import BoundedYoutuDocuments
    monkeypatch.setattr('os.cpu_count', lambda: 1)
    barrier = threading.Barrier(8)
    finished = []
    class Builder(BoundedYoutuDocuments):
        config = SimpleNamespace(construction=SimpleNamespace(max_workers=8))
        def process_document(self, document):
            barrier.wait(timeout=5)  # fails if a CPU+4 clamp is accidentally restored
            finished.append(document)
        def triple_deduplicate(self):
            assert set(finished) == set(range(16))
            finished.append('dedup')
        def process_level4(self):
            assert finished[-1] == 'dedup'
            finished.append('community')
    Builder().process_all_documents(list(range(16)))
    assert finished[-2:] == ['dedup', 'community']
