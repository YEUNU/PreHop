import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.execution_profile import apply_execution_profile, execution_profile


def profile():
    return {'version': 2, 'name': 'producer-test', 'settings': {
        'generation_concurrency': 60, 'embedding_batch_size': 16,
        'embedding_concurrency': 2, 'benchmark_concurrency': 8,
        'index_document_concurrency': 60, 'index_prefetch_documents': 120,
        'lightrag_document_concurrency': 32, 'youtu_document_concurrency': 32}}


def test_profile_binds_producers_and_validates_prefetch(tmp_path, monkeypatch):
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
assert PAPER_TRANSPORT.generation_concurrency == 60
'''], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    old = execution_profile()['sha256']
    value = profile(); value['settings']['lightrag_document_concurrency'] = 16
    p.write_text(json.dumps(value))
    assert execution_profile()['sha256'] != old
    value['settings']['index_prefetch_documents'] = 1
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='prefetch'):
        execution_profile()


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
