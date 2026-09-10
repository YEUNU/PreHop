import asyncio
import os
import subprocess
from pathlib import Path

import pytest


def profile():
    return {'version': 2, 'name': 'producer-test', 'settings': {
        'generation_concurrency': 60, 'embedding_batch_size': 16,
        'embedding_concurrency': 2, 'benchmark_concurrency': 8,
        'index_document_concurrency': 60, 'index_prefetch_documents': 120,
        'lightrag_document_concurrency': 32, 'youtu_document_concurrency': 32}}


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
    task = driver.loop.create_task(worker())
    driver.loop.run_until_complete(asyncio.sleep(0))
    driver.close()
    driver.close()
    assert closed == ['storage', 'worker']
    assert task.done()
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
