"""Execution records outcomes without additional admission or integrity gates."""
import asyncio
import json
import logging
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_same_target_can_run_concurrently_without_queue_preflight(monkeypatch):
    from cli import index
    reached = 0
    both_running = asyncio.Event()

    async def execute(*args):
        nonlocal reached
        reached += 1
        if reached == 2:
            both_running.set()
        await asyncio.wait_for(both_running.wait(), timeout=2)
        return 'complete'

    monkeypatch.delenv('RAG_QUEUE_PROXY_URL', raising=False)
    monkeypatch.setattr(index, '_run_indexing_unlocked', execute)
    assert await asyncio.gather(
        index.run_indexing('same-corpus', 'naive', 'default', 'hotpotqa'),
        index.run_indexing('same-corpus', 'naive', 'default', 'hotpotqa'),
    ) == ['complete', 'complete']


@pytest.mark.asyncio
async def test_structured_json_decodes_without_response_schema_rejection(monkeypatch):
    from core.structured_outputs import question_contract
    from core.vllm_client import VLLMClient
    from models.prehop.llm_json import generate_json_or_raise
    client = VLLMClient.__new__(VLLMClient)
    client.logger = logging.getLogger('test_execution_without_validation')
    client._retry_attempts = 1
    response = {'q_minus': [123], 'q_plus': [], 'extra': 'native value'}
    client.generate_response = AsyncMock(return_value=json.dumps(response))
    result = await generate_json_or_raise(client, [], 'index', structured_contract=question_contract('index'))
    assert result == response
    client.generate_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_actual_generation_failure_is_not_reported_as_success():
    from core.structured_outputs import question_contract
    from core.vllm_client import VLLMClient
    client = VLLMClient.__new__(VLLMClient)
    client.logger = logging.getLogger('test_execution_without_validation')
    client._retry_attempts = 1
    error = ConnectionError('upstream disconnected')
    client.generate_response = AsyncMock(side_effect=error)
    with pytest.raises(ConnectionError) as caught:
        await client._generate_structured_json([], question_contract('index'))
    assert caught.value is error


def test_reuse_reference_loads_recorded_path_without_comparing_hash(tmp_path, monkeypatch):
    from core import index_reuse
    monkeypatch.setattr(index_reuse, 'ROOT', tmp_path)
    path = tmp_path / 'source.json'
    path.write_text('{"status":"partial","seed":7}')
    loaded_path, value = index_reuse.bound({'path': 'source.json', 'sha256': 'old'})
    assert loaded_path == path
    assert value == {'status': 'partial', 'seed': 7}
