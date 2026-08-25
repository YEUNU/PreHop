import asyncio

import pytest

from cli.index import _reap_bounded_tasks, _submit_bounded_task


@pytest.mark.asyncio
async def test_bounded_scheduler_reuses_completed_slot_without_batch_barrier():
    slow_release = asyncio.Event()
    quick_finished = asyncio.Event()
    third_started = asyncio.Event()
    pending: dict[asyncio.Task[None], str] = {}

    async def slow_work():
        await slow_release.wait()

    async def quick_work():
        quick_finished.set()

    async def third_work():
        third_started.set()

    await _submit_bounded_task(pending, 2, "slow.txt", slow_work)
    await _submit_bounded_task(pending, 2, "quick.txt", quick_work)
    await quick_finished.wait()

    errors = await _submit_bounded_task(pending, 2, "third.txt", third_work)
    await third_started.wait()

    assert errors == []
    assert len(pending) == 2
    assert any(filename == "slow.txt" for filename in pending.values())

    slow_release.set()
    assert await _reap_bounded_tasks(pending, wait_for_one=False) == []
    assert pending == {}


@pytest.mark.asyncio
async def test_bounded_scheduler_reports_task_identity_on_failure():
    pending: dict[asyncio.Task[None], str] = {}

    async def fail():
        raise RuntimeError("broken")

    await _submit_bounded_task(pending, 1, "broken.txt", fail)
    errors = await _reap_bounded_tasks(pending, wait_for_one=False)

    assert len(errors) == 1
    assert errors[0][0] == "broken.txt"
    assert isinstance(errors[0][1], RuntimeError)
    assert pending == {}
