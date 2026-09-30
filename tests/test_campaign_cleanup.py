import asyncio

import pytest

from scripts.campaign_runtime import close_owned, drain


@pytest.mark.asyncio
async def test_drain_cancels_owned_siblings_on_failure():
    cleaned = asyncio.Event()

    async def waiting():
        try:
            await asyncio.sleep(60)
        finally:
            cleaned.set()

    async def failing():
        await asyncio.sleep(0)
        raise OSError('mandatory save failed')

    with pytest.raises(OSError, match='mandatory save'):
        await drain([asyncio.create_task(waiting()), asyncio.create_task(failing())])
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_cleanup_attempts_all_resources_and_preserves_the_primary_error():

    closed = []

    async def failing_close():
        closed.append('first')
        raise RuntimeError('cleanup')

    async def second_close():
        closed.append('second')

    original = OSError('checkpoint')
    await close_owned(failing_close, second_close, primary_error=original)
    assert closed == ['first', 'second']
    with pytest.raises(RuntimeError, match='cleanup'):
        await close_owned(failing_close, second_close)
