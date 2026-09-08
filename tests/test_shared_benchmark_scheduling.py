import asyncio
import threading
from pathlib import Path

import pytest

from scripts.benchmark_scheduler import run_rolling


@pytest.mark.parametrize("capacity", [2, 3, 4])
def test_finished_or_failed_target_is_replaced_without_waiting_for_sibling(capacity):
    release_first = threading.Event()
    first_started = threading.Event()
    guard = threading.Lock()
    active = set()
    peak = 0
    def execute(row):
        nonlocal peak
        name = row['strategy']
        with guard:
            active.add(name)
            peak = max(peak, len(active))
        try:
            if name == 'light':
                first_started.set()
                assert release_first.wait(5)
            elif name == 'ms':
                assert first_started.wait(5)
                return 1
            elif name == 'youtu':
                assert not release_first.is_set()
                with guard:
                    assert 'light' in active
                release_first.set()
            return 0
        finally:
            with guard:
                active.remove(name)
    rows = [{'strategy': name} for name in ['light', 'ms', 'youtu', 'gfm']]
    outcomes = run_rolling(rows, execute, lambda row: None, lambda: None, capacity=capacity)
    assert outcomes == {'light': 0, 'ms': 1, 'youtu': 0, 'gfm': 0}
    assert 2 <= peak <= capacity


@pytest.mark.asyncio
@pytest.mark.parametrize("capacity", [2, 3, 4])
async def test_real_measurement_decorator_allows_capacity_and_releases_on_failure(tmp_path, monkeypatch, capacity):
    import core.execution_profile as profile
    monkeypatch.setattr(profile, 'execution_profile', lambda: {'sha256': 'selected'})
    monkeypatch.setattr(profile, 'require_queue', lambda strategy: {'validated': True})
    monkeypatch.setattr(profile, 'Path', lambda value: tmp_path / Path(value).name)
    monkeypatch.setenv('RAG_MEASUREMENT_MAX_TARGETS', str(capacity))
    started = [asyncio.Event() for _ in range(capacity)]
    release = asyncio.Event()
    @profile.exclusive_measurement
    async def work(number):
        started[number].set()
        await release.wait()
        if number == 0:
            raise ValueError('native failure')
        return 'done'
    jobs = [asyncio.create_task(work(i)) for i in range(capacity)]
    try:
        await asyncio.wait_for(asyncio.gather(*(e.wait() for e in started)), 3)
        with pytest.raises(RuntimeError, match='All shared measurement slots'):
            await work(0)
        monkeypatch.setenv('RAG_MEASUREMENT_MAX_TARGETS', '1')
        with pytest.raises(RuntimeError, match='Another measured target'):
            await work(0)
    finally:
        release.set()
        results = await asyncio.gather(*jobs, return_exceptions=True)
    assert isinstance(results[0], ValueError)
    assert results[1] == 'done'
    assert await work(1) == 'done'


def test_shared_mode_requires_queue(monkeypatch):
    from core.execution_profile import measurement_slot
    monkeypatch.setenv('RAG_MEASUREMENT_MAX_TARGETS', '2')
    monkeypatch.setattr('core.execution_profile.require_queue', lambda strategy: None)
    with pytest.raises(RuntimeError, match='validated owned queue'), measurement_slot():
        pytest.fail('must reject before execution')
