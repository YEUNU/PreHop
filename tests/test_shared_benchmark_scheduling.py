import threading

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
            elif name == 'hop':
                assert not release_first.is_set()
                with guard:
                    assert 'light' in active
                release_first.set()
            return 0
        finally:
            with guard:
                active.remove(name)
    rows = [{'strategy': name} for name in ['light', 'ms', 'hop', 'gfm']]
    outcomes = run_rolling(rows, execute, lambda row: None, lambda: None, capacity=capacity)
    assert outcomes == {'light': 0, 'ms': 1, 'hop': 0, 'gfm': 0}
    assert 2 <= peak <= capacity
