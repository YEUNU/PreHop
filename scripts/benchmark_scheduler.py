"""Bounded benchmark scheduling with immediate replacement of finished targets."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def run_rolling(rows, execute, before_start, heartbeat, *, capacity=2):
    if type(capacity) is not int or not 1 <= capacity <= 4:
        raise ValueError('Benchmark schedule capacity must be between 1 and 4')
    rows = list(rows)
    names = [row['strategy'] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate benchmark target')
    pending = iter(rows)
    outcomes = {}
    with ThreadPoolExecutor(max_workers=capacity) as pool:
        active = {}

        def fill():
            while len(active) < capacity:
                row = next(pending, None)
                if row is None:
                    break
                before_start(row)
                active[pool.submit(execute, row)] = row['strategy']

        fill()
        while active:
            done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                name = active.pop(future)
                outcomes[name] = future.result()
            fill()
            heartbeat()
    return outcomes
