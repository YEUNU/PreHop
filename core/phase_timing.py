"""Checkpointed benchmark wall segments; cumulative snapshots are never summed."""
from __future__ import annotations

import math
import os
import time
import uuid


class BenchmarkTiming:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.segment_id = uuid.uuid4().hex
        self.prior: list[dict] = []
        self.identity = {'segment_id': self.segment_id, 'pid': os.getpid(),
                         'started_epoch': time.time()}
        from pathlib import Path
        try:
            stat = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
            self.identity['process_start_ticks'] = stat[19]
            self.identity['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        except OSError:
            self.identity['process_start_ticks'] = None
            self.identity['boot_id'] = None

    def restore(self, value: dict | None) -> None:
        if value is None:
            return
        validate_timing(value)
        self.prior = [dict(row) for row in value['segments']]

    def snapshot(self) -> dict:
        segments = [*self.prior, {**self.identity, 'wall_seconds': time.perf_counter() - self.started}]
        return {'version': 1, 'scope': 'benchmark_execution_through_last_checkpoint',
                'segments': segments, 'total_wall_seconds': sum(row['wall_seconds'] for row in segments)}


def validate_timing(value: dict) -> None:
    if not isinstance(value, dict) or value.get('version') != 1 or value.get('scope') != 'benchmark_execution_through_last_checkpoint':
        raise RuntimeError('Invalid benchmark wall timing contract')
    rows = value.get('segments')
    if not isinstance(rows, list) or not rows:
        raise RuntimeError('Missing benchmark wall timing segments')
    identifiers = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('segment_id'), str) or not row['segment_id']:
            raise RuntimeError('Missing benchmark timing process segment identity')
        identifiers.append(row['segment_id'])
        seconds = row.get('wall_seconds')
        if isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds < 0:
            raise RuntimeError('Invalid benchmark wall seconds')
        if not isinstance(row.get('pid'), int) or not isinstance(row.get('started_epoch'), (float, int)):
            raise TypeError('Missing benchmark timing process identity')
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError('Duplicate benchmark timing segment would double count resume')
    total = value.get('total_wall_seconds')
    if isinstance(total, bool) or not isinstance(total, (int, float)) or not math.isfinite(total) or not math.isclose(total, sum(row['wall_seconds'] for row in rows), rel_tol=1e-12, abs_tol=1e-9):
        raise RuntimeError('Benchmark timing total differs from unique segments')
