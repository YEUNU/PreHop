"""Checkpointed benchmark wall segments; cumulative snapshots are never summed."""
from __future__ import annotations

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
        self.prior = [dict(row) for row in value['segments']]

    def snapshot(self) -> dict:
        segments = [*self.prior, {**self.identity, 'wall_seconds': time.perf_counter() - self.started}]
        return {'version': 1, 'scope': 'benchmark_execution_through_last_checkpoint',
                'segments': segments, 'total_wall_seconds': sum(row['wall_seconds'] for row in segments)}
