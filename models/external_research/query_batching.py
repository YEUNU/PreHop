"""Coalesce concurrent adapter requests into the unchanged native QA batch API."""
from __future__ import annotations

import asyncio
import os
import time

from core.benchmark_failures import BenchmarkIntegrityError


class NativeQueryBatcher:
    def __init__(self, worker):
        self.worker = worker
        self.limit = max(1, int(os.environ.get("RAG_BENCHMARK_CONCURRENCY", "8")))
        self.pending = []
        self.task = None

    async def request(self, question):
        future = asyncio.get_running_loop().create_future()
        self.pending.append((question, future, time.perf_counter()))
        if self.task is None:
            self.task = asyncio.create_task(self._drain())
        return await future

    async def _drain(self):
        try:
            while self.pending:
                # A bounded window also flushes the final partial batch.
                await asyncio.sleep(0.01)
                batch, self.pending = self.pending[:self.limit], self.pending[self.limit:]
                batch = [item for item in batch if not item[1].cancelled()]
                if not batch:
                    continue
                started = time.perf_counter()
                try:
                    response = await asyncio.to_thread(
                        self.worker.request,
                        {"operation": "query_batch", "queries": [item[0] for item in batch]},
                    )
                    results = response.get("results")
                    if not isinstance(results, list) or len(results) != len(batch):
                        raise BenchmarkIntegrityError("Native QA batch response cardinality mismatch")
                    if any(not isinstance(result, dict) for result in results):
                        raise BenchmarkIntegrityError("Native QA batch returned a malformed result")
                    for (_, future, queued), result in zip(batch, results):
                        if not future.done():
                            if result.get("error"):
                                future.set_exception(RuntimeError(result["error"]))
                                continue
                            future.set_result({**result,
                                "worker_queue_seconds": started - queued + float(response.get("worker_queue_seconds", 0.0)),
                                "native_query_batch_size": len(batch),
                                "native_qa_max_workers": response.get("native_qa_max_workers")})
                except Exception as exc:
                    # Preserve native batch failure; no repair or per-query retry.
                    for _, future, _ in batch:
                        if not future.done():
                            future.set_exception(exc)
        finally:
            self.task = None
