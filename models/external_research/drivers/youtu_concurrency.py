"""Synchronize native adaptive schema I/O without serializing LLM calls.

This preserves native parallel agent behavior, not serial prompt equivalence.
Graph mutations remain protected by the native builder's existing graph lock.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


class BoundedYoutuDocuments:
    """Schedule native document calls by I/O capacity, retaining native graph stages."""

    def process_all_documents(self, documents):
        workers = self.config.construction.max_workers
        if type(workers) is not int or workers < 1:
            raise ValueError('Youtu document workers must be positive')
        logger = logging.getLogger('Prehop.youtu_adapter')
        logger.info('Native document calls: %d documents, %d adapter workers', len(documents), workers)
        remaining = iter(documents)
        sentinel = object()
        pending = set()
        succeeded = failed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            def fill():
                while len(pending) < 2 * workers:
                    document = next(remaining, sentinel)
                    if document is sentinel:
                        break
                    pending.add(pool.submit(self.process_document, document))
            fill()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.remove(future)
                    try:
                        future.result()
                        succeeded += 1
                    except Exception:  # noqa: BLE001 - native observer retains per-source failure details
                        failed += 1
                fill()
                logger.info('Native documents finished: %d/%d (%d failed)', succeeded + failed, len(documents), failed)
        # These are the exact native post-document stages, in native order.
        # The existing adapter coverage validator rejects any failed source.
        self.triple_deduplicate()
        self.process_level4()


class SynchronizedYoutuSchema:
    def __init__(self, *args, **kwargs):
        self._schema_io_lock = threading.RLock()
        super().__init__(*args, **kwargs)

    def _get_construction_prompt(self, chunk):
        with self._schema_io_lock:
            return super()._get_construction_prompt(chunk)

    def _update_schema_with_new_types(self, new_schema_types):
        with self._schema_io_lock:
            result = super()._update_schema_with_new_types(new_schema_types)
            # Native code logs and swallows file errors. Do not accept a graph
            # that silently lost an accepted schema update.
            for native_key, schema_key in (('nodes', 'Nodes'), ('relations', 'Relations'),
                                           ('attributes', 'Attributes')):
                if any(value not in self.schema.get(schema_key, [])
                       for value in new_schema_types.get(native_key, [])):
                    raise RuntimeError('Youtu native schema update did not persist all suggested types')
            return result
