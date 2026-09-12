"""Content-bound operational settings; method semantics stay in the registry."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def execution_profile() -> dict:
    path = os.environ.get('RAG_EXECUTION_PROFILE', '').strip()
    if not path:
        return {'version': 1, 'name': 'serial-v1', 'sha256': None, 'settings': {}}
    value = json.loads(Path(path).read_text())
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {**value, 'sha256': digest}


def transport_settings(profile):
    """Resolve client ceilings; v3 admission is owned by one shared queue."""
    settings = dict(profile['settings'])
    if profile['version'] == 3:
        settings['generation_concurrency'] = settings['inference_concurrency']
        settings['embedding_concurrency'] = settings['inference_concurrency']
    return settings


def queue_limits(profile):
    settings = profile['settings']
    if profile['version'] == 3:
        return {'shared': settings['inference_concurrency']}
    return {'generation': settings['generation_concurrency'], 'embedding': settings['embedding_concurrency']}


def apply_execution_profile(environment=None):
    """An explicitly selected file takes precedence over ambient throughput defaults."""
    environment = os.environ if environment is None else environment
    mapping = {'generation_concurrency': 'RAG_GENERATION_CONCURRENCY',
               'embedding_batch_size': 'RAG_EMBEDDING_BATCH_SIZE',
               'embedding_concurrency': 'RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS',
               'benchmark_concurrency': 'RAG_BENCHMARK_CONCURRENCY',
               'index_document_concurrency': 'RAG_MAX_PARALLEL_FILES',
               'index_prefetch_documents': 'RAG_FILE_SCHEDULE_BATCH'}
    for key, value in transport_settings(execution_profile()).items():
        if key in mapping:
            environment[mapping[key]] = str(value)
