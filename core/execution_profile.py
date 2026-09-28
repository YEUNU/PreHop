"""Content-bound operational settings; method semantics stay in the registry."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path


def execution_profile(environment: Mapping[str, str] | None = None) -> dict:
    environment = os.environ if environment is None else environment
    path = environment.get('RAG_EXECUTION_PROFILE', '').strip()
    if not path:
        return {'version': 1, 'name': 'serial-v1', 'sha256': None, 'settings': {}}
    value = json.loads(Path(path).read_text())
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {**value, 'sha256': digest}


def apply_execution_profile(environment: MutableMapping[str, str] | None = None) -> None:
    """An explicitly selected file takes precedence over ambient throughput defaults."""
    environment = os.environ if environment is None else environment
    mapping = {'generation_concurrency': 'RAG_GENERATION_CONCURRENCY',
               'embedding_batch_size': 'RAG_EMBEDDING_BATCH_SIZE',
               'embedding_concurrency': 'RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS',
               'benchmark_concurrency': 'RAG_BENCHMARK_CONCURRENCY',
               'index_document_concurrency': 'RAG_MAX_PARALLEL_FILES',
               'index_prefetch_documents': 'RAG_FILE_SCHEDULE_BATCH'}
    for key, value in execution_profile(environment)['settings'].items():
        if key in mapping:
            environment[mapping[key]] = str(value)


def resolved_execution_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Resolve registry defaults < caller environment < selected execution profile."""
    from core.strategy_registry import paper_environment_defaults

    resolved = {**paper_environment_defaults(), **(os.environ if environment is None else environment)}
    apply_execution_profile(resolved)
    return resolved
