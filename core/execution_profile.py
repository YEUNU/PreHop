"""Content-bound operational settings; method semantics stay in the registry."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path

FIELDS = ('generation_concurrency', 'embedding_batch_size', 'embedding_concurrency', 'benchmark_concurrency')
# Persisted execution profiles retain the retired producer field for compatibility.
PRODUCER_FIELDS = ('index_document_concurrency', 'index_prefetch_documents',
                   'lightrag_document_concurrency', 'youtu_document_concurrency')


def execution_profile() -> dict:
    path = os.environ.get('RAG_EXECUTION_PROFILE', '').strip()
    if not path:
        return {'version': 1, 'name': 'serial-v1', 'sha256': None, 'settings': {}}
    value = json.loads(Path(path).read_text())
    if set(value) != {'version', 'name', 'settings'} or type(value['version']) is not int or value['version'] not in (1, 2, 3):
        raise ValueError('Invalid execution profile schema')
    if not isinstance(value['name'], str) or not value['name'].strip():
        raise ValueError('Execution profile needs a name')
    settings = value['settings']
    fields = FIELDS if value['version'] == 1 else FIELDS + PRODUCER_FIELDS
    if value['version'] == 3:
        fields = ('inference_concurrency', 'embedding_batch_size', 'benchmark_concurrency') + PRODUCER_FIELDS
    optional = {'prehop_chunk_concurrency'} if value['version'] >= 2 else set()
    if not isinstance(settings, dict) or not set(fields) <= set(settings) or set(settings) - set(fields) - optional:
        raise ValueError('Execution profile must declare all operational concurrency/batch settings')
    for key, number in settings.items():
        if type(number) is not int or not 1 <= number <= 1024:
            raise ValueError(f'Invalid execution profile setting: {key}')
    if value['version'] >= 2 and settings['index_prefetch_documents'] < settings['index_document_concurrency']:
        raise ValueError('Index prefetch must cover active document workers')
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


def require_queue(strategy='core') -> dict | None:
    """Execution-only readiness check; offline admission does not need a live queue."""
    profile = execution_profile()
    if not profile['sha256']:
        return None
    import httpx

    from core.inference_transport import InferenceTransport
    transport = InferenceTransport.resolve(strategy)
    proxy = os.environ.get('RAG_QUEUE_PROXY_URL')
    if not proxy:
        raise RuntimeError('A throughput profile requires the owned inference queue during execution')
    with httpx.Client(timeout=10, trust_env=False) as client:
        response = client.get(proxy.rstrip('/') + '/queue-metrics',
                              headers={'Authorization': f'Bearer {transport.api_key}'})
        response.raise_for_status()
        value = response.json()
    if value.get('profile') != profile or value.get('gateway_identity_sha256') != transport.gateway_identity_sha256:
        raise RuntimeError('Queue profile or upstream gateway identity differs')
    if value.get('limits') != queue_limits(profile):
        raise RuntimeError('Queue global limits differ from execution profile')
    return value


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


def exclusive_measurement(function):
    """Own an exclusive measurement, or one of the explicitly shared slots."""
    import functools

    @functools.wraps(function)
    async def wrapped(*args, **kwargs):
        if not execution_profile()['sha256']:
            return await function(*args, **kwargs)
        strategy = inspect.signature(function).bind_partial(*args, **kwargs).arguments.get('strategy', 'core')
        with measurement_slot(strategy):
            return await function(*args, **kwargs)
    return wrapped


@contextmanager
def measurement_slot(strategy='core'):
    """Keep exclusive runs isolated; shared runs still require the owned queue."""
    import fcntl

    capacity = os.environ.get('RAG_MEASUREMENT_MAX_TARGETS', '1')
    if capacity not in {'1', '2', '3', '4'}:
        raise ValueError('Measurement target capacity must be between 1 and 4')
    shared = int(capacity) > 1
    if shared and require_queue(strategy) is None:
        raise RuntimeError('Shared measurements require a validated owned queue')
    with ExitStack() as stack:
        handle = stack.enter_context(Path(f'/tmp/prehop-measured-target-{os.getuid()}.lock').open('a+'))
        try:
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another measured target is active; do not mix target workloads') from exc
        if shared:
            for slot in range(int(capacity)):
                candidate = Path(f'/tmp/prehop-measured-target-{os.getuid()}-slot-{slot}.lock').open('a+')  # noqa: SIM115 - transferred to ExitStack after lock acquisition
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close()
                else:
                    stack.enter_context(candidate)
                    break
            else:
                raise RuntimeError('All shared measurement slots are occupied')
        yield
