"""Prehop-only, lossless payload tracing with concurrent parent/child spans."""
from __future__ import annotations

import contextvars
import functools
import gzip
import hashlib
import inspect
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path

_ACTIVE = contextvars.ContextVar('prehop_trace_active', default=None)
_IDENTITY = contextvars.ContextVar('prehop_trace_identity', default=None)


def _json_default(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'data') and callable(value.data):
        return value.data()
    return str(value)


class TraceRecorder:
    def __init__(self, directory, *, metadata=None, secrets=()):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        (self.directory / 'payloads').mkdir(mode=0o700)
        self.lock = threading.RLock()
        self.sequence = 0
        self.secrets = tuple(s for s in secrets if s and s != 'EMPTY')
        self.emit('session_start', metadata or {})

    def emit(self, event, payload, *, span_id=None, parent_id=None, identity=None):
        # Files are closed after every append. A process crash retains previous
        # complete records; an unmatched start identifies interrupted work.
        value = json.loads(json.dumps(payload, ensure_ascii=True, default=_json_default))
        def sanitize(item):
            if isinstance(item, dict):
                return {key: '[REDACTED]' if key.lower() in {'api_key', 'authorization', 'password'}
                        else redact_body(value) if key == 'body' and isinstance(value, str)
                        else sanitize(value) for key, value in item.items()}
            if isinstance(item, list):
                return [sanitize(value) for value in item]
            if isinstance(item, str):
                for secret in self.secrets:
                    item = item.replace(secret, '[REDACTED]')
            return item
        def redact_body(body):
            # Preserve JSON syntax and numerical telemetry even if a configured
            # credential happens to consist of digits. Headers are never saved.
            def token(match):
                original = match.group(0)
                decoded = json.loads(original)
                cleaned = sanitize(decoded)
                return original if cleaned == decoded else json.dumps(cleaned, ensure_ascii=True)
            return re.sub(r'"(?:[^"\\]|\\.)*"', token, body)
        raw = json.dumps(sanitize(value), ensure_ascii=True, sort_keys=True).encode()
        digest = hashlib.sha256(raw).hexdigest()
        with self.lock:
            blob = self.directory / 'payloads' / (digest + '.json.gz')
            if not blob.exists():
                fd = os.open(blob, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, 'wb') as file:
                    file.write(gzip.compress(raw, compresslevel=1, mtime=0))
            self.sequence += 1
            row = {'version': 1, 'sequence': self.sequence, 'event': event,
                   'time_ns': time.time_ns(), 'pid': os.getpid(), 'span_id': span_id,
                   'parent_id': parent_id, 'identity': identity or {},
                   'payload': str(blob.relative_to(self.directory)), 'payload_sha256': digest}
            fd = os.open(self.directory / 'events.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, 'a', encoding='utf-8') as file:
                file.write(json.dumps(row, ensure_ascii=True) + '\n')
        return row

    @contextmanager
    def span(self, name, payload):
        previous = _ACTIVE.get()
        parent = previous[1] if previous and previous[0] is self else None
        span = uuid.uuid4().hex
        from core.structured_diagnostics import caller_metadata
        identity = {**caller_metadata(), **(_IDENTITY.get() or {})}
        token = _ACTIVE.set((self, span))
        started = time.perf_counter()
        try:
            self.emit(name + '.start', payload, span_id=span, parent_id=parent, identity=identity)
            yield span
        except BaseException as exc:
            self.emit(name + '.error', {'error_type': type(exc).__name__, 'message': str(exc),
                      'elapsed_seconds': time.perf_counter() - started},
                      span_id=span, parent_id=parent, identity=identity)
            raise
        finally:
            _ACTIVE.reset(token)

    def result(self, name, span, value):
        self.emit(name + '.result', value, span_id=span, identity=dict(_IDENTITY.get() or {}))

    @property
    def reference(self):
        return {'version': 1, 'events_path': str(self.directory / 'events.jsonl'),
                'payload_encoding': 'gzip-json', 'scope': 'prehop',
                'raw_http_bodies': True}


@contextmanager
def trace_identity(**identity):
    token = _IDENTITY.set({**(_IDENTITY.get() or {}), **identity})
    try:
        yield
    finally:
        _IDENTITY.reset(token)


def traced(function):
    signature = inspect.signature(function)

    @functools.wraps(function)
    async def wrapped(self, *args, **kwargs):
        recorder = getattr(self, 'trace_recorder', None)
        if recorder is None:
            return await function(self, *args, **kwargs)
        values = dict(signature.bind(self, *args, **kwargs).arguments)
        values.pop('self', None)
        name = function.__qualname__
        identity = {k: values[k] for k in ('source', 'document_filename') if k in values}
        parameters = values.get('parameters')
        if isinstance(parameters, dict) and isinstance(parameters.get('documents'), list):
            identity['sources'] = sorted({str(doc['doc_id']) for doc in parameters['documents']
                                          if isinstance(doc, dict) and doc.get('doc_id')})
        with (trace_identity(**identity),
              recorder.span(name, values) as span):
            result = await function(self, *args, **kwargs)
            recorder.result(name, span, result)
            return result
    return wrapped


async def _http_request(request):
    active = _ACTIVE.get()
    if not active:
        return
    recorder, parent = active
    span = uuid.uuid4().hex
    request.extensions['prehop_trace'] = (recorder, span, parent, dict(_IDENTITY.get() or {}))
    body = await request.aread()
    # Embedding permit waiters can occupy every default-executor worker.
    # Persist inline, like other trace events, so HTTP completion can release
    # its permit without waiting for that same executor.
    recorder.emit('http.request',
        {'method': request.method, 'path': request.url.path, 'body': body.decode('utf-8')},
        span_id=span, parent_id=parent, identity=dict(_IDENTITY.get() or {}))


async def _http_response(response):
    context = response.request.extensions.get('prehop_trace')
    if not context:
        return
    recorder, span, parent, identity = context
    body = await response.aread()
    recorder.emit('http.response',
        {'status_code': response.status_code, 'body': body.decode('utf-8', errors='surrogateescape')},
        span_id=span, parent_id=parent, identity=identity)


def attach_client(client, recorder):
    """Observe this Prehop client; SDK retry policy and shared clients stay intact."""
    original = client._retry_with_backoff

    async def retry(coro_func, *args, **kwargs):
        # OpenAI resource -> SDK client -> httpx client. Hooks are context gated,
        # so other strategies sharing an HTTP pool never emit Prehop traces.
        sdk = getattr(getattr(coro_func, '__self__', None), '_client', None)
        http = getattr(sdk, '_client', None)
        if http is not None and hasattr(http, 'event_hooks'):
            if _http_request not in http.event_hooks['request']:
                http.event_hooks['request'].append(_http_request)
            if _http_response not in http.event_hooks['response']:
                http.event_hooks['response'].append(_http_response)
        call_kwargs = {k: v for k, v in kwargs.items() if k != '_request_budget'}
        with recorder.span('inference', {'args': args, 'kwargs': call_kwargs}) as span:
            async def attempt(*a, **kw):
                with recorder.span('sdk_attempt', {'args': a, 'kwargs': kw}) as attempt_id:
                    response = await coro_func(*a, **kw)
                    recorder.result('sdk_attempt', attempt_id, response)
                    return response
            result = await original(attempt, *args, **kwargs)
            recorder.result('inference', span, result)
            return result
    client._retry_with_backoff = retry
    def observe(name, function):
        @functools.wraps(function)
        async def call(*args, **kwargs):
            from core.inference_telemetry import observe_structured_attempts
            from core.structured_diagnostics import caller_metadata
            def observation(metadata, *, valid):
                active = _ACTIVE.get()
                recorder.emit('structured.attempt', {'valid': valid, 'metadata': metadata},
                              span_id=active[1] if active else None,
                              identity={**caller_metadata(), **(_IDENTITY.get() or {})})
            with (recorder.span(name, {'args': args, 'kwargs': kwargs}) as span,
                  observe_structured_attempts(observation) if name == 'generate_json' else nullcontext()):
                value = await function(*args, **kwargs)
                recorder.result(name, span, value)
                return value
        return call
    for name in ('generate_response', 'generate_json', 'get_embeddings'):
        setattr(client, name, observe(name, getattr(client, name)))
    return client


class TracedNeo4j:
    def __init__(self, service, recorder):
        self.service = service
        self.trace_recorder = recorder

    def __getattr__(self, name):
        return getattr(self.service, name)

    @traced
    async def execute_query(self, query, parameters=None, **kwargs):
        return await self.service.execute_query(query, parameters, **kwargs)
