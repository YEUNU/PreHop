"""Bounded, loopback-only transparent inference queue. No retries or payload edits."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx


class QueueServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, upstream, upstream_key, token, limits, profile, *, timeout=600):
        self.upstream = upstream.rstrip('/')
        self.upstream_key = upstream_key
        self.token = token
        self.profile = profile
        self.limits = limits
        self.permits = {key: threading.BoundedSemaphore(value) for key, value in limits.items()}
        # Bound connection handlers as well as upstream requests.
        self.handler_capacity = 4 * sum(limits.values()) + 2
        # TCPServer defaults to a listen backlog of five on Python 3.12. A burst
        # can fail before process_request sees it, even with free upstream slots.
        self.request_queue_size = min(4096, self.handler_capacity)
        self.slots = threading.BoundedSemaphore(self.handler_capacity)
        self.guard = threading.Lock()
        self.rejected_connections = 0
        self.stats = {key: {'requests': 0, 'errors': 0, 'active': 0, 'waiting': 0, 'peak_active': 0,
                                'queue_seconds': 0.0, 'upstream_seconds': 0.0} for key in limits}
        self.client = httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False,
                                   limits=httpx.Limits(max_connections=sum(limits.values()) + 2))
        super().__init__(('127.0.0.1', 0), QueueHandler)

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            with self.guard:
                self.rejected_connections += 1
            try:
                request.sendall(b'HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()

    def drain(self, timeout=30):
        deadline = time.monotonic() + timeout
        while True:
            with self.guard:
                if not any(row['active'] or row['waiting'] for row in self.stats.values()):
                    return
            if time.monotonic() >= deadline:
                raise RuntimeError('Inference queue still has outstanding requests after target exit')
            time.sleep(.05)

    def snapshot(self):
        with self.guard:
            return {'version': 1, 'profile': self.profile, 'limits': self.limits,
                    'rejected_connections': self.rejected_connections,
                    'listen_backlog': self.request_queue_size, 'handler_capacity': self.handler_capacity,
                    'gateway_identity_sha256': hashlib.sha256(self.upstream.encode()).hexdigest(),
                    'metrics': {key: dict(value) for key, value in self.stats.items()}}


class QueueHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_args):
        pass  # Never log credentials or request bodies.

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        server = self.server
        if self.headers.get('Authorization') != f'Bearer {server.token}':
            self.send_error(401)
            return
        if self.command == 'GET' and self.path == '/v1/queue-metrics':
            body = json.dumps(server.snapshot()).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        routes = {'/v1/chat/completions': 'generation', '/v1/completions': 'generation',
                  '/v1/embeddings': 'embedding', '/v1/models': None}
        if self.path not in routes or (self.command == 'GET') != (self.path == '/v1/models'):
            self.send_error(404)
            return
        if self.headers.get('Transfer-Encoding'):
            self.send_error(400, 'Content-Length required')
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 <= length <= 64 * 1024 * 1024:
                raise ValueError()
        except ValueError:
            self.send_error(413)
            return
        self.connection.settimeout(600)
        body = self.rfile.read(length)
        kind = routes[self.path]
        queued = time.perf_counter()
        permit = server.permits.get(kind)
        if permit:
            with server.guard:
                server.stats[kind]['waiting'] += 1
            permit.acquire()
        started = time.perf_counter()
        if kind:
            with server.guard:
                stats = server.stats[kind]
                stats['waiting'] -= 1
                stats['requests'] += 1
                stats['active'] += 1
                stats['peak_active'] = max(stats['peak_active'], stats['active'])
                stats['queue_seconds'] += started - queued
        status = 502
        response_started = False
        try:
            headers = {'Authorization': f'Bearer {server.upstream_key}',
                       'Content-Type': self.headers.get('Content-Type', 'application/json'),
                       'Accept-Encoding': 'identity'}
            # Preserve bytes, status and streaming; never follow a vendor redirect.
            with server.client.stream(self.command, server.upstream + self.path[len('/v1'):],
                                      content=body, headers=headers) as response:
                status = response.status_code
                self.send_response(status)
                for name in ('content-type', 'content-encoding', 'retry-after', 'x-request-id'):
                    if name in response.headers:
                        self.send_header(name, response.headers[name])
                self.send_header('Connection', 'close')
                self.end_headers()
                response_started = True
                for chunk in response.iter_raw():
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (httpx.HTTPError, OSError):
            status = 502
            if not response_started:
                self.send_error(502, 'Upstream transport failed')
        finally:
            self.close_connection = True
            if kind:
                with server.guard:
                    stats['active'] -= 1
                    stats['upstream_seconds'] += time.perf_counter() - started
                    stats['errors'] += int(status >= 400)
            if permit:
                permit.release()


class OwnedQueue:
    """In-process campaign service; children inherit only its loopback route."""
    def __init__(self, metrics_path):
        self.metrics_path = metrics_path
        self.server = None
        self.prior = {}
        self.lock = None

    def start(self):
        import fcntl
        import os
        import secrets
        from pathlib import Path

        from core.execution_profile import execution_profile
        from core.inference_transport import InferenceTransport
        from core.strategy_registry import PAPER_TRANSPORT
        if os.environ.get('RAG_QUEUE_PROXY_URL'):
            raise RuntimeError('An owned campaign cannot inherit another queue service')
        profile = execution_profile()
        if not profile['sha256']:
            return
        self.lock = Path(f'/tmp/prehop-inference-queue-{os.getuid()}.lock').open('a+')  # noqa: SIM115 - lifetime spans start/close
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Campaign startup precedes per-method seed resolution.
            prior_seed = os.environ.get('RAG_LLM_SEED')
            os.environ['RAG_LLM_SEED'] = str(PAPER_TRANSPORT.benchmark_seed)
            try:
                transport = InferenceTransport.resolve('core')
            finally:
                if prior_seed is None:
                    os.environ.pop('RAG_LLM_SEED', None)
                else:
                    os.environ['RAG_LLM_SEED'] = prior_seed
            token = secrets.token_urlsafe(32)
            self.server = QueueServer(transport.generation_base_url, transport.api_key, token,
                                      {'generation': transport.generation_concurrency,
                                       'embedding': transport.embedding_concurrency}, profile,
                                      timeout=transport.timeout_seconds or None)
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            for key, value in {'RAG_QUEUE_PROXY_URL': f'http://127.0.0.1:{self.server.server_port}/v1',
                               'RAG_QUEUE_TOKEN': token}.items():
                self.prior[key] = os.environ.get(key)
                os.environ[key] = value
        except BaseException:
            self.lock.close()
            self.lock = None
            raise

    def drain(self):
        if self.server:
            self.server.drain()

    def persist(self):
        if self.server:
            from utils.io import _write_json
            _write_json(self.metrics_path, self.server.snapshot())

    def close(self):
        import os
        try:
            if self.server:
                self.server.shutdown()
                self.server.server_close()
                self.server.client.close()
                self.persist()
        finally:
            for key, value in self.prior.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if self.lock:
                self.lock.close()
