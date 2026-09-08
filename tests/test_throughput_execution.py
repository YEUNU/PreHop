"""Offline integration coverage for global queue bounds and paper cost semantics."""
import concurrent.futures
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from core.amortized_cost import indexing_cost, query_cost, validate_cost
from core.execution_profile import execution_profile
from core.inference_queue import QueueServer


def test_normalized_cost_is_inverse_throughput_not_mean_latency():
    cost = query_cost(20.0, 100, complete=True)
    assert cost['seconds_per_unit'] == .2
    assert cost['units_per_second'] == 5
    assert cost['continuous_run_eligible']
    for kwargs in ({'complete': False}, {'complete': True, 'resumed': True}):
        assert query_cost(20, 100, **kwargs)['seconds_per_unit'] is None
    assert query_cost(float('nan'), 100, complete=True)['continuous_run_eligible'] is False
    assert query_cost(10, 0, complete=True)['seconds_per_unit'] is None


def test_index_denominator_is_manifest_sources():
    stats = {'timing_seconds': {'total_elapsed_seconds': 120},
             'corpus_manifest_paragraph_count': 60, 'status': 'complete', 'chunk_count': 900}
    cost = indexing_cost(stats)
    assert cost['seconds_per_unit'] == 2
    with pytest.raises(ValueError):
        validate_cost({**cost, 'seconds_per_unit': .1}, cost)


def test_profile_is_content_bound_and_registry_cli_uses_it(tmp_path, monkeypatch):
    path = tmp_path / 'profile.json'
    value = {'version': 1, 'name': 'test', 'settings': {'generation_concurrency': 7,
             'embedding_batch_size': 16, 'embedding_concurrency': 2, 'benchmark_concurrency': 4}}
    path.write_text(json.dumps(value))
    monkeypatch.setenv('RAG_EXECUTION_PROFILE', str(path))
    digest = execution_profile()['sha256']
    path.write_text(json.dumps(value, indent=4))
    assert execution_profile()['sha256'] == digest
    result = subprocess.run([sys.executable, 'core/strategy_registry.py', '--paper-defaults-tsv'],
                            check=True, capture_output=True, text=True)
    assert 'RAG_BENCHMARK_CONCURRENCY\t4' in result.stdout
    value['settings']['benchmark_concurrency'] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        execution_profile()


@pytest.fixture
def queue():
    received = []
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = self.rfile.read(int(self.headers['Content-Length']))
            received.append((self.path, payload, self.headers.get('Authorization')))
            time.sleep(.035)
            if payload == b'fail':
                self.send_response(429)
                self.send_header('Retry-After', '3')
                self.end_headers()
                self.wfile.write(b'{"error":"busy"}')
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(payload)
    class UpstreamServer(ThreadingHTTPServer):
        request_queue_size = 256
    upstream = UpstreamServer(('127.0.0.1', 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    server = QueueServer(f'http://127.0.0.1:{upstream.server_port}/v1', 'upstream-key', 'local-token',
                         {'generation': 2, 'embedding': 1}, {'test': True})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server, received
    finally:
        server.shutdown()
        server.server_close()
        server.client.close()
        upstream.shutdown()
        upstream.server_close()


def test_queue_bounds_concurrency_preserves_bytes_and_errors(queue):
    server, received = queue
    base = f'http://127.0.0.1:{server.server_port}/v1'
    headers = {'Authorization': 'Bearer local-token'}
    payload = b'data: {"delta":"unchanged"}\n\ndata: [DONE]\n\n'
    with httpx.Client(timeout=10, trust_env=False) as client:
        def send(i):
            endpoint = '/embeddings' if i % 3 == 0 else '/chat/completions'
            return client.post(base + endpoint, content=payload, headers=headers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(send, range(12)))
        assert all(response.status_code == 200 and response.content == payload for response in responses)
        failure = client.post(base + '/chat/completions', content=b'fail', headers=headers)
        assert failure.status_code == 429 and failure.headers['retry-after'] == '3'
        assert len(received) == 13  # Proxy does not retry.
        assert all(row[2] == 'Bearer upstream-key' for row in received)
        assert client.get(base + '/queue-metrics').status_code == 401
        assert client.post(base + '/unknown', headers=headers).status_code == 404
    metrics = server.snapshot()['metrics']
    assert metrics['generation']['peak_active'] == 2
    assert metrics['embedding']['peak_active'] == 1
    assert metrics['generation']['errors'] == 1
    assert metrics['generation']['active'] == metrics['embedding']['active'] == 0


def test_queue_is_shared_across_client_processes(queue):
    server, received = queue
    command = [sys.executable, '-c', '''import urllib.request
req=urllib.request.Request(__import__('sys').argv[1], data=b'process',
 headers={'Authorization':'Bearer local-token'})
assert urllib.request.urlopen(req).read()==b'process'
''', f'http://127.0.0.1:{server.server_port}/v1/embeddings']
    children = [subprocess.Popen(command) for _ in range(4)]
    assert [child.wait(timeout=10) for child in children] == [0] * 4
    assert len(received) == 4
    assert server.snapshot()['metrics']['embedding']['peak_active'] == 1


def test_queue_rejects_overflow_without_forwarding(queue):
    server, received = queue
    acquired = 0
    while server.slots.acquire(blocking=False):
        acquired += 1
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            response = client.post(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                                   content=b'overflow', headers={'Authorization': 'Bearer local-token'})
        assert response.status_code == 503
        assert not received
        assert server.snapshot()['rejected_connections'] == 1
    finally:
        for _ in range(acquired):
            server.slots.release()


@pytest.mark.asyncio
async def test_measured_targets_cannot_overlap(tmp_path, monkeypatch):
    import asyncio

    from core.execution_profile import exclusive_measurement
    monkeypatch.setattr('core.execution_profile.execution_profile', lambda: {'sha256': 'selected'})
    monkeypatch.setattr('core.execution_profile.Path', lambda _: tmp_path / 'measured.lock')
    started, release = asyncio.Event(), asyncio.Event()

    @exclusive_measurement
    async def work():
        started.set()
        await release.wait()
        return 'done'

    first = asyncio.create_task(work())
    await asyncio.wait_for(started.wait(), timeout=2)
    try:
        with pytest.raises(RuntimeError, match='Another measured target'):
            await work()
    finally:
        release.set()
        assert await first == 'done'
    assert await work() == 'done'  # Lock released even across repeated calls.


@pytest.mark.parametrize('profile_version', [1, 3])
def test_foreground_wrapper_owns_queue_and_propagates_exit(queue, tmp_path, profile_version):
    import os
    from pathlib import Path
    server, received = queue
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'version': 1, 'name': 'wrapper-test', 'settings': {
        'generation_concurrency': 2, 'embedding_batch_size': 16,
        'embedding_concurrency': 1, 'benchmark_concurrency': 2}}))
    if profile_version == 3:
        value = json.loads(Path('configs/execution_profiles/index-shared-120.json').read_text())
        value['name'] = 'wrapper-test'
        profile.write_text(json.dumps(value))
    metrics = tmp_path / 'metrics.json'
    env = dict(os.environ, RAG_PAPER_MODE='false', RAG_INFERENCE_BASE_URL=server.upstream,
               RAG_INFERENCE_API_KEY='upstream-key', RAG_GENERATION_MODEL='generation',
               RAG_EMBEDDING_MODEL='embedding')
    env.pop('RAG_QUEUE_PROXY_URL', None)
    env.pop('RAG_QUEUE_TOKEN', None)
    command = [sys.executable, 'scripts/run_with_inference_queue.py', '--profile', str(profile),
               '--metrics', str(metrics), '--', sys.executable, '-c', '''import os, urllib.request
from core.execution_profile import require_queue
assert require_queue() is not None
req=urllib.request.Request(os.environ['RAG_QUEUE_PROXY_URL']+'/chat/completions',
 data=b'wrapper',headers={'Authorization':'Bearer '+os.environ['RAG_QUEUE_TOKEN']})
assert urllib.request.urlopen(req).read()==b'wrapper'
raise SystemExit(7)
''']
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 7, result.stderr
    report = json.loads(metrics.read_text())
    assert report['metrics']['generation']['requests'] == 1
    assert report['metrics']['generation']['active'] == 0
    assert report['profile']['name'] == 'wrapper-test'
    if profile_version == 3:
        assert report['limits'] == {'shared': 120}
        assert report['metrics']['shared']['requests'] == 1
    assert received[0][2] == 'Bearer upstream-key'
    assert 'upstream-key' not in metrics.read_text()


def test_pilot_preserves_failed_candidates_and_writes_no_selected_profile(queue, tmp_path):
    import os
    server, _ = queue
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'version': 1, 'name': 'pilot-test', 'settings': {
        'generation_concurrency': 2, 'embedding_batch_size': 16,
        'embedding_concurrency': 1, 'benchmark_concurrency': 2}}))
    requests = tmp_path / 'requests.jsonl'
    # Fake upstream echoes this JSON without choices: HTTP 200 alone is insufficient.
    requests.write_text(json.dumps({'endpoint': 'chat/completions',
                                   'body': {'model': 'generation', 'messages': []}}) + '\n')
    output, selected = tmp_path / 'pilot.json', tmp_path / 'selected.json'
    env = dict(os.environ, RAG_INFERENCE_BASE_URL=server.upstream,
               RAG_INFERENCE_API_KEY='upstream-key', RAG_GENERATION_MODEL='generation')
    result = subprocess.run([sys.executable, 'scripts/pilot_inference_queue.py',
                             '--requests', str(requests), '--profiles', str(profile),
                             '--output', str(output), '--selected-profile', str(selected)],
                            env=env, capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 1, result.stderr
    report = json.loads(output.read_text())
    assert report['results'][0]['successes'] == 0
    assert report['selected_profile_sha256'] is None
    assert not selected.exists()


def test_development_client_cannot_bypass_owned_queue(monkeypatch):
    from core.vllm_client import VLLMClient
    monkeypatch.setenv('RAG_PAPER_MODE', 'false')
    monkeypatch.setenv('RAG_INFERENCE_BASE_URL', 'http://upstream.test/v1')
    monkeypatch.setenv('RAG_INFERENCE_API_KEY', 'upstream-secret')
    monkeypatch.setenv('RAG_GENERATION_MODEL', 'generation')
    monkeypatch.setenv('RAG_EMBEDDING_MODEL', 'embedding')
    monkeypatch.setenv('RAG_QUEUE_PROXY_URL', 'http://127.0.0.1:12345/v1')
    monkeypatch.setenv('RAG_QUEUE_TOKEN', 'local-token')
    monkeypatch.setattr('core.vllm_client.tiktoken.get_encoding', lambda _: None)
    client = VLLMClient()
    assert client.vllm_url == client.embed_url == 'http://127.0.0.1:12345/v1'
    assert client.api_key == 'local-token'


def test_high_concurrency_burst_reaches_queue_without_tcp_backlog_loss(queue):
    fixture, received = queue
    server = QueueServer(fixture.upstream, 'upstream-key', 'burst-token',
                         {'generation': 32, 'embedding': 2}, {'test': 'burst'})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with httpx.Client(timeout=15, trust_env=False) as client:
            def send(_):
                return client.post(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                                   content=b'burst', headers={'Authorization': 'Bearer burst-token'})
            with concurrent.futures.ThreadPoolExecutor(max_workers=96) as pool:
                responses = list(pool.map(send, range(96)))
        assert all(response.status_code == 200 and response.content == b'burst' for response in responses)
        assert len(received) == 96
        assert server.snapshot()['metrics']['generation']['requests'] == 96
        assert server.request_queue_size >= 96
    finally:
        server.shutdown()
        server.server_close()
        server.client.close()


def test_shared_profile_resolves_one_limit_and_removes_old_client_caps(monkeypatch, tmp_path):
    from pathlib import Path

    from core.execution_profile import apply_execution_profile, queue_limits
    path = Path('configs/execution_profiles/index-shared-120.json').resolve()
    monkeypatch.setenv('RAG_EXECUTION_PROFILE', str(path))
    profile = execution_profile()
    assert queue_limits(profile) == {'shared': 120}
    env = {'RAG_GENERATION_CONCURRENCY': '60', 'RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS': '2'}
    apply_execution_profile(env)
    assert env['RAG_GENERATION_CONCURRENCY'] == env['RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS'] == '120'
    result = subprocess.run([sys.executable, '-c', '''
from core.strategy_registry import PAPER_TRANSPORT, get_strategy
assert PAPER_TRANSPORT.generation_concurrency == PAPER_TRANSPORT.embedding_concurrency == 120
policy = dict(get_strategy('youtu_graphrag').paper_index_policy)
assert policy['construction_concurrency'] == 60
assert policy['schema_update_policy'] == 'locked-native-v1'
'''], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    value = json.loads(path.read_text())
    value['settings']['embedding_concurrency'] = 2
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(value))
    monkeypatch.setenv('RAG_EXECUTION_PROFILE', str(bad))
    with pytest.raises(ValueError):
        execution_profile()


@pytest.mark.asyncio
@pytest.mark.parametrize('limit,kind', [(3, 'generation'), (3, 'embedding'), (120, 'mixed')])
async def test_shared_queue_borrows_all_slots_and_bounds_mixed_requests(limit, kind):
    import asyncio

    release = threading.Event()
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            payload = self.rfile.read(int(self.headers['Content-Length']))
            assert release.wait(timeout=20)
            self.send_response(429 if payload == b'fail' else 200)
            self.end_headers()
            self.wfile.write(payload)
    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 256
    upstream = Server(('127.0.0.1', 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    server = QueueServer(f'http://127.0.0.1:{upstream.server_port}/v1', 'key', 'token',
                         {'shared': limit}, {'version': 3})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    tasks = []
    try:
        async with httpx.AsyncClient(timeout=25, trust_env=False,
                limits=httpx.Limits(max_connections=256)) as client:
            base = f'http://127.0.0.1:{server.server_port}/v1'
            async def send(i):
                embed = kind == 'embedding' or kind == 'mixed' and i % 2 == 0
                return await client.post(base + ('/embeddings' if embed else '/chat/completions'),
                    content=b'body', headers={'Authorization': 'Bearer token'})
            tasks = [asyncio.create_task(send(i)) for i in range(limit + 4)]
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                metrics = server.snapshot()['metrics']
                assert metrics['shared']['active'] == metrics['generation']['active'] + metrics['embedding']['active']
                assert metrics['shared']['active'] <= limit
                if metrics['shared']['active'] == limit and metrics['shared']['waiting'] == 4:
                    break
                await asyncio.sleep(.02)
            else:
                pytest.fail('Shared queue did not fill and backpressure excess requests')
            if kind == 'mixed':
                assert metrics['generation']['active'] > 0 and metrics['embedding']['active'] > 0
            else:
                assert metrics[kind]['active'] == limit
            release.set()
            responses = await asyncio.gather(*tasks)
            assert all(r.status_code == 200 and r.content == b'body' for r in responses)
            failure = await client.post(base + '/embeddings', content=b'fail',
                                       headers={'Authorization': 'Bearer token'})
            assert failure.status_code == 429
            metrics = server.snapshot()['metrics']
            assert metrics['shared']['peak_active'] == limit
            assert metrics['shared']['requests'] == limit + 5
            assert metrics['shared']['errors'] == metrics['embedding']['errors'] == 1
            assert metrics['shared']['active'] == metrics['shared']['waiting'] == 0
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        server.shutdown()
        server.server_close()
        server.client.close()
        upstream.shutdown()
        upstream.server_close()


def test_cancelled_waiting_run_never_reaches_upstream(queue):
    server, received = queue
    permit = server.permits['generation']
    permit.acquire()
    permit.acquire()
    base = f'http://127.0.0.1:{server.server_port}/v1'
    headers = {'Authorization': 'Bearer local-token', 'X-Prehop-Run-ID': 'discarded'}
    results = []
    def request():
        results.append(httpx.post(base + '/chat/completions', headers=headers, content=b'old', timeout=5).status_code)
    worker = threading.Thread(target=request)
    worker.start()
    try:
        deadline = time.monotonic() + 3
        while server.snapshot()['metrics']['generation']['waiting'] != 1:
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert httpx.post(base + '/queue-cancel', headers=headers).status_code == 200
        worker.join(3)
        assert results == [409]
        assert received == []
        assert server.snapshot()['metrics']['generation']['waiting'] == 0
    finally:
        permit.release()
        permit.release()
    assert httpx.post(base + '/chat/completions', headers={'Authorization': 'Bearer local-token'}, content=b'live').status_code == 200


def test_disconnected_waiter_is_removed(queue):
    import socket
    server, received = queue
    permit = server.permits['generation']
    permit.acquire()
    permit.acquire()
    connection = socket.create_connection(('127.0.0.1', server.server_port))
    connection.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer local-token\r\nContent-Length: 3\r\n\r\nold')
    try:
        deadline = time.monotonic() + 3
        while not server.snapshot()['metrics']['generation']['waiting']:
            assert time.monotonic() < deadline
            time.sleep(.01)
        connection.close()
        while server.snapshot()['metrics']['generation']['waiting']:
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert received == []
    finally:
        connection.close()
        permit.release()
        permit.release()
