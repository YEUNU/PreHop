#!/usr/bin/env python3
"""Own a bounded queue for a sequential paper campaign or a single target."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a child command is required after --')
    if args.metrics.exists():
        parser.error('preserve existing metrics; select a fresh output path')
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
    os.environ['RAG_EXECUTION_PROFILE'] = str(args.profile.resolve())
    from core.execution_profile import execution_profile, queue_limits
    from core.inference_queue import QueueServer
    from core.inference_transport import InferenceTransport
    from core.strategy_registry import paper_environment_defaults
    profile = execution_profile()
    for name, value in paper_environment_defaults().items():
        os.environ[name] = value
    # Validate the canonical upstream before introducing the local transport.
    os.environ.pop('RAG_QUEUE_PROXY_URL', None)
    os.environ.pop('RAG_QUEUE_TOKEN', None)
    os.environ.setdefault('RAG_LLM_SEED', '42')
    transport = InferenceTransport.resolve('core')
    lock_path = Path(f'/tmp/prehop-inference-queue-{os.getuid()}.lock')
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        token = secrets.token_urlsafe(32)
        server = QueueServer(transport.generation_base_url, transport.api_key, token,
                             queue_limits(profile), profile,
                             timeout=transport.timeout_seconds or None)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        env = dict(os.environ, RAG_QUEUE_PROXY_URL=f'http://127.0.0.1:{server.server_port}/v1',
                   RAG_QUEUE_TOKEN=token)
        child = None
        def forward(signum, _frame):
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signum)
        previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            child = subprocess.Popen(command, env=env, start_new_session=True)
            result = child.wait()
            server.drain()
            return result
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            server.shutdown()
            server.server_close()
            server.client.close()
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            with args.metrics.open('x') as stream:
                json.dump(server.snapshot(), stream, indent=2)
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
