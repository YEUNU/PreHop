#!/usr/bin/env python3
"""Compare fixed request workloads across profiles; writes no paper results."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', required=True, type=Path,
                        help='JSONL rows: {"endpoint":"chat/completions"|"embeddings", "body":{...}}')
    parser.add_argument('--profiles', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--selected-profile', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.selected_profile.exists():
        parser.error('output paths must be fresh')
    import httpx
    from dotenv import load_dotenv

    from core.execution_profile import execution_profile
    from core.inference_queue import QueueServer
    load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
    raw = args.requests.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows or len(rows) > 4096:
        parser.error('pilot needs 1..4096 representative requests')
    for row in rows:
        if set(row) != {'endpoint', 'body'} or row['endpoint'] not in {'chat/completions', 'embeddings'}:
            parser.error('invalid request row')
        model = os.environ['RAG_EMBEDDING_MODEL' if row['endpoint'] == 'embeddings' else 'RAG_GENERATION_MODEL']
        if not isinstance(row['body'], dict) or row['body'].get('model') != model or row['body'].get('stream'):
            parser.error('pilot requests require the configured model and non-streaming responses')
    # Share the execution lock with production queue owners.
    import fcntl
    with Path(f'/tmp/prehop-inference-queue-{os.getuid()}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = []
        for profile_path in args.profiles:
            os.environ['RAG_EXECUTION_PROFILE'] = str(profile_path.resolve())
            profile = execution_profile()
            settings = profile['settings']
            for row in rows:
                if row['endpoint'] == 'embeddings':
                    inputs = row['body'].get('input')
                    count = len(inputs) if isinstance(inputs, list) else 1
                    if count > settings['embedding_batch_size']:
                        parser.error('embedding payload exceeds candidate batch size; keep batches fixed across candidates')
            token = secrets.token_urlsafe(32)
            server = QueueServer(os.environ['RAG_INFERENCE_BASE_URL'], os.environ['RAG_INFERENCE_API_KEY'], token,
                                 {'generation': settings['generation_concurrency'],
                                  'embedding': settings['embedding_concurrency']}, profile)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                with httpx.Client(timeout=600, trust_env=False) as client:
                    def send(row, server=server, token=token):
                        try:
                            response = client.post(f'http://127.0.0.1:{server.server_port}/v1/' + row['endpoint'],
                                                   json=row['body'], headers={'Authorization': f'Bearer {token}'})
                            response.raise_for_status()
                            value = response.json()
                            valid = bool(value.get('choices')) if row['endpoint'] == 'chat/completions' else bool(value.get('data'))
                            return valid, 'ok' if valid else 'invalid_response_envelope'
                        except (httpx.HTTPError, ValueError) as exc:
                            response = getattr(exc, 'response', None)
                            category = f'http_{response.status_code}' if response is not None else type(exc).__name__
                            return False, category
                    started = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=2 * sum(server.limits.values())) as pool:
                        outcomes = list(pool.map(send, rows))
                        successes = sum(success for success, _ in outcomes)
                    elapsed = time.perf_counter() - started
                result = {'profile': profile, 'wall_seconds': elapsed, 'successes': successes,
                          'requests': len(rows), 'successful_requests_per_second': successes / elapsed,
                          'outcome_counts': dict(Counter(category for _, category in outcomes)),
                          'queue': server.snapshot()}
                results.append(result)
                print(json.dumps({key: result[key] for key in ('wall_seconds', 'successes', 'requests', 'successful_requests_per_second')}), flush=True)
            finally:
                server.shutdown()
                server.server_close()
                server.client.close()
    eligible = [row for row in results if row['successes'] == row['requests']]
    selected = max(eligible, key=lambda row: row['successful_requests_per_second']) if eligible else None
    payload = {'version': 1, 'scope': 'non_reportable_transport_pilot',
               'requests_sha256': hashlib.sha256(raw).hexdigest(), 'results': results,
               'selected_profile_sha256': selected['profile']['sha256'] if selected else None,
               'limitations': ['One pass in supplied order; repeat with reversed order to assess warmup/cache effects.',
                               'HTTP/response-envelope success does not establish extraction validity or quality equivalence.',
                               'This request replay does not tune native query workers or benchmark concurrency.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(payload, stream, indent=2)
    if not selected:
        return 1
    args.selected_profile.parent.mkdir(parents=True, exist_ok=True)
    with args.selected_profile.open('x') as stream:
        json.dump({key: value for key, value in selected['profile'].items() if key != 'sha256'}, stream, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
