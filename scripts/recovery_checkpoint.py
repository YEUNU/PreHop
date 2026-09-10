"""Opt-in checkpoint barrier for a freshly owned recovery test child."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def process_start(pid: int) -> str:
    # Field22 follows the final ')' delimiting the possibly spaced process name.
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


async def checkpoint_barrier(result_path: Path, summary: dict) -> None:
    configured = os.environ.get('RAG_RECOVERY_TEST_CONTROL', '')
    if not configured:
        return
    control = Path(configured).resolve()
    run_id = os.environ.get('RAG_RUN_ID', '')
    expected_root = ROOT / 'data/results' / run_id / 'recovery'
    contract = json.loads(control.read_text())
    if summary.get('status') != 'in_progress' or len(summary.get('details', [])) != 1:
        return
    result = result_path.resolve()
    from core.admission import sha256_file
    trace = result.with_name(result.stem + '.traces.jsonl')
    notice = {'run_id': run_id, 'nonce': contract['nonce'], 'pid': os.getpid(),
              'process_start': process_start(os.getpid()), 'result_path': str(result),
              'result_sha256': sha256_file(result), 'trace_path': str(trace), 'trace_sha256': sha256_file(trace),
              'completed': 1, 'total': summary['total_queries']}
    pending = expected_root / 'checkpoint_ready.pending'
    with pending.open('x') as stream:
        json.dump(notice, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, expected_root / 'checkpoint_ready.json')
    # This only blocks the explicitly opted-in owned test child, after both files close.
    await asyncio.sleep(120)
    raise RuntimeError('Owned recovery checkpoint barrier timed out without interruption')
