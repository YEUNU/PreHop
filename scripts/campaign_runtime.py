"""Shared process ownership, locking and redacted child logs for experiment runners."""
from __future__ import annotations

import fcntl
import json
import os
import selectors
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHILD_LOG_DRAIN_SECONDS = 5.0


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.pending')
    with temporary.open('x') as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_start(pid: int) -> str:
    # Field 22 follows the final ')' around the possibly spaced process name.
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def identity(pid: int) -> dict:
    return {'pid': pid, 'start': process_start(pid), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def alive(value: dict | None) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get('pid'), int):
        return False
    try:
        return identity(value['pid']) == value
    except (OSError, ProcessLookupError):
        return False


def resource_lock_path() -> Path:
    # Shared across execution worktrees for this OS user, not just one campaign.
    return Path(f'/run/user/{os.getuid()}/prehop-paper-resource.lock')


def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError('Another paper campaign or its owned child still holds the resource lock') from exc
    return handle


def redact(text: str, environment: dict[str, str]) -> str:
    from urllib.parse import urlsplit
    values = set()
    for key, value in environment.items():
        if not value:
            continue
        if any(word in key for word in ('PASSWORD', 'API_KEY', 'TOKEN')):
            values.add(value)
        if key.endswith(('_BASE_URL', '_URI', '_URL')):
            parsed = urlsplit(value)
            values.update(part for part in (value, parsed.netloc, parsed.hostname,
                          f'{parsed.scheme}://{parsed.netloc}') if part)
    for value in sorted(values, key=len, reverse=True):
        text = text.replace(value, '[REDACTED]')
    return text


def run_child(argv: list[str], env: dict[str, str], log_base: Path, handle, update) -> int:
    """Drain both owned pipes without waiting forever on inherited descendant FDs."""
    child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, pass_fds=(handle.fileno(),))
    pending = {}
    streams = {}
    pipes = {'stdout': child.stdout, 'stderr': child.stderr}
    last_update = time.monotonic()
    exited_at = None
    with selectors.DefaultSelector() as selector:
        try:
            update({'child': identity(child.pid), 'child_argv': argv})
            for name, pipe in pipes.items():
                streams[name] = log_base.with_suffix(f'.{name}.log').open('x')
                pending[name] = b''
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, name)
            while selector.get_map() or child.poll() is None:
                if child.poll() is not None and exited_at is None:
                    exited_at = time.monotonic()
                if exited_at is not None and time.monotonic() - exited_at >= CHILD_LOG_DRAIN_SECONDS:
                    break
                for key, _events in selector.select(timeout=.2):
                    name = key.data
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if chunk:
                        pending[name] += chunk
                        while b'\n' in pending[name]:
                            line, pending[name] = pending[name].split(b'\n', 1)
                            streams[name].write(redact((line + b'\n').decode('utf-8', errors='replace'), env))
                        streams[name].flush()
                    else:
                        # Complete final lines (including no trailing newline)
                        # are redacted as a whole, never at arbitrary read cuts.
                        streams[name].write(redact(pending[name].decode('utf-8', errors='replace'), env))
                        streams[name].flush()
                        pending[name] = b''
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                if time.monotonic() - last_update >= 10:
                    update({'heartbeat_at': time.time()})
                    last_update = time.monotonic()
            incomplete = [key.data for key in selector.get_map().values()]
            update({'log_drain_complete': not incomplete, 'log_drain_incomplete_streams': sorted(incomplete),
                    'log_drain_unwritten_partial_bytes': {name: len(pending[name]) for name in incomplete}})
        finally:
            for pipe in pipes.values():
                pipe.close()
            for stream in streams.values():
                stream.close()
    return child.wait()
