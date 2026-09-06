"""Owned nohup sessions and read-only periodic campaign observations."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path


def owner_directory() -> Path:
    path = Path(f'/run/user/{os.getuid()}/prehop-paper-owners')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def process_record(pid: int) -> dict:
    from scripts.paper_campaign import identity
    record = identity(pid)
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {**record, 'ppid': int(fields[1]), 'pgid': int(fields[2]), 'sid': int(fields[3]), 'state': fields[0]}


def session_processes(owner: dict) -> list[dict]:
    """Only the recorded boot/session and processes no older than its leader."""
    from scripts.paper_campaign import identity
    discover = True
    try:
        current = identity(owner['pid'])
        if current != owner:
            discover = False  # Only previously recorded exact descendants survive leader PID reuse.
    except (OSError, ProcessLookupError):
        pass
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if owner.get('boot_id') != boot:
        return []
    processes = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            row = process_record(int(path.name))
            if int(row['start']) >= int(owner['start']) and row['state'] != 'Z':
                processes[row['pid']] = row
        except (OSError, ProcessLookupError, ValueError):
            continue
    selected = {pid for pid, row in processes.items() if discover and row['sid'] == owner['pid']}
    if discover and owner['pid'] in processes:
        selected.add(owner['pid'])
        while True:
            children = {pid for pid, row in processes.items() if row['ppid'] in selected}
            if children.issubset(selected):
                break
            selected.update(children)
    receipt = owner_directory() / f'{owner["pid"]}-{owner["start"]}.json'
    if receipt.exists():
        for known in json.loads(receipt.read_text()).get('observed_descendants', []):
            row = processes.get(known['pid'])
            if row and all(row[key] == known[key] for key in ('pid', 'start', 'boot_id')):
                selected.add(known['pid'])
    return [{key: processes[pid][key] for key in ('pid', 'start', 'boot_id')} for pid in sorted(selected)]


def observe_owner(owner: dict) -> list[dict]:
    from scripts.paper_campaign import atomic_json
    path = owner_directory() / f'{owner["pid"]}-{owner["start"]}.json'
    record = json.loads(path.read_text())
    observed = session_processes(owner)
    record['observed_descendants'] = [row for row in observed if row != owner]
    atomic_json(path, record)
    return observed


def ensure_no_detached_owners(own_pid: int | None = None) -> None:
    from scripts.paper_campaign import identity
    own_identity = identity(own_pid) if own_pid is not None else None
    for path in owner_directory().glob('*.json'):
        owner = json.loads(path.read_text())['supervisor']
        if owner != own_identity and session_processes(owner):
            raise RuntimeError('Another nohup campaign still owns its session or descendants')


def register_owner(plan_path: Path) -> dict:
    from scripts.paper_campaign import atomic_json, identity
    pid = os.getpid()
    if os.getsid(0) != pid or os.getpgrp() != pid or signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
        raise RuntimeError('Detached supervisor requires actual nohup SIGHUP-ignore and its own setsid session')
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    value = ctypes.c_int()
    if libc.prctl(36, 1, 0, 0, 0) != 0 or libc.prctl(37, ctypes.byref(value), 0, 0, 0) != 0 or value.value != 1:
        raise RuntimeError('Detached supervisor could not verify Linux child-subreaper ownership')
    owner = identity(pid)
    ensure_no_detached_owners(pid)
    atomic_json(owner_directory() / f'{pid}-{owner["start"]}.json',
                {'supervisor': owner, 'sid': pid, 'pgid': pid, 'subreaper': True, 'plan_path': str(plan_path)})
    return owner


def terminate_owned(owner: dict, timeout: float = 30.0) -> list[dict]:
    """TERM individually verified own processes; never use KILL or a global process group."""
    from scripts.paper_campaign import alive
    sent = set()
    deadline = time.monotonic() + timeout
    while True:
        receipt = owner_directory() / f'{owner["pid"]}-{owner["start"]}.json'
        observed = observe_owner(owner) if receipt.exists() else session_processes(owner)
        remaining = [row for row in observed if row['pid'] != os.getpid()]
        for row in remaining:
            key = (row['pid'], row['start'])
            if key in sent:
                continue
            try:
                descriptor = _pidfd_syscall(434, row['pid'], 0)
                try:
                    if alive(row):
                        _pidfd_syscall(424, descriptor, signal.SIGTERM, 0, 0)
                        sent.add(key)
                finally:
                    os.close(descriptor)
            except ProcessLookupError:
                pass
        if not remaining or time.monotonic() >= deadline:
            return remaining
        time.sleep(.1)


def _pidfd_syscall(number: int, *args: int) -> int:
    """Linux pidfds also work with Python builds that omit os.pidfd_open."""
    import ctypes
    import platform
    if platform.system() != 'Linux' or platform.machine() not in {'x86_64', 'aarch64', 'riscv64'}:
        raise RuntimeError('Owned cleanup requires a supported Linux pidfd ABI')
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(number, *args)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def spawn_nohup(argv: list[str], *, env: dict, cwd: Path, log: Path) -> subprocess.Popen:
    descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        return subprocess.Popen(['nohup', *argv], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=stream, start_new_session=True, umask=0o077)


def launch(plan_path: Path, *, resume: bool = False) -> dict:
    from scripts import paper_campaign as campaign
    from scripts.paper_stage_runner import reference
    plan = json.loads(plan_path.read_text())
    campaign.check_plan(plan)
    campaign.ensure_no_other_campaigns()
    current = plan_path.parent / 'status.json'
    if current.exists():
        previous = json.loads(current.read_text())
        if not resume or campaign.alive(previous.get('supervisor')) or campaign.alive(previous.get('child')):
            raise RuntimeError('Campaign already launched or has a live owned process; no duplicate launch')
    elif resume:
        raise RuntimeError('Cannot resume a campaign that has never launched')
    handle = campaign.lock(campaign.resource_lock_path())
    handle.close()
    env = campaign.safe_environment()
    stamp = str(time.time_ns())
    private = plan_path.parent / f'environment-{stamp}.private.json'
    descriptor = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(env, stream)
    argv = [plan['python'], str(Path(campaign.__file__).resolve()), 'supervise', str(plan_path), '--detached']
    if resume:
        argv.append('--resume')
    process = spawn_nohup(argv, env=env, cwd=campaign.ROOT, log=plan_path.parent / f'nohup-{stamp}.log')
    expected = campaign.identity(process.pid)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if current.exists():
            status = json.loads(current.read_text())
            if status.get('supervisor') == expected:
                if status.get('state') == 'failed':
                    raise RuntimeError('Detached supervisor failed; inspect its preserved status')
                if campaign.alive(expected) and status.get('session_id') == process.pid:
                    break
        if process.poll() is not None:
            raise RuntimeError('Detached supervisor exited before verified startup; inspect its preserved log')
        time.sleep(.1)
    else:
        raise RuntimeError('Detached startup unverified; inspect preserved PID and status before retry')
    receipt = {'backend': 'nohup', 'supervisor': expected, 'session_id': process.pid, 'process_group_id': process.pid,
               'plan': reference(plan_path), 'status_path': str(current), 'private_environment_path': str(private),
               'logout_persistence': 'actual_nohup_ignored_hup_and_setsid', 'launched_at': time.time()}
    monitor = spawn_nohup([plan['python'], str(Path(campaign.__file__).resolve()), 'monitor', str(plan_path)],
                          env=env, cwd=campaign.ROOT, log=plan_path.parent / f'monitor-{stamp}.log')
    receipt['monitor'] = campaign.identity(monitor.pid)
    receipt['monitor_interval_seconds'] = 10800
    campaign.atomic_json(plan_path.parent / f'launch-{stamp}.json', receipt)
    return receipt


def monitor(plan_path: Path, *, interval: float = 10800, poll: float = 5) -> int:
    """Append observations only; never restart, mutate results, or send chat messages."""
    from scripts import paper_campaign as campaign
    plan = json.loads(plan_path.read_text())
    status_path = plan_path.parent / 'status.json'
    output = plan_path.parent / f'monitor-{os.getpid()}-observations.jsonl'
    last = 0.0
    while True:
        status = json.loads(status_path.read_text())
        supervisor_alive = campaign.alive(status.get('supervisor'))
        if supervisor_alive:
            try:
                supervisor_alive = process_record(status['supervisor']['pid'])['state'] != 'Z'
            except (OSError, ProcessLookupError):
                supervisor_alive = False
        terminal_state = status.get('state') in {'failed', 'completed'}
        cleanup_finished = status.get('owned_cleanup') == 'TERM_only'
        terminal = terminal_state and (cleanup_finished or not supervisor_alive)
        vanished = not supervisor_alive and not terminal
        now = time.time()
        if terminal or vanished or now - last >= interval:
            row = {'observed_at': now, 'state': status.get('state'), 'stage': status.get('stage'),
                   'completed_steps': status.get('completed_steps', []), 'exit_code': status.get('exit_code'),
                   'supervisor_alive': supervisor_alive,
                   'failure_category': status.get('failure_category'),
                   'owned_cleanup_complete': status.get('owned_cleanup_complete'),
                   'remaining_owned_processes': status.get('remaining_owned_processes', []),
                   'checkpoints': campaign.checkpoint_progress(plan['campaign']),
                   'admissions': campaign.admission_statuses(plan['campaign']),
                   'eta_seconds': None, 'eta_reason': 'no comparable measured remaining-target throughput',
                   'automatic_chat_notification': False}
            with output.open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
            last = now
        if terminal or vanished:
            campaign.atomic_json(plan_path.parent / f'monitor-{os.getpid()}-terminal.json', row)
            return 0 if status.get('state') == 'completed' else 1
        time.sleep(poll)
