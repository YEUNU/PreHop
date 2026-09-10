#!/usr/bin/env python3
"""Plan, launch and inspect a configuration-bound, session-independent paper campaign."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.pending')
    with temporary.open('x') as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def identity(pid: int) -> dict:
    from scripts.recovery_checkpoint import process_start
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


def build_steps(campaign: str, attempt: str, python: str, target_attempts: dict | None = None) -> list[dict]:
    from core.strategy_registry import PRIMARY_STRATEGIES
    from scripts.campaign_attempts import attempt_mapping, selected_attempt
    target_attempts = attempt_mapping(target_attempts)
    matrix_args = ["--target-attempts-json", json.dumps(target_attempts, sort_keys=True)] if target_attempts else []
    ledger = f'data/results/{campaign}/gate_ledger.json'
    stage = [python, 'scripts/paper_stage_runner.py']
    steps = [{'id': 'runtime_setup', 'argv': [*stage, 'reattest', campaign, '--attempt', attempt]}]
    for name in ('preflight_16', 'chat_probe', 'embedding_probe', 'bisection_probe'):
        steps.append({'id': name, 'argv': [python, 'scripts/paper_gate_ledger.py', 'execute', '--ledger', ledger,
                                          '--campaign', campaign, '--stage', name]})
    for method in PRIMARY_STRATEGIES:
        for dataset in ('multihoprag', 'hotpotqa'):
            steps.append({'id': f'cold/{dataset}/{method}', 'argv': [python, 'scripts/paper_cold_canary.py', campaign,
                          method, dataset, '--attempt', selected_attempt(attempt, target_attempts, f'cold/{dataset}/{method}')]})
    steps.extend([{'id': 'cold_canary_16', 'argv': [*stage, 'cold-aggregate', campaign, '--attempt', attempt, *matrix_args]},
                  {'id': 'resume_stale_rejection', 'argv': [*stage, 'recovery', campaign, '--attempt', attempt]}])
    for method in PRIMARY_STRATEGIES:
        for dataset in ('multihoprag', 'hotpotqa'):
            steps.append({'id': f'one-query/{dataset}/{method}', 'argv': [*stage, 'one-query', campaign,
                          '--strategy', method, '--dataset', dataset, '--attempt', selected_attempt(attempt, target_attempts, f'one-query/{dataset}/{method}')]})
    steps.extend([{'id': 'one_query_matrix_16', 'argv': [*stage, 'one-query-aggregate', campaign, '--attempt', attempt, *matrix_args]},
        {'id': 'full_target_admitted', 'argv': [*stage, 'full-target', campaign, '--strategy', 'naive',
                                              '--dataset', 'multihoprag', '--attempt', attempt]},
        {'id': 'full_matrix', 'argv': ['bash', 'scripts/run_paper_matrix.sh', campaign]}])
    return steps


def create_plan(campaign: str, commit: str, attempt: str) -> Path:
    from scripts.paper_cold_canary import save
    from scripts.paper_gate_ledger import _context
    from scripts.paper_stage_runner import selected_python_environment, validate_name
    from utils.provenance import code_provenance
    validate_name(campaign)
    validate_name(attempt)
    selected = selected_python_environment()
    root = ROOT / 'data/results' / campaign
    plan = {'schema_version': 1, 'campaign': campaign, 'commit': commit, 'attempt': attempt,
        'python': selected['PYTHON_BIN'], 'python_prefix': selected['UV_PROJECT_ENVIRONMENT'],
        'provenance': code_provenance(),
        'context': _context(), 'steps': build_steps(campaign, attempt, selected['PYTHON_BIN'])}
    target = root / 'supervisor/plan.json'
    save(target, plan)
    return target


def create_successor_plan(previous_path: Path, failed_step: str, attempt: str, segment: str) -> Path:
    from scripts.campaign_attempts import attempt_mapping
    from scripts.paper_cold_canary import save
    from scripts.paper_stage_runner import reference, validate_name
    from utils.provenance import code_provenance
    validate_name(segment)
    attempt_mapping({failed_step: attempt})
    handle = lock(resource_lock_path())
    try:
        ensure_no_other_campaigns()
        previous = json.loads(previous_path.read_text())
        status_path = previous_path.parent / 'status.json'
        status = json.loads(status_path.read_text())
        attempts = {**previous.get('target_attempts', {}), failed_step: attempt}
        successor = {**previous, 'schema_version': 2, 'target_attempts': attempts,
                     'predecessor_plan': reference(previous_path), 'predecessor_status': reference(status_path),
                     'retry_step': failed_step, 'inherited_completed_steps': list(status.get('completed_steps', [])),
                     'provenance': code_provenance(),
                     'steps': build_steps(previous['campaign'], previous['attempt'], previous['python'], attempts)}
        _branch, _dataset, _method = failed_step.split('/')
        for step in successor['steps']:
            if step['id'] in successor['inherited_completed_steps']:
                step_evidence(successor, step)
        target = ROOT / 'data/results' / previous['campaign'] / 'supervisor/segments' / segment / 'plan.json'
        save(target, successor)
        return target
    finally:
        handle.close()


def safe_environment() -> dict[str, str]:
    """Only approved runtime selectors and canonical connection fields enter the unit."""
    from core.inference_transport import _FORBIDDEN_AMBIENT_PROVIDER_KEYS, preserve_provider_environment
    from core.paper_policy import method_environment_defaults, preserve_method_environment
    from core.strategy_registry import paper_environment_defaults
    from scripts.paper_stage_runner import selected_python_environment
    current = selected_python_environment()
    preserve_provider_environment(current)
    preserve_method_environment(current)
    allowed = {'RAG_MEASUREMENT_MAX_TARGETS', 'RAG_PREHOP_TRACE', 'RAG_PREHOP_TRACE_DIR', 'RAG_EXECUTION_PROFILE', 'RAG_QUEUE_PROXY_URL', 'RAG_QUEUE_TOKEN', 'PYTHON_BIN', 'UV_PROJECT_ENVIRONMENT', 'RAG_OFFICIAL_BASELINE_HOME',
        'RAG_INFERENCE_BASE_URL', 'RAG_INFERENCE_API_KEY', 'RAG_GENERATION_MODEL', 'RAG_EMBEDDING_MODEL',
        'RAG_GENERATION_REVISION', 'RAG_EMBEDDING_REVISION', 'NEO4J_URI', 'NEO4J_URL', 'NEO4J_USERNAME', 'NEO4J_USER',
        'NEO4J_PASSWORD', 'NEO4J_DATABASE', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'PATH', 'HOME', 'LITELLM_MODE'}
    allowed.update(_FORBIDDEN_AMBIENT_PROVIDER_KEYS)
    allowed.update(method_environment_defaults())
    allowed.update(paper_environment_defaults())
    env = {key: current[key] for key in allowed if key in current}
    env.update(PYTHONDONTWRITEBYTECODE='1', RAG_SKIP_PROJECT_ENV='true', RAG_PAPER_MODE='true')
    return env


def require_logout_persistence() -> None:
    subprocess.run(['loginctl', 'show-user', str(os.getuid()), '-p', 'Linger', '--value'],
                               check=True, text=True, capture_output=True)
    subprocess.run(['systemctl', '--user', 'show-environment'], check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def unit_processes(unit: str) -> list[dict]:
    output = subprocess.run(['systemctl', '--user', 'show', unit, '--property=ControlGroup', '--value'],
                            check=True, capture_output=True, text=True).stdout.strip()
    if not output:
        return []
    path = Path('/sys/fs/cgroup') / output.lstrip('/')
    processes = {}
    for file in path.rglob('cgroup.procs'):
        for raw in file.read_text().splitlines():
            try:
                value = identity(int(raw))
                processes[value['pid']] = value
            except (OSError, ProcessLookupError):
                continue
    return list(processes.values())


def ensure_no_other_campaigns(own_unit: str = '') -> None:
    from scripts.paper_detached_runtime import ensure_no_detached_owners
    ensure_no_detached_owners(os.getpid())
    try:
        output = subprocess.run(['systemctl', '--user', 'list-units', 'prehop-paper-*', '--all', '--plain', '--no-legend'],
                                check=True, capture_output=True, text=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        if own_unit:
            raise
        return  # nohup ownership and the shared flock do not require a user manager.
    own_unit.removesuffix('.service') + '.service' if own_unit else ''
    for line in output.splitlines():
        line.split()[0] if line.split() else ''


def launch(plan_path: Path, *, resume: bool = False, backend: str = 'nohup') -> dict:
    if backend == 'nohup':
        from scripts.paper_detached_runtime import launch as launch_detached
        return launch_detached(plan_path, resume=resume)
    from scripts.paper_stage_runner import reference
    plan = json.loads(plan_path.read_text())
    require_logout_persistence()
    ensure_no_other_campaigns()
    root = plan_path.parent
    current = root / 'status.json'
    if current.exists():
        json.loads(current.read_text())
    resource_lock = lock(resource_lock_path())
    resource_lock.close()  # Supervisor acquires and passes the same lock into its child.
    launch_id = str(time.time_ns())
    unit = f'prehop-paper-{plan["campaign"]}-{launch_id}'
    env_path = root / f'environment-{launch_id}.private'
    descriptor = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        for key, value in sorted(safe_environment().items()):
            quoted = value.replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`').replace('$', '\\$')
            stream.write(f'{key}="{quoted}"\n')
    command = ['systemd-run', '--user', '--unit', unit,
        '--property', f'WorkingDirectory={ROOT}', '--property', f'EnvironmentFile={env_path}',
        '--property', 'Restart=no', '--property', 'UMask=0077', '--property', 'KillMode=control-group',
        '--property', 'SendSIGKILL=no', '--property', 'TimeoutStopSec=30',
        '--property', 'StandardOutput=null', '--property', 'StandardError=null',
        plan['python'], str(Path(__file__).resolve()), 'supervise', str(plan_path), '--unit', unit]
    if resume:
        command.append('--resume')
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    observed = None
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        probe = subprocess.run(['systemctl', '--user', 'show', unit, '--property=MainPID', '--property=ActiveState', '--property=Result'],
                               check=True, capture_output=True, text=True)
        fields = dict(line.split('=', 1) for line in probe.stdout.splitlines() if '=' in line)
        if current.exists():
            running = json.loads(current.read_text())
            pid = int(fields.get('MainPID', '0'))
            if pid and fields.get('ActiveState') == 'active' and running.get('unit') == unit and alive(running.get('supervisor')) and running['supervisor']['pid'] == pid:
                observed = running['supervisor']
                break
        time.sleep(.1)
    receipt = {'unit': unit, 'plan': reference(plan_path), 'status_path': str(current), 'supervisor': observed,
               'logout_persistence': 'verified_own_user_linger', 'launched_at': time.time()}
    atomic_json(root / f'launch-{launch_id}.json', receipt)
    return receipt


def step_evidence(plan: dict, step: dict) -> list[dict]:
    """Read a completed step's artifact reference without revalidating its contents."""
    from scripts.campaign_attempts import selected_attempt
    from scripts.paper_stage_runner import reference
    campaign = plan['campaign']
    if step['id'] == 'full_matrix':
        return final_admissions(campaign)
    if '/' in step['id']:
        branch, dataset, method = step['id'].split('/')
        folder = 'cold_v2' if branch == 'cold' else 'one_query'
        path = ROOT / 'data/results' / campaign / folder / selected_attempt(plan['attempt'], plan.get('target_attempts'), step['id']) / dataset / method / 'evidence.json'
    else:
        ledger = json.loads((ROOT / 'data/results' / campaign / 'gate_ledger.json').read_text())
        path = ROOT / ledger['stages'][step['id']]['evidence_path']
    return [reference(path)]


def admission_statuses(campaign: str) -> list[dict]:
    from core.strategy_registry import PRIMARY_STRATEGIES
    from scripts.paper_stage_runner import reference
    statuses = []
    for method in PRIMARY_STRATEGIES:
        for dataset in ('multihoprag', 'hotpotqa'):
            path = ROOT / 'data/results' / f'{campaign}-{dataset}-{method}' / 'admission.json'
            row = {'target': f'{dataset}/{method}', 'status': 'missing'}
            if path.is_file():
                try:
                    row.update(status='admitted', **reference(path))
                except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
                    row.update(status='invalid', error_category=type(exc).__name__)
            statuses.append(row)
    return statuses


def final_admissions(campaign: str) -> list[dict]:
    statuses = admission_statuses(campaign)
    return statuses


def checkpoint_progress(campaign: str) -> list[dict]:
    """Read only actual benchmark checkpoints; indexing progress remains in logs."""
    from core.strategy_registry import PRIMARY_STRATEGIES
    names = {f'{method}_{dataset}.json' for method in PRIMARY_STRATEGIES for dataset in ('multihoprag', 'hotpotqa')}
    rows = []
    for folder in (ROOT / 'data/results').glob(campaign + '-*'):
        for path in folder.rglob('*.json'):
            if path.name not in names:
                continue
            try:
                payload = json.loads(path.read_text())
                if isinstance(payload.get('total_queries'), int):
                    rows.append({'path': str(path), 'completed': payload.get('queries_count'),
                                 'total': payload['total_queries'], 'status': payload.get('status')})
            except (OSError, ValueError, TypeError):
                continue
    return rows


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


# Bound only the drain after the direct child exits. Running stages retain
# their ordinary unlimited runtime and heartbeat updates.
CHILD_LOG_DRAIN_SECONDS = 5.0


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


def supervise(plan_path: Path, *, resume: bool = False, unit: str = '', detached: bool = False) -> int:
    from scripts.paper_detached_runtime import register_owner, session_processes, terminate_owned
    owner = register_owner(plan_path) if detached else None
    plan = json.loads(plan_path.read_text())
    root = plan_path.parent
    status_path = root / 'status.json'
    handle = lock(resource_lock_path())
    try:
        ensure_no_other_campaigns(unit)
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    except Exception:
        handle.close()
        raise
    if previous and (not resume or alive(previous.get('supervisor')) or alive(previous.get('child')) or any(alive(row) for row in previous.get('remaining_owned_processes', []))):
        handle.close()
        raise RuntimeError('Existing campaign owner or child prevents this supervisor launch')
    from utils.provenance import code_provenance
    status = {'campaign': plan['campaign'], 'commit': plan['commit'], 'unit': unit, 'state': 'planned',
              'backend': 'nohup' if detached else 'systemd', 'session_id': os.getsid(0) if detached else None,
              'supervisor': identity(os.getpid()), 'child': None, 'started_at': time.time(),
              'completed_steps': list(previous.get('completed_steps', []) if resume else plan.get('inherited_completed_steps', []))}
    def update(fields):
        if 'heartbeat_at' in fields:
            fields['checkpoints'] = checkpoint_progress(plan['campaign'])
        if detached:
            from scripts.paper_detached_runtime import observe_owner
            fields['observed_owned_processes'] = observe_owner(owner)
        status.update(fields)
        status['updated_at'] = time.time()
        atomic_json(status_path, status)
        with (root / 'events.jsonl').open('a') as events:
            events.write(json.dumps({'time': status['updated_at'], **fields}, sort_keys=True) + '\n')
    if detached:
        import signal
        def interrupted(signum, frame):
            raise RuntimeError('Owned detached supervisor received termination signal')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
    from core.inference_queue import OwnedQueue
    queue = OwnedQueue(root / f'inference-queue-{time.time_ns()}.json')
    try:
        queue.start()
        update({'state': 'running'})
        env = safe_environment()
        for index, step in enumerate(plan['steps']):
            if step['id'] in status['completed_steps']:
                step_evidence(plan, step)
                continue
            stamp = time.time_ns()
            log_base = root / f'{index:02d}-{stamp}'
            update({'stage': step['id'], 'child': None, 'exit_code': None, 'log_drain_complete': None,
                    'segment_provenance': code_provenance(),
                    'stdout_log': str(log_base.with_suffix('.stdout.log')), 'stderr_log': str(log_base.with_suffix('.stderr.log'))})
            exit_code = run_child(step['argv'], env, log_base, handle, update)
            queue.drain()
            queue.persist()
            update({'child': None, 'exit_code': exit_code})
            remaining = [row for row in (session_processes(owner) if detached else unit_processes(unit))
                         if row['pid'] != os.getpid()]
            update({'remaining_owned_processes': remaining})
            if exit_code:
                raise RuntimeError(f'Owned stage {step["id"]} exited with status {exit_code}')
            evidence = step_evidence(plan, step)
            status['completed_steps'].append(step['id'])
            update({'last_evidence': evidence, 'completed_steps': status['completed_steps']})
        admissions = final_admissions(plan['campaign'])
        update({'state': 'completed', 'exit_code': 0, 'finished_at': time.time(), 'admissions': admissions})
        return 0
    except Exception as exc:  # noqa: BLE001 - persist terminal status for every stage failure
        if status.get('stage') == 'full_matrix':
            update({'admissions': admission_statuses(plan['campaign'])})
        update({'state': 'failed', 'exit_code': status.get('exit_code') or 1,
                'failure_category': type(exc).__name__, 'failure': redact(str(exc), os.environ), 'finished_at': time.time()})
        return 1
    finally:
        if detached:
            import signal
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            remaining = terminate_owned(owner)
            update({'remaining_owned_processes': remaining, 'owned_cleanup': 'TERM_only',
                    'owned_cleanup_complete': not remaining})
        queue.close()
        if handle is not None:
            handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'retry-plan', 'launch', 'supervise', 'status', 'monitor'])
    parser.add_argument('target', help='Campaign ID for plan; exact plan.json path otherwise')
    parser.add_argument('--commit')
    parser.add_argument('--attempt', default='a1')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--unit', default='')
    parser.add_argument('--backend', choices=['nohup', 'systemd'], default='nohup')
    parser.add_argument('--detached', action='store_true')
    parser.add_argument('--retry-step')
    parser.add_argument('--segment')
    args = parser.parse_args()
    if args.action == 'status':
        path = Path(args.target).resolve().parent / 'status.json'
        print(path.read_text())
        return 0
    from scripts.runner_environment import _load_runner_environment
    _load_runner_environment()
    from core.execution_profile import apply_execution_profile
    apply_execution_profile()
    if args.action == 'plan':
        print(create_plan(args.target, args.commit, args.attempt))
        return 0
    plan_path = Path(args.target).resolve()
    if args.action == 'retry-plan':
        if not args.retry_step or not args.segment:
            parser.error('retry-plan requires --retry-step and --segment')
        print(create_successor_plan(plan_path, args.retry_step, args.attempt, args.segment))
        return 0
    if args.action == 'launch':
        print(json.dumps(launch(plan_path, resume=args.resume, backend=args.backend), sort_keys=True))
        return 0
    if args.action == 'monitor':
        from scripts.paper_detached_runtime import monitor
        return monitor(plan_path)
    if not args.unit and not args.detached:
        parser.error('supervise requires actual ownership; use launch')
    return supervise(plan_path, resume=args.resume, unit=args.unit, detached=args.detached)


if __name__ == '__main__':
    raise SystemExit(main())
