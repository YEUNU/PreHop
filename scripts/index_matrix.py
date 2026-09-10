#!/usr/bin/env python3
"""Own a tested, sequential primary index matrix; never claim benchmark admission."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source_digest():
    paths = [ROOT / 'main.py']
    for directory in ('core', 'cli', 'models', 'scripts', 'utils'):
        paths.extend(path for path in (ROOT / directory).rglob('*') if path.suffix in {'.py', '.sh'})
    return hashlib.sha256(json.dumps([(str(path.relative_to(ROOT)), hashlib.sha256(path.read_bytes()).hexdigest())
                                     for path in sorted(paths)], separators=(',', ':')).encode()).hexdigest()


def targets(campaign):
    from core.strategy_registry import PRIMARY_STRATEGIES
    return [{'strategy': strategy, 'dataset': dataset, 'run_id': f'{campaign}-index-{dataset}-{strategy}'}
            for strategy in PRIMARY_STRATEGIES for dataset in ('multihoprag', 'hotpotqa')]


def child(campaign, strategy, dataset, phase):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    from core.paper_policy import configure_target_environment
    run_id = f'{campaign}-{phase}-{dataset}-{strategy}'
    configure_target_environment(strategy, dataset, run_id)
    os.environ.update(RAG_CHUNK_CACHE='off', RAG_EMBEDDING_CACHE='off', RAG_JUDGE_ENABLED='false',
                      RAG_JUDGE_BATCH='false', PYTHONDONTWRITEBYTECODE='1')
    from scripts.check_paper_runtime import check
    check(strategy, dataset)
    from cli.index import run_indexing
    from core.admission import sha256_file
    from core.amortized_cost import indexing_cost, validate_cost
    from scripts.paper_cold_canary import ensure_fresh_namespace, save
    base = ROOT / 'data/results' / campaign / phase / dataset / strategy
    stats_path = ROOT / os.environ['RAG_INDEX_STATS_PATH']
    if stats_path.exists() or base.exists():
        raise FileExistsError('Preserve existing index attempt; allocate a fresh campaign')
    if phase == 'smoke':
        from scripts.cold_canary_fixture import stage_fixture
        corpus, _, _ = stage_fixture(base, dataset)
    elif phase in {'pilot', 'pilot-wide'}:
        from scripts.index_pilot_fixture import stage_index_pilot
        corpus = stage_index_pilot(base, dataset, count=64 if phase == 'pilot-wide' else 32)
    else:
        corpus = ROOT / 'data' / f'{dataset}_corpus'
    async def run():
        try:
            await ensure_fresh_namespace(strategy, dataset)
            await run_indexing(str(corpus), strategy, 'default', dataset)
        finally:
            if strategy in {'prehop', 'naive'}:
                from core.neo4j_service import Neo4jService
                await Neo4jService.global_close()
    asyncio.run(run())
    stats = json.loads(stats_path.read_text())
    manifest = json.loads((corpus / 'corpus_manifest.json').read_text())
    if stats.get('status') != 'complete' or stats.get('corpus_manifest_fingerprint') != manifest['fingerprint']:
        raise RuntimeError('Index completion or corpus identity failed')
    cost = indexing_cost(stats)
    validate_cost(stats.get('amortized_indexing_cost'), cost)
    if not cost['continuous_run_eligible']:
        raise RuntimeError('Completed index has no valid measured cost')
    save(base / 'completion.json', {'status': 'index_complete', 'phase': phase, 'strategy': strategy,
        'dataset': dataset, 'run_id': run_id, 'source_count': manifest['paragraph_count'],
        'stats_path': str(stats_path.relative_to(ROOT)), 'stats_sha256': sha256_file(stats_path),
        'amortized_indexing_cost': cost, 'benchmark_admitted': False})


def supervise(plan_path):
    from core.inference_queue import OwnedQueue
    from scripts.paper_campaign import atomic_json, identity, lock, resource_lock_path, run_child, safe_environment
    from scripts.paper_detached_runtime import register_owner, session_processes, terminate_owned
    plan = json.loads(plan_path.read_text())
    base = plan_path.parent
    if (base / 'status.json').exists():
        raise RuntimeError('Index supervisor already has state; no implicit restart')
    owner = register_owner(plan_path)
    handle = lock(resource_lock_path())
    queue = OwnedQueue(base / 'queue-metrics.json')
    status = {'kind': 'index_only', 'state': 'starting', 'supervisor': identity(os.getpid()),
              'targets': {f'{row["dataset"]}/{row["strategy"]}': {'state': 'planned', 'run_id': row['run_id']}
                          for row in plan['targets']}, 'benchmark_admitted': False}
    current_target = None
    def update(fields):
        status.update(fields)
        status['updated_at'] = time.time()
        atomic_json(base / 'status.json', status)
        from core.campaign_outcomes import index_outcomes, outcome_markdown
        outcome_report = index_outcomes(status)
        atomic_json(base / 'outcomes.json', outcome_report)
        (base / 'outcomes.md').write_text(outcome_markdown(outcome_report))
    def stopped(signum, frame):
        raise RuntimeError(f'Index supervisor interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        logging.getLogger(__name__).info('Execution source: %s', source_digest())
        queue.start()
        env = safe_environment()
        update({'state': 'running'})
        # Finish bounded smoke builds before committing any full-corpus work.
        for phase in ('smoke', 'index'):
            for row in plan['targets']:
                key = f'{row["dataset"]}/{row["strategy"]}'
                current_target = status['targets'][key]
                if phase == 'index' and current_target.get('smoke_exit_code') != 0:
                    current_target['state'] = 'blocked_by_smoke_failure'
                    update({})
                    continue
                logging.getLogger(__name__).info('Execution source: %s', source_digest())
                log = base / f'{phase}-{row["dataset"]}-{row["strategy"]}'
                current_target.update(state=f'{phase}_running', started_at=time.time())
                update({'stage': f'{phase}/{key}', 'stdout_log': str(log.with_suffix('.stdout.log')),
                        'stderr_log': str(log.with_suffix('.stderr.log'))})
                command = [sys.executable, str(Path(__file__).resolve()), 'child', plan['campaign'],
                           '--strategy', row['strategy'], '--dataset', row['dataset'], '--phase', phase]
                code = run_child(command, env, log, handle, update)
                queue.drain()
                queue.persist()
                leftovers = [process for process in session_processes(owner) if process['pid'] != os.getpid()]
                if leftovers:
                    raise RuntimeError('Target left native descendants running; stopping to preserve resource isolation')
                current_target.update({f'{phase}_exit_code': code, 'state': f'{phase}_complete' if code == 0 else f'{phase}_failed',
                                       'finished_at': time.time()})
                update({'child': None})
        completed = sum(row['state'] == 'index_complete' for row in status['targets'].values())
        update({'state': 'completed' if completed == len(plan['targets']) else 'failed',
                'complete_indexes': completed, 'finished_at': time.time()})
        return 0 if completed == len(plan['targets']) else 1
    except Exception as exc:  # noqa: BLE001 - persist every supervisor failure
        if current_target is not None:
            current_target['failure'] = f'{type(exc).__name__}: {exc}'
        update({'state': 'failed', 'failure': f'{type(exc).__name__}: {exc}', 'finished_at': time.time()})
        return 1
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        remaining = terminate_owned(owner)
        queue.close()
        update({'remaining_owned_processes': remaining, 'owned_cleanup_complete': not remaining})
        handle.close()


def launch(campaign):
    from core.paper_compatibility import context_configuration
    from scripts.paper_campaign import alive, atomic_json, lock, resource_lock_path, safe_environment
    from scripts.paper_stage_runner import selected_python_environment, validate_name
    validate_name(campaign)
    selected_python_environment()
    base = ROOT / 'data/results' / campaign / 'index-supervisor'
    base.mkdir(parents=True, exist_ok=False)
    plan = {'version': 1, 'kind': 'index_only', 'campaign': campaign, 'targets': targets(campaign),
            'context': context_configuration(), 'source_sha256': source_digest()}
    plan_path = base / 'plan.json'
    atomic_json(plan_path, plan)
    handle = lock(resource_lock_path())
    handle.close()
    environment = safe_environment()
    environment['RAG_LLM_SEED'] = '42'
    with (base / 'supervisor.log').open('xb') as log:
        process = subprocess.Popen(['nohup', sys.executable, str(Path(__file__).resolve()), 'supervise', str(plan_path)],
                                   cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(150):
        path = base / 'status.json'
        if path.exists():
            status = json.loads(path.read_text())
            if status['state'] == 'failed':
                raise RuntimeError(f'Index supervisor failed: {status.get("failure")}')
            if status['state'] == 'running' and alive(status.get('supervisor')):
                receipt = {'pid': process.pid, 'plan': str(plan_path), 'status': str(path),
                           'scope': '16 primary indexes; no full benchmark admission'}
                atomic_json(base / 'launch.json', receipt)
                print(json.dumps(receipt))
                return
        if process.poll() is not None:
            raise RuntimeError(f'Index supervisor exited with {process.returncode}; inspect {base / "supervisor.log"}')
        time.sleep(.1)
    raise RuntimeError(f'Supervisor startup not confirmed; inspect {base} before retrying')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'supervise', 'child'))
    parser.add_argument('target')
    parser.add_argument('--strategy')
    parser.add_argument('--dataset', choices=('multihoprag', 'hotpotqa'))
    parser.add_argument('--phase', choices=('smoke', 'pilot', 'pilot-wide', 'index'))
    args = parser.parse_args()
    if args.action == 'child':
        from core.runtime_requirements import ensure_method_runtime
        ensure_method_runtime(args.strategy)
    from dotenv import load_dotenv
    if os.environ.get('RAG_SKIP_PROJECT_ENV') != 'true':
        load_dotenv(ROOT / '.env', override=False)
    from core.execution_profile import apply_execution_profile, execution_profile
    apply_execution_profile()
    if not execution_profile()['sha256']:
        parser.error('Select a tested RAG_EXECUTION_PROFILE before starting Python')
    os.environ.update(RAG_PAPER_MODE='true', PYTHONDONTWRITEBYTECODE='1')
    from core.strategy_registry import paper_environment_defaults
    for name, value in paper_environment_defaults().items():
        os.environ.setdefault(name, value)
    if args.action == 'child':
        if not args.strategy or not args.dataset or not args.phase:
            parser.error('child requires strategy, dataset, and phase')
        child(args.target, args.strategy, args.dataset, args.phase)
        return 0
    if args.action == 'supervise':
        return supervise(Path(args.target).resolve())
    launch(args.target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
