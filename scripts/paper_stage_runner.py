#!/usr/bin/env python3
"""Execute owned recovery, complete-corpus query and fresh full-target gates."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_name(value: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value):
        raise ValueError('Unsafe or empty campaign/run/attempt identifier')
    return value


def selected_python_environment() -> dict[str, str]:
    environment = os.environ.copy()
    prefix = Path(sys.prefix).resolve()
    selected = environment.get('PYTHON_BIN', '')
    configured = environment.get('UV_PROJECT_ENVIRONMENT', '')
    if selected and Path(os.path.abspath(selected)).parent.parent.resolve() != prefix:
        raise RuntimeError('PYTHON_BIN differs from the running main environment')
    if configured and Path(configured).resolve() != prefix:
        raise RuntimeError('UV_PROJECT_ENVIRONMENT differs from the running main environment')
    executable = prefix / 'bin/python'
    if not executable.is_file():
        raise RuntimeError('Selected main environment has no Python launcher')
    environment.update(PYTHON_BIN=str(executable), UV_PROJECT_ENVIRONMENT=str(prefix))
    return environment


def reference(path: Path) -> dict:
    from core.admission import sha256_file
    return {'path': str(path.resolve().relative_to(ROOT)), 'sha256': sha256_file(path)}


def stage_base(campaign: str, name: str, attempt: str) -> Path:
    path = ROOT / 'data/results' / validate_name(campaign) / name / validate_name(attempt)
    path.mkdir(parents=True, exist_ok=False)
    return path


def aggregate(campaign: str, stage: str, attempt: str, target_attempts: dict | None = None) -> None:
    from core.strategy_registry import PRIMARY_STRATEGIES
    from scripts.paper_cold_canary import save
    from scripts.paper_gate_ledger import ready, record
    ledger = ROOT / 'data/results' / campaign / 'gate_ledger.json'
    ready(ledger, stage)
    from scripts.campaign_attempts import selected_attempt, validated_attempts
    target_attempts = validated_attempts(target_attempts)
    phase = 'cold' if stage == 'cold_canary_16' else 'one-query'
    branch = 'cold_v2' if stage == 'cold_canary_16' else 'one_query'
    base = ROOT / 'data/results' / campaign / branch / validate_name(attempt)
    targets = {f'{dataset}/{strategy}': reference(ROOT / 'data/results' / campaign / branch / selected_attempt(attempt, target_attempts, f'{phase}/{dataset}/{strategy}') / dataset / strategy / 'evidence.json')
               for dataset in ('multihoprag', 'musique') for strategy in PRIMARY_STRATEGIES}
    path = base / 'matrix_evidence.json'
    save(path, {'schema_version': 1, 'stage': stage, 'status': 'canary_passed', 'targets': targets})
    record(ledger, stage, path)


def reattest(campaign: str, attempt: str) -> None:
    from core.runtime_requirements import runtime_identity
    from core.strategy_registry import PRIMARY_STRATEGIES
    from scripts.check_paper_runtime import check
    from scripts.paper_cold_canary import save
    from scripts.paper_gate_ledger import ready, record
    ledger = ROOT / 'data/results' / campaign / 'gate_ledger.json'
    ready(ledger, 'runtime_setup')
    base = stage_base(campaign, 'reattest', attempt)
    checks = {}
    original = os.environ.copy()
    try:
        from core.paper_policy import configure_target_environment
        for strategy in PRIMARY_STRATEGIES:
            configure_target_environment(strategy, 'multihoprag', f'{campaign}-reattest-{strategy}')
            check(strategy, 'multihoprag')
            checks[strategy] = 'passed'
    finally:
        os.environ.clear()
        os.environ.update(original)
    receipt = save(base / 'receipt.json', {'stage': 'runtime_setup', 'exit_code': 0, 'argv': sys.argv,
        'mode': 'read_only_reattest', 'checks': checks,
        'runtime_identities': {strategy: runtime_identity(strategy) for strategy in PRIMARY_STRATEGIES}})
    path = base / 'evidence.json'
    save(path, {'schema_version': 1, 'stage': 'runtime_setup', 'status': 'canary_passed', 'receipt': receipt})
    record(ledger, 'runtime_setup', path)


async def fresh_index(strategy: str, dataset: str) -> None:
    from cli.index import run_indexing
    from core.admission import current_corpus_identity
    from scripts.paper_cold_canary import ensure_fresh_namespace
    raw = ROOT / os.environ['RAG_INDEX_STATS_PATH']
    if raw.exists():
        raise FileExistsError('Index statistics already exist')
    current_corpus_identity(dataset)
    await ensure_fresh_namespace(strategy, dataset)
    try:
        await run_indexing(str(ROOT / 'data' / f'{dataset}_corpus'), strategy, 'default', dataset)
    finally:
        if strategy in {'prehop', 'naive'}:
            from core.neo4j_service import Neo4jService
            await Neo4jService.global_close()


def full_target(campaign: str, strategy: str, dataset: str, attempt: str) -> None:
    from core.paper_policy import configure_target_environment
    from scripts.paper_cold_canary import ensure_fresh_namespace
    from scripts.paper_gate_ledger import ready, record
    ledger = ROOT / 'data/results' / campaign / 'gate_ledger.json'
    selected_python_environment()
    ready(ledger, 'full_target_admitted')
    run_id = f'{campaign}-full-{dataset}-{strategy}-{attempt}'
    configure_target_environment(strategy, dataset, run_id)
    if (ROOT / 'data/results' / run_id).exists() or (ROOT / os.environ['RAG_INDEX_STATS_PATH']).exists():
        raise FileExistsError('Full gate requires a new result and index namespace')
    async def check_namespace():
        try:
            await ensure_fresh_namespace(strategy, dataset)
        finally:
            if strategy in {'prehop', 'naive'}:
                from core.neo4j_service import Neo4jService
                await Neo4jService.global_close()
    asyncio.run(check_namespace())
    subprocess.run(['bash', 'scripts/run_paper_target.sh', dataset, strategy, run_id], cwd=ROOT, env=selected_python_environment(), check=True)
    ready(ledger, 'full_target_admitted')
    record(ledger, 'full_target_admitted', ROOT / 'data/results' / run_id / 'admission.json')


def recovery_child(run_id: str, mode: str) -> None:
    from scripts.recovery_checkpoint import process_start
    if mode not in {'interrupt', 'resume'}:
        raise RuntimeError('Owned recovery child requires an explicit mode')
    base = ROOT / 'data/results' / validate_name(run_id) / 'recovery'
    control_path = base / 'control.json'
    if os.environ.get('RAG_RECOVERY_TEST_CONTROL') != str(control_path) or os.environ.get('RAG_RECOVERY_TEST_PROFILE') != 'owned-checkpoint-v1':
        raise RuntimeError('Recovery child requires an explicit owned control profile')
    control = json.loads(control_path.read_text())
    if control.get('run_id') != run_id or control.get('owner_pid') != os.getppid() or control.get('owner_start') != process_start(os.getppid()) or control.get('nonce') != os.environ.get('RAG_RECOVERY_CHILD_NONCE') or mode != os.environ.get('RAG_RECOVERY_CHILD_MODE'):
        raise RuntimeError('Recovery child is not authorized by its live owning parent')
    result_path = base / 'benchmark/naive/multihoprag/seed_42/naive_multihoprag.json'
    if mode == 'interrupt' and (base / 'benchmark').exists():
        raise FileExistsError('First recovery child requires fresh benchmark paths')
    if mode == 'resume':
        interruption = json.loads((base / 'interruption.json').read_text())
        if interruption.get('run_id') != run_id or interruption.get('nonce') != control['nonce'] or interruption.get('exit_code') != -signal.SIGTERM or json.loads(result_path.read_text()).get('status') != 'in_progress':
            raise RuntimeError('Resume child requires its owned interrupted checkpoint')
    from core.paper_policy import configure_target_environment
    configure_target_environment('naive', 'multihoprag', run_id)
    os.environ['RAG_BENCHMARK_CONCURRENCY'] = '1'
    os.environ['RAG_BENCHMARK_CHECKPOINT_EVERY'] = '1'
    if mode == 'resume':
        os.environ['RAG_BENCHMARK_RESUME'] = 'true'
        os.environ.pop('RAG_RECOVERY_TEST_CONTROL', None)
        os.environ.pop('RAG_RECOVERY_TEST_PROFILE', None)
    from cli.benchmark import run_benchmark
    async def run():
        try:
            await run_benchmark(str(ROOT / 'data/multihoprag_queries.json'), 'naive', 'default',
                                corpus_tag='multihoprag', output_dir=base / 'benchmark', limit=2, seed=42)
        finally:
            from core.neo4j_service import Neo4jService
            await Neo4jService.global_close()
    asyncio.run(run())


def recovery(campaign: str, attempt: str) -> None:
    from core.admission import sha256_file
    from core.paper_policy import configure_target_environment
    from scripts.paper_cold_canary import save
    from scripts.paper_gate_ledger import ready, record
    from scripts.recovery_checkpoint import process_start
    ledger = ROOT / 'data/results' / campaign / 'gate_ledger.json'
    ready(ledger, 'resume_stale_rejection')
    run_id = f'{campaign}-recovery-multihoprag-naive-{attempt}'
    configure_target_environment('naive', 'multihoprag', run_id)
    base = ROOT / 'data/results' / run_id / 'recovery'
    base.mkdir(parents=True, exist_ok=False)
    asyncio.run(fresh_index('naive', 'multihoprag'))
    nonce = secrets.token_hex(24)
    save(base / 'control.json', {'run_id': run_id, 'owner_pid': os.getpid(), 'owner_start': process_start(os.getpid()), 'nonce': nonce})
    env = os.environ.copy()
    env.update(RAG_RECOVERY_TEST_PROFILE='owned-checkpoint-v1', RAG_RECOVERY_TEST_CONTROL=str(base / 'control.json'),
               RAG_RECOVERY_CHILD_NONCE=nonce, RAG_RECOVERY_CHILD_MODE='interrupt')
    command = [sys.executable, str(Path(__file__).resolve()), '_recovery_child', run_id, '--mode', 'interrupt']
    log_path = base / 'interrupted.log'
    with log_path.open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        child_start = process_start(child.pid)
        notification = base / 'checkpoint_ready.json'
        while not notification.exists():
            if child.poll() is not None:
                raise RuntimeError(f'Owned recovery child exited before a complete checkpoint: {child.returncode}')
            time.sleep(.2)
        notice = json.loads(notification.read_text())
        if notice.get('pid') != child.pid or notice.get('process_start') != child_start or notice.get('nonce') != nonce or notice.get('run_id') != run_id:
            raise RuntimeError('Recovery checkpoint child ownership mismatch; no signal sent')
        if child.poll() is not None or process_start(child.pid) != child_start:
            raise RuntimeError('Owned child changed before signal; no signal sent')
        result_path, trace_path = Path(notice['result_path']), Path(notice['trace_path'])
        if base not in result_path.parents or base not in trace_path.parents:
            raise RuntimeError('Checkpoint paths escaped owned run; no signal sent')
        if sha256_file(result_path) != notice['result_sha256'] or sha256_file(trace_path) != notice['trace_sha256']:
            raise RuntimeError('Checkpoint files changed while paused; no signal sent')
        checkpoint = base / 'checkpoint'
        checkpoint.mkdir()
        shutil.copy2(result_path, checkpoint / result_path.name)
        shutil.copy2(trace_path, checkpoint / trace_path.name)
        child.send_signal(signal.SIGTERM)
        exit_code = child.wait()
    if exit_code != -signal.SIGTERM:
        raise RuntimeError('Owned interruption did not terminate with requested SIGTERM')
    interrupted = save(base / 'interruption.json', {**notice, 'notice': reference(notification), 'control': reference(base / 'control.json'), 'argv': command, 'exit_code': exit_code, 'signal': 'SIGTERM',
        'checkpoint': reference(checkpoint / result_path.name), 'trace': reference(checkpoint / trace_path.name)})
    # The real resume parser must reject a typed changed semantic setting on the preserved copy.
    from cli.benchmark import _resume_benchmark_rows
    prior = json.loads((checkpoint / result_path.name).read_text())
    data = json.loads((ROOT / 'data/multihoprag_queries.json').read_text())[:2]
    if sha256_file(checkpoint / result_path.name) != notice['result_sha256'] or sha256_file(checkpoint / trace_path.name) != notice['trace_sha256']:
        raise RuntimeError('Preserved checkpoint copy differs from paused native output')
    expected = dict(prior['ablation'])
    expected['graph_hop_depth'] = int(expected['graph_hop_depth']) + 1
    try:
        _resume_benchmark_rows(checkpoint / result_path.name, data, {'ablation': expected}, judge_enabled=False)
    except RuntimeError as exc:
        if 'Resume metadata mismatch for ablation:' not in str(exc):
            raise
        category = 'semantic_ablation_mismatch'
    else:
        raise RuntimeError('Stale semantic checkpoint unexpectedly accepted')
    if sha256_file(checkpoint / result_path.name) != notice['result_sha256'] or sha256_file(checkpoint / trace_path.name) != notice['trace_sha256']:
        raise RuntimeError('Stale validation mutated preserved checkpoint bytes')
    stale = save(base / 'stale.json', {'run_id': run_id, 'exit_code': 1, 'category': category,
        'field': 'graph_hop_depth', 'prior': prior['ablation']['graph_hop_depth'], 'requested': expected['graph_hop_depth'],
        'checkpoint': reference(checkpoint / result_path.name), 'trace': reference(checkpoint / trace_path.name)})
    command = [sys.executable, str(Path(__file__).resolve()), '_recovery_child', run_id, '--mode', 'resume']
    resume_env = {**env, 'RAG_RECOVERY_CHILD_MODE': 'resume'}
    with (base / 'resumed.log').open('x') as log:
        completed = subprocess.run(command, cwd=ROOT, env=resume_env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError('Actual production benchmark resume failed; outputs preserved')
    resumed = save(base / 'resume.json', {'run_id': run_id, 'exit_code': 0, 'argv': command,
        'result': reference(result_path), 'trace': reference(trace_path)})
    from core.runtime_requirements import runtime_identity
    receipt = save(base / 'receipt.json', {'runtime_identity': runtime_identity('naive'), 'index_stats': reference(ROOT / os.environ['RAG_INDEX_STATS_PATH']), 'stage': 'resume_stale_rejection', 'exit_code': 0, 'argv': sys.argv,
        'executor_sha256': sha256_file(Path(__file__)), 'run_id': run_id})
    evidence = base / 'evidence.json'
    save(evidence, {'schema_version': 1, 'stage': 'resume_stale_rejection', 'status': 'canary_passed',
        'receipt': receipt, 'interruption': interrupted, 'resume': resumed, 'stale_config': stale,
        'resume_passed': True, 'stale_config_rejected': True})
    ready(ledger, 'resume_stale_rejection')
    record(ledger, 'resume_stale_rejection', evidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['reattest', 'cold-aggregate', 'one-query', 'one-query-aggregate', 'recovery', 'full-target', '_recovery_child'])
    parser.add_argument('campaign')
    parser.add_argument('--attempt', default='a1')
    parser.add_argument('--target-attempts-json', default='{}')
    parser.add_argument('--strategy', default='naive')
    parser.add_argument('--dataset', choices=['multihoprag', 'musique'], default='multihoprag')
    parser.add_argument('--mode', choices=['interrupt', 'resume'])
    args = parser.parse_args()
    from scripts.check_paper_runtime import _load_runner_environment
    _load_runner_environment()
    validate_name(args.campaign)
    validate_name(args.attempt)
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError('Run the stage executor from the repository root')
    from core.strategy_registry import PRIMARY_STRATEGIES
    if args.strategy not in PRIMARY_STRATEGIES:
        raise ValueError('Unknown primary strategy')
    if args.action == 'reattest':
        reattest(args.campaign, args.attempt)
    elif args.action == 'recovery':
        recovery(args.campaign, args.attempt)
    elif args.action == '_recovery_child':
        recovery_child(args.campaign, args.mode)
    elif args.action == 'full-target':
        full_target(args.campaign, args.strategy, args.dataset, args.attempt)
    elif args.action == 'one-query':
        from scripts.paper_cold_canary import workflow
        asyncio.run(workflow(args.campaign, args.strategy, args.dataset, args.attempt, full_corpus=True))
    else:
        aggregate(args.campaign, 'cold_canary_16' if args.action == 'cold-aggregate' else 'one_query_matrix_16', args.attempt, json.loads(args.target_attempts_json))


if __name__ == '__main__':
    main()
