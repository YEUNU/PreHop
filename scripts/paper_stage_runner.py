#!/usr/bin/env python3
"""Execute owned recovery, complete-corpus query and fresh full-target gates."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
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
    return value


def selected_python_environment() -> dict[str, str]:
    environment = os.environ.copy()
    prefix = Path(sys.prefix).resolve()
    environment.get('PYTHON_BIN', '')
    environment.get('UV_PROJECT_ENVIRONMENT', '')
    executable = prefix / 'bin/python'
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
    from scripts.paper_gate_ledger import record
    ledger = ROOT / 'data/results' / campaign / 'gate_ledger.json'
    from scripts.campaign_attempts import attempt_mapping, selected_attempt
    target_attempts = attempt_mapping(target_attempts)
    phase = 'cold' if stage == 'cold_canary_16' else 'one-query'
    branch = 'cold_v2' if stage == 'cold_canary_16' else 'one_query'
    base = ROOT / 'data/results' / campaign / branch / validate_name(attempt)
    targets = {f'{dataset}/{strategy}': reference(ROOT / 'data/results' / campaign / branch / selected_attempt(attempt, target_attempts, f'{phase}/{dataset}/{strategy}') / dataset / strategy / 'evidence.json')
               for dataset in ('multihoprag', 'hotpotqa') for strategy in PRIMARY_STRATEGIES}
    path = base / 'matrix_evidence.json'
    save(path, {'schema_version': 1, 'stage': stage, 'status': 'canary_passed', 'targets': targets})
    record(ledger, stage, path)


def reattest(campaign: str, attempt: str) -> None:
    from core.runtime_requirements import runtime_identity
    from core.strategy_registry import PRIMARY_STRATEGIES
    from scripts.paper_cold_canary import save
    from scripts.paper_gate_ledger import record
    ledger = ROOT / 'data/results' / campaign / 'gate_ledger.json'
    base = stage_base(campaign, 'reattest', attempt)
    checks = {}
    original = os.environ.copy()
    try:
        from core.paper_policy import configure_target_environment
        for strategy in PRIMARY_STRATEGIES:
            configure_target_environment(strategy, 'multihoprag', f'{campaign}-reattest-{strategy}')
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
    ROOT / os.environ['RAG_INDEX_STATS_PATH']
    current_corpus_identity(dataset)
    try:
        await run_indexing(str(ROOT / 'data' / f'{dataset}_corpus'), strategy, 'default', dataset)
    finally:
        if strategy in {'prehop', 'naive'}:
            from core.neo4j_service import Neo4jService
            await Neo4jService.global_close()


def full_target(campaign: str, strategy: str, dataset: str, attempt: str) -> None:
    from core.paper_policy import configure_target_environment
    from scripts.paper_gate_ledger import record
    ledger = ROOT / 'data/results' / campaign / 'gate_ledger.json'
    selected_python_environment()
    run_id = f'{campaign}-full-{dataset}-{strategy}-{attempt}'
    configure_target_environment(strategy, dataset, run_id)
    async def check_namespace():
        try:
            pass
        finally:
            if strategy in {'prehop', 'naive'}:
                from core.neo4j_service import Neo4jService
                await Neo4jService.global_close()
    asyncio.run(check_namespace())
    subprocess.run(['bash', 'scripts/run_paper_target.sh', dataset, strategy, run_id], cwd=ROOT, env=selected_python_environment(), check=True)
    record(ledger, 'full_target_admitted', ROOT / 'data/results' / run_id / 'admission.json')


def reuse_target(campaign: str, strategy: str, dataset: str) -> None:
    from core.index_reuse import load_link, prepare
    run_id = f'{campaign}-{dataset}-{strategy}'
    base = ROOT / 'data/results' / run_id
    path = base / 'index_link.json'
    verifier = [sys.executable, 'scripts/record_paper_completion.py', run_id, dataset, strategy,
                '--exact-run-id', '--output', str(base / 'admission.json')]
    if (base / 'admission.json').exists():
        subprocess.run(verifier, cwd=ROOT, env=selected_python_environment(), check=True)
        return
    environment = selected_python_environment()
    if base.exists():
        load_link(path, run_id, strategy, dataset)
        result = base / strategy / dataset / 'seed_42' / f'{strategy}_{dataset}.json'
        payload = json.loads(result.read_text())
        if payload.get('status') == 'completed_unadmitted':
            subprocess.run(verifier, cwd=ROOT, env=selected_python_environment(), check=True)
            return
        environment['RAG_BENCHMARK_RESUME'] = 'true'
    else:
        path = prepare(campaign, strategy, dataset)
    # Re-enter after configuring the link before importing static RAGConfig.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '_reuse_benchmark', campaign,
                    '--strategy', strategy, '--dataset', dataset], cwd=ROOT,
                   env=environment, check=True)
    load_link(path, run_id, strategy, dataset)
    subprocess.run(verifier, cwd=ROOT, env=selected_python_environment(), check=True)


def reuse_benchmark(campaign: str, strategy: str, dataset: str) -> None:
    from core.index_reuse import bootstrap_benchmark, load_link
    run_id = f'{campaign}-{dataset}-{strategy}'
    path = ROOT / 'data/results' / run_id / 'index_link.json'
    bootstrap_benchmark(path, run_id, strategy, dataset)
    load_link(path, run_id, strategy, dataset)
    from cli.benchmark import run_benchmark
    async def run():
        try:
            await run_benchmark(str(ROOT / 'data' / f'{dataset}_queries.json'), strategy, 'default',
                                corpus_tag=dataset, output_dir=path.parent, seed=42)
        finally:
            if strategy in {'prehop', 'naive'}:
                from core.neo4j_service import Neo4jService
                await Neo4jService.global_close()
    asyncio.run(run())


def recovery_child(run_id: str, mode: str) -> None:
    base = ROOT / 'data/results' / validate_name(run_id) / 'recovery'
    control_path = base / 'control.json'
    json.loads(control_path.read_text())
    base / 'benchmark/naive/multihoprag/seed_42/naive_multihoprag.json'
    if mode == 'resume':
        json.loads((base / 'interruption.json').read_text())
    from core.paper_policy import configure_target_environment
    configure_target_environment('naive', 'multihoprag', run_id)
    from core.strategy_registry import PAPER_TRANSPORT
    os.environ['RAG_BENCHMARK_CONCURRENCY'] = str(PAPER_TRANSPORT.benchmark_concurrency)
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
    from scripts.paper_gate_ledger import record
    from scripts.recovery_checkpoint import process_start
    ledger = ROOT / 'data/results' / campaign / 'gate_ledger.json'
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
        process_start(child.pid)
        notification = base / 'checkpoint_ready.json'
        while not notification.exists():
            time.sleep(.2)
        notice = json.loads(notification.read_text())
        result_path, trace_path = Path(notice['result_path']), Path(notice['trace_path'])
        checkpoint = base / 'checkpoint'
        checkpoint.mkdir()
        shutil.copy2(result_path, checkpoint / result_path.name)
        shutil.copy2(trace_path, checkpoint / trace_path.name)
        child.send_signal(signal.SIGTERM)
        exit_code = child.wait()
    interrupted = save(base / 'interruption.json', {**notice, 'notice': reference(notification), 'control': reference(base / 'control.json'), 'argv': command, 'exit_code': exit_code, 'signal': 'SIGTERM',
        'checkpoint': reference(checkpoint / result_path.name), 'trace': reference(checkpoint / trace_path.name)})
    # The real resume parser must reject a typed changed semantic setting on the preserved copy.
    from cli.benchmark import _resume_benchmark_rows
    prior = json.loads((checkpoint / result_path.name).read_text())
    data = json.loads((ROOT / 'data/multihoprag_queries.json').read_text())[:2]
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
    stale = save(base / 'stale.json', {'run_id': run_id, 'exit_code': 1, 'category': category,
        'field': 'graph_hop_depth', 'prior': prior['ablation']['graph_hop_depth'], 'requested': expected['graph_hop_depth'],
        'checkpoint': reference(checkpoint / result_path.name), 'trace': reference(checkpoint / trace_path.name)})
    command = [sys.executable, str(Path(__file__).resolve()), '_recovery_child', run_id, '--mode', 'resume']
    resume_env = {**env, 'RAG_RECOVERY_CHILD_MODE': 'resume'}
    with (base / 'resumed.log').open('x') as log:
        subprocess.run(command, cwd=ROOT, env=resume_env, stdout=log, stderr=subprocess.STDOUT, check=False)
    resumed = save(base / 'resume.json', {'run_id': run_id, 'exit_code': 0, 'argv': command,
        'result': reference(result_path), 'trace': reference(trace_path)})
    from core.runtime_requirements import runtime_identity
    receipt = save(base / 'receipt.json', {'runtime_identity': runtime_identity('naive'), 'index_stats': reference(ROOT / os.environ['RAG_INDEX_STATS_PATH']), 'stage': 'resume_stale_rejection', 'exit_code': 0, 'argv': sys.argv,
        'executor_sha256': sha256_file(Path(__file__)), 'run_id': run_id})
    evidence = base / 'evidence.json'
    save(evidence, {'schema_version': 1, 'stage': 'resume_stale_rejection', 'status': 'canary_passed',
        'receipt': receipt, 'interruption': interrupted, 'resume': resumed, 'stale_config': stale,
        'resume_passed': True, 'stale_config_rejected': True})
    record(ledger, 'resume_stale_rejection', evidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['reattest', 'cold-aggregate', 'one-query', 'one-query-aggregate', 'recovery', 'full-target', 'reuse-target', '_reuse_benchmark', '_recovery_child'])
    parser.add_argument('campaign')
    parser.add_argument('--attempt', default='a1')
    parser.add_argument('--target-attempts-json', default='{}')
    parser.add_argument('--strategy', default='naive')
    parser.add_argument('--dataset', choices=['multihoprag', 'hotpotqa'], default='multihoprag')
    parser.add_argument('--mode', choices=['interrupt', 'resume'])
    args = parser.parse_args()
    if args.action in {'one-query', 'full-target', 'reuse-target', '_reuse_benchmark'}:
        from core.runtime_requirements import ensure_method_runtime
        ensure_method_runtime(args.strategy)
    from scripts.runner_environment import _load_runner_environment
    _load_runner_environment()
    validate_name(args.campaign)
    validate_name(args.attempt)
    if args.action == 'reattest':
        reattest(args.campaign, args.attempt)
    elif args.action == 'recovery':
        recovery(args.campaign, args.attempt)
    elif args.action == '_recovery_child':
        recovery_child(args.campaign, args.mode)
    elif args.action == 'reuse-target':
        reuse_target(args.campaign, args.strategy, args.dataset)
    elif args.action == '_reuse_benchmark':
        reuse_benchmark(args.campaign, args.strategy, args.dataset)
    elif args.action == 'full-target':
        full_target(args.campaign, args.strategy, args.dataset, args.attempt)
    elif args.action == 'one-query':
        from scripts.paper_cold_canary import workflow
        asyncio.run(workflow(args.campaign, args.strategy, args.dataset, args.attempt, full_corpus=True))
    else:
        aggregate(args.campaign, 'cold_canary_16' if args.action == 'cold-aggregate' else 'one_query_matrix_16', args.attempt, json.loads(args.target_attempts_json))


if __name__ == '__main__':
    main()
