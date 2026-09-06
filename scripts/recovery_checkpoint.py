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
    if control != expected_root / 'control.json' or not run_id or os.environ.get('RAG_RECOVERY_TEST_PROFILE') != 'owned-checkpoint-v1':
        raise RuntimeError('Recovery hook requires an explicit owned test profile and exact control path')
    contract = json.loads(control.read_text())
    if contract.get('run_id') != run_id or contract.get('owner_pid') != os.getppid() or contract.get('owner_start') != process_start(os.getppid()):
        raise RuntimeError('Recovery hook parent ownership mismatch')
    if summary.get('status') != 'in_progress' or len(summary.get('details', [])) != 1:
        return
    result = result_path.resolve()
    if expected_root not in result.parents or summary['details'][0].get('error'):
        raise RuntimeError('Recovery test checkpoint is not a successful owned result')
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


def validate_recovery_evidence(value: dict) -> None:
    """Re-read actual checkpoint and resumed row/trace bytes; reject receipt-only proof."""
    from core.admission import sha256_file
    from core.paper_policy import canonical_query_policy
    from scripts.paper_gate_ledger import _bound_json
    _, stopped = _bound_json(value.get('interruption'))
    _, resumed = _bound_json(value.get('resume'))
    _, stale = _bound_json(value.get('stale_config'))
    run_id = stopped.get('run_id')
    if not run_id or resumed.get('run_id') != run_id or stale.get('run_id') != run_id:
        raise RuntimeError('Recovery run identities differ')
    if (stopped.get('signal'), stopped.get('exit_code')) != ('SIGTERM', -15) or not isinstance(stopped.get('pid'), int) or not stopped.get('process_start') or not stopped.get('nonce'):
        raise RuntimeError('Recovery requires a bound owned-process interruption')
    if not stopped.get('argv') or resumed.get('exit_code') != 0 or not resumed.get('argv'):
        raise RuntimeError('Recovery invocation failed or missing')
    checkpoint_path, checkpoint = _bound_json(stopped.get('checkpoint'))
    result_path, result = _bound_json(resumed.get('result'))
    def trace(reference):
        if not isinstance(reference, dict):
            raise TypeError('Recovery trace reference missing')
        path = (ROOT / reference.get('path', '')).resolve()
        if ROOT not in path.parents or sha256_file(path) != reference.get('sha256'):
            raise RuntimeError('Recovery trace reference changed')
        return path, [json.loads(line) for line in path.read_text().splitlines()]
    old_trace_path, old_traces = trace(stopped.get('trace'))
    new_trace_path, new_traces = trace(resumed.get('trace'))
    root = ROOT / 'data/results' / run_id / 'recovery'
    if any(root not in path.parents for path in (checkpoint_path, result_path, old_trace_path, new_trace_path)):
        raise RuntimeError('Recovery artifact escapes its fresh owned run')
    if stopped.get('result_sha256') != stopped['checkpoint']['sha256'] or stopped.get('trace_sha256') != stopped['trace']['sha256']:
        raise RuntimeError('Recovery copies differ from interrupted checkpoint')
    if stale.get('checkpoint') != stopped['checkpoint'] or stale.get('trace') != stopped['trace']:
        raise RuntimeError('Stale rejection did not use the preserved interrupted checkpoint')
    if stale.get('category') != 'semantic_ablation_mismatch' or stale.get('exit_code') != 1 or stale.get('field') != 'graph_hop_depth':
        raise RuntimeError('Stale rejection lacks the expected semantic mismatch')
    if stale.get('prior') != checkpoint.get('ablation', {}).get('graph_hop_depth') or stale.get('requested') != stale.get('prior', 0) + 1:
        raise RuntimeError('Stale test did not change the declared semantic setting')
    old_rows, new_rows = checkpoint.get('details'), result.get('details')
    if not isinstance(old_rows, list) or not isinstance(new_rows, list) or len(old_rows) != 1 or len(new_rows) != 2:
        raise RuntimeError('Recovery needs one complete checkpoint row and two final rows')
    if checkpoint.get('status') != 'in_progress' or result.get('status') != 'completed_unadmitted' or checkpoint.get('total_queries') != 2 or result.get('total_queries') != 2:
        raise RuntimeError('Recovery status/count is not an interrupted two-query exploratory benchmark')
    if stopped.get('completed') != 1 or stopped.get('total') != 2 or any(row.get('error') for row in new_rows):
        raise RuntimeError('Recovery has errors or incomplete checkpoint evidence')
    if checkpoint.get('benchmark_concurrency') != 1 or checkpoint.get('benchmark_checkpoint_every') != 1:
        raise RuntimeError('Recovery did not serialize checkpoint writes')
    if len(old_traces) != 1 or len(new_traces) != 2 or len({row['query_id'] for row in new_rows}) != 2:
        raise RuntimeError('Recovery trace/row count or IDs invalid')
    retained = next((row for row in new_rows if row['query_id'] == old_rows[0]['query_id']), None)
    retained_trace = next((row for row in new_traces if row['query_id'] == old_rows[0]['query_id']), None)
    if retained != old_rows[0] or retained_trace != old_traces[0]:
        raise RuntimeError('Recovery reran or changed a retained row/trace')
    meta = result.get('resume', {})
    if meta.get('retained_rows') != 1 or meta.get('resumed_rows') != 1 or meta.get('rerun_error_rows') != 0:
        raise RuntimeError('Production resume metadata does not prove one retained and one new query')
    queries = json.loads((ROOT / 'data/multihoprag_queries.json').read_text())[:2]
    expected = {row['_id']: row['query'] for row in queries}
    if {row['query_id']: row['query'] for row in new_rows} != expected:
        raise RuntimeError('Recovery is not the predetermined first two real queries')
    for field, expected_value in canonical_query_policy('naive').items():
        if result.get('ablation', {}).get(field) != expected_value or checkpoint.get('ablation', {}).get(field) != expected_value:
            raise RuntimeError('Recovery semantic policy is stale')
    # Producer ownership is content-bound to the original exclusive control/notice files.
    notice_path, notice = _bound_json(stopped.get('notice'))
    control_path, control = _bound_json(stopped.get('control'))
    if notice_path != root / 'checkpoint_ready.json' or control_path != root / 'control.json' or Path(str(notice.get('result_path'))) != result_path or Path(str(notice.get('trace_path'))) != new_trace_path:
        raise RuntimeError('Recovery control/notice paths differ from their actual owned artifacts')
    for field in ('run_id', 'pid', 'process_start', 'nonce', 'result_sha256', 'trace_sha256', 'completed', 'total'):
        if stopped.get(field) != notice.get(field):
            raise RuntimeError('Interruption receipt differs from actual checkpoint notification')
    if control.get('run_id') != run_id or control.get('nonce') != notice.get('nonce') or not control.get('owner_pid') or not control.get('owner_start'):
        raise RuntimeError('Recovery parent control identity missing')
    from cli.benchmark import _query_ids_sha256, _query_records_sha256, _resume_benchmark_rows
    metadata = {field: result.get(field) for field in ('strategy', 'corpus_tag', 'dataset', 'evaluation_scope',
        'evaluated_queries_count', 'evaluated_query_ids_sha256', 'evaluated_query_records_sha256',
        'corpus_manifest_fingerprint', 'models', 'ablation')}
    retained_rows, _ = _resume_benchmark_rows(checkpoint_path, queries, metadata, judge_enabled=False)
    stale_metadata = {**metadata, 'ablation': {**metadata['ablation'], 'graph_hop_depth': stale['requested']}}
    try:
        _resume_benchmark_rows(checkpoint_path, queries, stale_metadata, judge_enabled=False)
    except RuntimeError as exc:
        if 'Resume metadata mismatch for ablation:' not in str(exc):
            raise
    else:
        raise RuntimeError('Stale checkpoint unexpectedly passed the actual resume parser')
    if len(retained_rows) != 1:
        raise RuntimeError('Production resume validator did not retain exactly one query')
    if result.get('strategy') != 'naive' or result.get('corpus_tag') != 'multihoprag' or result.get('judge_enabled') is not False:
        raise RuntimeError('Recovery used a different method/dataset/judge contract')
    if result.get('evaluated_query_ids_sha256') != _query_ids_sha256(queries) or result.get('evaluated_query_records_sha256') != _query_records_sha256(queries):
        raise RuntimeError('Recovery query identity differs from the exact selected real records')
    for position, (row, trace_row, query) in enumerate(zip(new_rows, new_traces, queries, strict=True), start=1):
        if row.get('idx') != position or trace_row.get('idx') != position or row.get('query_id') != query['_id'] or trace_row.get('query_id') != query['_id'] or trace_row.get('query') != query['query']:
            raise RuntimeError('Recovery final row/trace identity or order mismatch')
        if not isinstance(row.get('answer'), str) or not row['answer'].strip() or not row.get('retrieved_sources') or not trace_row.get('interaction_trace'):
            raise RuntimeError('Recovery final query lacks actual answer/evidence/trace')
    from core.admission import current_corpus_identity
    from core.paper_policy import configure_target_environment
    from core.runtime_requirements import runtime_identity
    from scripts.check_paper_runtime import check
    from scripts.verify_index_policy import verify
    original = os.environ.copy()
    try:
        configure_target_environment('naive', 'multihoprag', run_id)
        _, receipt = _bound_json(value.get('receipt'))
        stats_path, stats = _bound_json(receipt.get('index_stats'))
        verify(stats_path, 'naive', 'multihoprag', run_id)
        check('naive', 'multihoprag')
        corpus = current_corpus_identity('multihoprag')
        if receipt.get('runtime_identity') != runtime_identity('naive'):
            raise RuntimeError('Recovery runtime identity is stale')
        for payload in (checkpoint, result):
            snapshot = payload.get('active_index_snapshot', {})
            if snapshot.get('status') != 'matched' or snapshot.get('source_count') != corpus['paragraph_count']:
                raise RuntimeError('Recovery has no verified complete native corpus snapshot')
            if payload.get('corpus_manifest_fingerprint') != corpus['fingerprint'] or payload.get('index_manifest_fingerprint') != corpus['fingerprint']:
                raise RuntimeError('Recovery full-corpus identity is stale')
            if payload.get('index_manifest_stats_sha256') != receipt['index_stats']['sha256'] or payload.get('index_provenance', {}).get('run_id') != run_id or payload.get('index_provenance', {}).get('policy_sha256') != stats.get('index_policy_sha256'):
                raise RuntimeError('Recovery actual native index binding differs')
    finally:
        os.environ.clear()
        os.environ.update(original)
