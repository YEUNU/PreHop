"""Explicit complete-index reuse; source evidence and native files stay immutable."""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

from core.admission import current_corpus_identity, identity_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def safe_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value):
        raise ValueError('Invalid reuse identity')
    return value


def local_path(value: str) -> Path:
    path = ROOT / value
    if path.is_symlink() or ROOT not in path.resolve().parents or any(parent.is_symlink() for parent in path.parents if parent != ROOT):
        raise RuntimeError('Reuse path must remain inside the execution repository')
    return path.resolve()


def ref(path: Path) -> dict:
    return {'path': str(path.resolve().relative_to(ROOT)), 'sha256': sha256_file(path)}


def bound(value: dict) -> tuple[Path, dict]:
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        raise RuntimeError('Incomplete reuse content reference')
    path = local_path(value['path'])
    if sha256_file(path) != value['sha256']:
        raise RuntimeError('Reuse source content changed')
    return path, json.loads(path.read_text())


def inventory(path: Path) -> list[dict]:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError('Reuse output must be a real directory')
    rows = []
    for candidate in sorted(path.rglob('*')):
        if candidate.is_symlink():
            raise RuntimeError('Reuse output cannot contain symlinks')
        if candidate.is_file():
            rows.append({'path': str(candidate.relative_to(path)), 'size': candidate.stat().st_size,
                         'sha256': sha256_file(candidate)})
    if not rows:
        raise RuntimeError('Reuse output is empty')
    return rows


def source_evidence(value: dict, strategy: str, dataset: str) -> tuple[dict, dict]:
    """Reapply the production canary gate to the complete original index."""
    from scripts.paper_gate_ledger import _validate_canary_artifacts
    _, evidence = bound(value)
    if any(evidence.get(k) != v for k, v in {'stage': 'one_query_matrix_16', 'status': 'canary_passed',
            'strategy': strategy, 'dataset': dataset, 'exit_code': 0}.items()):
        raise RuntimeError('Reuse requires exact complete-corpus one-query evidence')
    index_path, index = bound(evidence['index'])
    _, query = bound(evidence['query'])
    _, admission = bound(evidence['admission'])
    _, invocation = bound(evidence['invocation'])
    if invocation.get('exit_code') != 0 or invocation.get('target') != f'{dataset}/{strategy}':
        raise RuntimeError('Source invocation is invalid')
    if admission.get('status') != 'canary_passed' or admission.get('errors') != []:
        raise RuntimeError('Source canary was not admitted')
    for row in (index, query, admission):
        if row.get('run_id') != index.get('run_id') or row.get('strategy') != strategy or row.get('dataset') != dataset:
            raise RuntimeError('Source canary identities differ')
    if admission.get('index_sha256') != evidence['index']['sha256'] or admission.get('query_sha256') != evidence['query']['sha256']:
        raise RuntimeError('Source canary admission binding differs')
    if index.get('fresh_index') is not True or index.get('status') != 'complete' or query.get('query_count') != 1:
        raise RuntimeError('Reuse source is not a complete indexed canary')
    _validate_canary_artifacts('one_query_matrix_16', strategy, dataset, index_path, query, index)
    raw_path, raw = bound(index['native_index_stats'])
    expected = local_path(f'data/index_stats/{strategy}_{dataset}_{safe_name(index["run_id"])}.json')
    if raw_path != expected:
        raise RuntimeError('Reuse source stats path is not exact')
    for field in ('status', 'strategy', 'corpus_tag', 'run_id', 'index_policy', 'index_policy_sha256',
                  'corpus_manifest_fingerprint', 'timing_seconds'):
        if raw.get(field) != index.get(field):
            raise RuntimeError('Native index stats differ from source evidence')
    import math
    timing = raw.get('timing_seconds')
    if not isinstance(timing, dict) or 'total_elapsed_seconds' not in timing or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in timing.values()):
        raise RuntimeError('Original successful index timing must be finite and complete')
    return index, raw


def configure(link: dict) -> None:
    from core.paper_policy import configure_target_environment
    from core.strategy_registry import get_strategy
    configure_target_environment(link['strategy'], link['dataset'], link['source_run_id'])
    os.environ.update(RAG_BENCHMARK_TIMESTAMP=link['target_run_id'], RAG_JUDGE_ENABLED='false',
                      RAG_JUDGE_BATCH='false', RAG_CHUNK_CACHE='off', RAG_EMBEDDING_CACHE='off')
    spec = get_strategy(link['strategy'])
    if spec.output_env:
        os.environ[spec.output_env] = str(local_path(link['clone']['query_output_root']))


def bootstrap_benchmark(link_path: Path, target: str, strategy: str, dataset: str) -> None:
    """Configure public runtime inputs before validation can import RAGConfig."""
    path = local_path(str(link_path))
    if path != local_path(f'data/results/{safe_name(target)}/index_link.json'):
        raise RuntimeError('Unexpected benchmark reuse link path')
    link = json.loads(path.read_text())
    if any(link.get(k) != v for k, v in {'target_run_id': target, 'strategy': strategy, 'dataset': dataset}.items()):
        raise RuntimeError('Unexpected benchmark reuse link identity')
    safe_name(link['source_run_id'])
    configure(link)
    os.environ['RAG_INDEX_REUSE_LINK'] = str(path)


def validate(link_path: Path, target: str, strategy: str, dataset: str, *, pristine_clone: bool = False) -> dict:
    from core.paper_compatibility import target_configuration
    from core.paper_policy import configure_target_environment
    from core.strategy_registry import get_strategy
    link_path = local_path(str(link_path))
    if link_path != local_path(f'data/results/{safe_name(target)}/index_link.json'):
        raise RuntimeError('Reuse link must be stored under its exact fresh result target')
    link = json.loads(link_path.read_text())
    for field, expected in {'version': 1, 'target_run_id': target, 'strategy': strategy, 'dataset': dataset}.items():
        if link.get(field) != expected:
            raise RuntimeError('Reuse link target identity differs')
    gate_snapshot, source_ledger = bound(link['gate_ledger'])
    if gate_snapshot != link_path.parent / 'source_gate_ledger.json':
        raise RuntimeError('Reuse gate snapshot must belong to the exact fresh result target')
    configuration = source_ledger.get('context', {}).get('targets', {}).get(f'{dataset}/{strategy}')
    if configuration is None or identity_sha256(configuration) != link.get('configuration_sha256'):
        raise RuntimeError('Source gate configuration differs from reuse configuration')
    source_stage = source_ledger.get('stages', {}).get('one_query_matrix_16', {})
    if source_stage.get('status') != 'canary_passed':
        raise RuntimeError('Reuse source matrix gate was not passed')
    _, matrix = bound({'path': source_stage['evidence_path'], 'sha256': source_stage['evidence_sha256']})
    if matrix.get('targets', {}).get(f'{dataset}/{strategy}') != link['source_evidence']:
        raise RuntimeError('Reuse source differs from the recorded complete-corpus matrix target')
    index, raw = source_evidence(link['source_evidence'], strategy, dataset)
    source = safe_name(index['run_id'])
    if source == target or link.get('source_run_id') != source:
        raise RuntimeError('Reuse requires distinct source-index and fresh-result identities')
    if link.get('index_stats') != index['native_index_stats'] or link.get('index_timing_seconds') != raw.get('timing_seconds'):
        raise RuntimeError('Reuse source stats or original timing binding differs')
    if link.get('corpus_identity_sha256') != identity_sha256(current_corpus_identity(dataset)):
        raise RuntimeError('Reuse corpus content changed')
    if link.get('configuration_sha256') != identity_sha256(target_configuration(strategy, dataset)):
        raise RuntimeError('Reuse effective method configuration changed')
    configure_target_environment(strategy, dataset, source)
    import math
    preparation = link.get('preparation_elapsed_seconds')
    if isinstance(preparation, bool) or not isinstance(preparation, (float, int)) or not math.isfinite(preparation) or preparation < 0:
        raise RuntimeError('Invalid measured reuse preparation time')
    if link.get('source_index_namespace') != os.environ['RAG_INDEX_NAMESPACE']:
        raise RuntimeError('Reuse source DB namespace differs')
    spec = get_strategy(strategy)
    clone = link.get('clone')
    if spec.output_env:
        original = local_path(os.environ[spec.output_env])
        target_root = local_path(f'{spec.output_default}/runs/{target}')
        if not isinstance(clone, dict) or clone.get('source_output_root') != str(original.relative_to(ROOT)) or clone.get('query_output_root') != str(target_root.relative_to(ROOT)):
            raise RuntimeError('Reuse clone roots are not the exact source and target')
        if clone.get('operation') != 'byte_identical_copy_without_metadata_rebinding':
            raise RuntimeError('Reuse clone requires its explicit copy operation')
        current = inventory(original)
        if clone.get('source_inventory') != current or clone.get('initial_clone_inventory_sha256') != identity_sha256(current):
            raise RuntimeError('Original native index/cache bytes changed')
        # At admission query caches may legitimately change only in the clone.
        if not target_root.is_dir() or any(p.is_symlink() for p in target_root.rglob('*')):
            raise RuntimeError('Query clone is missing or contains symlinks')
        if pristine_clone and inventory(target_root) != current:
            raise RuntimeError('Fresh query clone differs from original native bytes')
    elif clone is not None:
        raise RuntimeError('Service index reuse cannot invent a native output clone')
    configure(link)
    return link


def prepare(campaign: str, strategy: str, dataset: str) -> Path:
    """Allocate only a fresh result link and byte-identical native query clone."""
    from core.paper_compatibility import target_configuration
    from core.paper_policy import configure_target_environment
    from core.strategy_registry import get_strategy
    from scripts.paper_cold_canary import save
    from scripts.paper_gate_ledger import verify
    safe_name(campaign)
    ledger = local_path(f'data/results/{campaign}/gate_ledger.json')
    verify(ledger, campaign)
    gate = json.loads(ledger.read_text())['stages']['one_query_matrix_16']
    _, matrix = bound({'path': gate['evidence_path'], 'sha256': gate['evidence_sha256']})
    evidence_ref = matrix['targets'][f'{dataset}/{strategy}']
    index, raw = source_evidence(evidence_ref, strategy, dataset)
    target = safe_name(f'{campaign}-{dataset}-{strategy}')
    base = local_path(f'data/results/{target}')
    if base.exists():
        raise FileExistsError('Reuse full benchmark requires fresh result paths')
    configure_target_environment(strategy, dataset, index['run_id'])
    spec = get_strategy(strategy)
    started = time.perf_counter()
    clone = None
    if spec.output_env:
        original = local_path(os.environ[spec.output_env])
        destination = local_path(f'{spec.output_default}/runs/{target}')
        if destination.exists():
            raise FileExistsError('Reuse query workspace must be fresh')
        original_inventory = inventory(original)
        shutil.copytree(original, destination)
        if inventory(destination) != original_inventory or inventory(original) != original_inventory:
            raise RuntimeError('Native clone bytes differ; incomplete attempt preserved')
        clone = {'source_output_root': str(original.relative_to(ROOT)), 'query_output_root': str(destination.relative_to(ROOT)),
                 'source_inventory': original_inventory, 'initial_clone_inventory_sha256': identity_sha256(original_inventory),
                 'operation': 'byte_identical_copy_without_metadata_rebinding'}
    base.mkdir(parents=True, exist_ok=False)
    gate_snapshot = base / 'source_gate_ledger.json'
    ledger_bytes = ledger.read_bytes()
    with gate_snapshot.open('xb') as stream:
        stream.write(ledger_bytes)
    if gate_snapshot.read_bytes() != ledger_bytes or ledger.read_bytes() != ledger_bytes:
        raise RuntimeError('Gate ledger changed during immutable snapshot capture')
    value = {'version': 1, 'strategy': strategy, 'dataset': dataset, 'target_run_id': target,
             'source_run_id': index['run_id'], 'source_index_namespace': os.environ['RAG_INDEX_NAMESPACE'],
             'source_evidence': evidence_ref, 'gate_ledger': ref(gate_snapshot),
             'source_gate_ledger_provenance': ref(ledger), 'index_stats': index['native_index_stats'],
             'corpus_identity_sha256': identity_sha256(current_corpus_identity(dataset)),
             'configuration_sha256': identity_sha256(target_configuration(strategy, dataset)),
             'index_timing_seconds': raw.get('timing_seconds'), 'clone': clone,
             'preparation_elapsed_seconds': time.perf_counter() - started}
    path = base / 'index_link.json'
    save(path, value)
    validate(path, target, strategy, dataset, pristine_clone=True)
    return path


def validate_costs(link: dict, payload: dict) -> None:
    import math

    from core.phase_timing import validate_timing
    validate_timing(payload.get('benchmark_timing'))
    costs = payload.get('phase_costs')
    if not isinstance(costs, dict):
        raise TypeError('Reused index is missing original indexing and fresh benchmark costs')
    wall = payload['benchmark_timing']['total_wall_seconds']
    expected = {'index_source_timing_seconds': link['index_timing_seconds'],
                'reuse_preparation_seconds': link['preparation_elapsed_seconds'],
                'benchmark_wall_seconds': wall,
                'index_plus_benchmark_wall_seconds': link['index_timing_seconds']['total_elapsed_seconds'] + wall}
    if any(costs.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Reuse phase cost binding differs from original index and unique benchmark segments')
    latency = sum(float(row.get('latency', 0)) for row in payload.get('details', []))
    if not math.isclose(costs.get('query_latency_sum_seconds', -1), latency, rel_tol=1e-12, abs_tol=1e-9):
        raise RuntimeError('Query service latency sum differs from full result rows')
