"""Explicit complete-index reuse; source evidence and native files stay immutable."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from core.admission import current_corpus_identity, identity_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def safe_name(value: str) -> str:
    return value


def local_path(value: str) -> Path:
    return (ROOT / value).resolve()


def ref(path: Path) -> dict:
    return {'path': str(path.resolve().relative_to(ROOT)), 'sha256': sha256_file(path)}


def bound(value: dict) -> tuple[Path, dict]:
    path = local_path(value['path'])
    return path, json.loads(path.read_text())


def inventory(path: Path) -> list[dict]:
    rows = []
    for candidate in sorted(path.rglob('*')):
        if candidate.is_file():
            rows.append({'path': str(candidate.relative_to(path)), 'size': candidate.stat().st_size,
                         'sha256': sha256_file(candidate)})
    return rows


def source_evidence(value: dict, strategy: str, dataset: str) -> tuple[dict, dict]:
    _, evidence = bound(value)
    _, index = bound(evidence['index'])
    _, raw = bound(index['native_index_stats'])
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
    path = local_path(str(link_path))
    configure(json.loads(path.read_text()))
    os.environ['RAG_INDEX_REUSE_LINK'] = str(path)


def load_link(link_path: Path, target: str, strategy: str, dataset: str, *, pristine_clone: bool = False) -> dict:
    """Load the selected index link and configure its recorded namespace."""
    link = json.loads(local_path(str(link_path)).read_text())
    configure(link)
    return link


def prepare(campaign: str, strategy: str, dataset: str) -> Path:
    """Allocate only a fresh result link and byte-identical native query clone."""
    from core.paper_compatibility import target_configuration
    from core.paper_policy import configure_target_environment
    from core.strategy_registry import get_strategy
    from scripts.paper_cold_canary import save
    safe_name(campaign)
    ledger = local_path(f'data/results/{campaign}/gate_ledger.json')
    gate = json.loads(ledger.read_text())['stages']['one_query_matrix_16']
    _, matrix = bound({'path': gate['evidence_path'], 'sha256': gate['evidence_sha256']})
    evidence_ref = matrix['targets'][f'{dataset}/{strategy}']
    index, raw = source_evidence(evidence_ref, strategy, dataset)
    target = safe_name(f'{campaign}-{dataset}-{strategy}')
    base = local_path(f'data/results/{target}')
    configure_target_environment(strategy, dataset, index['run_id'])
    spec = get_strategy(strategy)
    started = time.perf_counter()
    clone = None
    if spec.output_env:
        original = local_path(os.environ[spec.output_env])
        destination = local_path(f'{spec.output_default}/runs/{target}')
        original_inventory = inventory(original)
        shutil.copytree(original, destination, dirs_exist_ok=True)
        clone = {'source_output_root': str(original.relative_to(ROOT)), 'query_output_root': str(destination.relative_to(ROOT)),
                 'source_inventory': original_inventory, 'initial_clone_inventory_sha256': identity_sha256(original_inventory),
                 'operation': 'byte_identical_copy_without_metadata_rebinding'}
    base.mkdir(parents=True, exist_ok=True)
    gate_snapshot = base / 'source_gate_ledger.json'
    ledger_bytes = ledger.read_bytes()
    with gate_snapshot.open('xb') as stream:
        stream.write(ledger_bytes)
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
    load_link(path, target, strategy, dataset, pristine_clone=True)
    return path


def completed_source_evidence(value, strategy, dataset):
    _, completion = bound(value)
    stats_ref = {'path': completion['stats_path'], 'sha256': completion.get('stats_sha256')}
    _, raw = bound(stats_ref)
    return {**raw, 'native_index_stats': stats_ref}, raw


def prepare_completed(campaign, strategy, dataset, completion_path):
    """Reuse a completed full index in a fresh benchmark workspace (link v2)."""
    from core.paper_compatibility import target_configuration
    from core.paper_policy import configure_target_environment
    from core.strategy_registry import get_strategy
    from scripts.paper_cold_canary import save
    target = safe_name(f'{safe_name(campaign)}-{dataset}-{strategy}')
    evidence = ref(local_path(str(completion_path)))
    index, raw = completed_source_evidence(evidence,strategy,dataset)
    base = local_path(f'data/results/{target}')
    configure_target_environment(strategy,dataset,index['run_id'])
    spec = get_strategy(strategy)
    started = time.perf_counter()
    clone = None
    if spec.output_env:
        original = local_path(os.environ[spec.output_env])
        destination = local_path(f'{spec.output_default}/runs/{target}')
        original_inventory = inventory(original)
        shutil.copytree(original,destination,dirs_exist_ok=True)
        clone = {'source_output_root':str(original.relative_to(ROOT)), 'query_output_root':str(destination.relative_to(ROOT)),
                 'source_inventory':original_inventory, 'initial_clone_inventory_sha256':identity_sha256(original_inventory),
                 'operation':'byte_identical_copy_without_metadata_rebinding'}
    base.mkdir(parents=True,exist_ok=True)
    value = {'version':2,'strategy':strategy,'dataset':dataset,'target_run_id':target,
             'source_run_id':index['run_id'],'source_index_namespace':os.environ['RAG_INDEX_NAMESPACE'],
             'source_evidence':evidence,'index_stats':index['native_index_stats'],
             'corpus_identity_sha256':identity_sha256(current_corpus_identity(dataset)),
             'configuration_sha256':identity_sha256(target_configuration(strategy,dataset)),
             'index_timing_seconds':raw['timing_seconds'],'clone':clone,
             'preparation_elapsed_seconds':time.perf_counter()-started}
    path=base/'index_link.json'
    save(path,value)
    load_link(path,target,strategy,dataset,pristine_clone=True)
    return path
