#!/usr/bin/env python3
"""Run one fresh, preregistered cold integration target after prerequisite gates."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path: Path, value: dict) -> dict:
    from core.admission import sha256_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)
        stream.write('\n')
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256_file(path)}


async def ensure_fresh_namespace(strategy: str, dataset: str) -> None:
    cache = os.environ.get('RAG_CHUNK_CACHE_DIR')
    if cache and Path(cache).exists():
        raise FileExistsError('Fresh index cannot reuse an existing chunk-generation cache')
    if strategy == 'ms_graphrag':
        from models.ms_graphrag.official_indexer import output_dir_for
        if output_dir_for(dataset).exists():
            raise FileExistsError('Native output already exists')
    elif strategy not in {'prehop', 'naive', 'hoprag'}:
        from models.official_baseline_runtime import corpus_output_dir
        if corpus_output_dir(strategy, dataset).exists():
            raise FileExistsError('Native output already exists')
    else:
        from core.index_namespace import index_namespace
        from core.neo4j_service import Neo4jService
        service = Neo4jService()
        namespace = index_namespace(dataset)
        rows = await service.execute_query(
            'MATCH (n) UNWIND labels(n) AS label WITH DISTINCT label WHERE label CONTAINS $namespace RETURN label',
            {'namespace': namespace})
        markers = await service.execute_query(
            'MATCH (m:RAGIndexSnapshot {index_namespace:$namespace}) RETURN count(m) AS count',
            {'namespace': namespace})
        schema = await service.execute_query('SHOW INDEXES YIELD name, labelsOrTypes RETURN name, labelsOrTypes')
        if rows or not markers or markers[0]['count'] or any(
            namespace in row['name'] or any(namespace in label for label in (row.get('labelsOrTypes') or []))
            for row in schema
        ):
            raise RuntimeError('Cold namespace already has nodes, metadata, or schema indexes')


async def workflow(campaign: str, strategy: str, dataset: str, attempt: str, *, full_corpus: bool = False) -> None:
    from core.paper_policy import configure_target_environment
    from core.strategy_registry import PRIMARY_STRATEGIES
    if strategy not in PRIMARY_STRATEGIES or dataset not in {'multihoprag', 'musique'}:
        raise ValueError('Unknown primary target')
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value) for value in (campaign, attempt)):
        raise ValueError('Campaign and attempt must be safe nonempty identifiers')
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError('Run the cold executor from the repository root')
    stage = 'one_query_matrix_16' if full_corpus else 'cold_canary_16'
    branch = 'one_query' if full_corpus else 'cold_v2'
    run_id = f'{campaign}-{branch}-{dataset}-{strategy}-{attempt}'
    configure_target_environment(strategy, dataset, run_id)
    from scripts.paper_gate_ledger import _validate_canary_artifacts, ready
    ledger = ROOT / 'data/results' / campaign / 'gate_ledger.json'
    ready(ledger, stage)
    from core.admission import current_post_query_inventory, sha256_file
    from core.runtime_requirements import runtime_identity
    from scripts.cold_canary_fixture import fixture_identity, stage_fixture
    base = ROOT / 'data/results' / campaign / branch / attempt / dataset / strategy
    raw_path = ROOT / os.environ['RAG_INDEX_STATS_PATH']
    if base.exists() or raw_path.exists():
        raise FileExistsError('Preserve earlier canary attempt or native stats')
    engine = None
    try:
        await ensure_fresh_namespace(strategy, dataset)
        if full_corpus:
            corpus = ROOT / 'data' / f'{dataset}_corpus'
            rows = json.loads((ROOT / 'data' / f'{dataset}_queries.json').read_text())
            if not isinstance(rows, list) or not rows:
                raise RuntimeError('Canonical full query manifest is empty')
            row = rows[0]
        else:
            corpus, _, row = stage_fixture(base, dataset)
        from cli.index import _load_corpus_manifest, _validate_staged_snapshot, run_indexing
        loaded = _load_corpus_manifest(corpus)
        files = sorted(path.name for path in corpus.glob('*.txt'))
        _validate_staged_snapshot(files, loaded, corpus)
        source_ref = {'path': str((corpus / 'corpus_manifest.json').relative_to(ROOT)),
                      'sha256': sha256_file(corpus / 'corpus_manifest.json')}
        record_ref = save(base / 'query_record.json', row)
        print(f"native_index_start strategy={strategy} dataset={dataset} sources={loaded['paragraph_count']}", flush=True)
        await run_indexing(str(corpus), strategy, 'default', dataset)
        raw = json.loads(raw_path.read_text())
        if raw.get('status') != 'complete':
            raise RuntimeError('Native index not complete')
        if strategy == 'ms_graphrag':
            from models.ms_graphrag.ms_adapter import MSGraphRAGAdapter
            engine = MSGraphRAGAdapter(corpus_tag=dataset)
        elif strategy == 'prehop':
            from models.prehop.graphrag import GraphRAG
            engine = GraphRAG(strategy=strategy, corpus_tag=dataset)
        elif strategy == 'naive':
            from models.naive.naive_rag import NaiveRAG
            engine = NaiveRAG(strategy=strategy, corpus_tag=dataset)
        else:
            from models.external_research.adapter import ExternalResearchAdapter
            engine = ExternalResearchAdapter(strategy, corpus_tag=dataset)
        from cli.benchmark import _verify_active_index_snapshot
        snapshot = await _verify_active_index_snapshot(
            engine, strategy, dataset, {**loaded, 'path': str(corpus / 'corpus_manifest.json')}, True)
        if snapshot.get('source_count') != loaded['paragraph_count']:
            raise RuntimeError('Observed native snapshot source count differs from complete selected corpus')
        index = {**raw, 'dataset': dataset, 'fresh_index': True, 'source_count': snapshot['source_count'],
                 'source_manifest': source_ref,
                 **({} if full_corpus else {'cold_fixture': fixture_identity()}),
                 'native_index_stats': {'path': str(raw_path.relative_to(ROOT)), 'sha256': sha256_file(raw_path)}}
        # Preserve native worker observations needed by the production verifier.
        if snapshot.get('official_stats') is not None:
            index['official_stats'] = snapshot['official_stats']
        index_path = base / 'index_evidence.json'
        index_ref = save(index_path, index)
        answer, sources, _trace = await engine.run_workflow(row['query'])
        documents = [{'source_id': Path(item['source']).stem, 'native_source': item['source'],
                      'title': item['doc'], 'text': item['text']} for item in sources]
        if not isinstance(answer, str) or not answer.strip() or not documents:
            raise RuntimeError('Cold query requires nonempty answer and native evidence')
        query = {'strategy': strategy, 'dataset': dataset, 'run_id': run_id, 'query_count': 1,
                 'query_id': row['_id'], 'query': row['query'], 'query_record': record_ref,
                 'query_records_sha256': record_ref['sha256'], 'answer': answer, 'documents': documents,
                 'index_provenance': {'policy_sha256': index['index_policy_sha256']},
                 'index_stats_sha256': index_ref['sha256'], 'runtime_identity': runtime_identity(strategy),
                 'post_query_artifact_inventory': current_post_query_inventory(strategy, dataset),
                 'active_index_snapshot': snapshot}
        query_ref = save(base / 'query.json', query)
        _validate_canary_artifacts(stage, strategy, dataset, index_path, query, index)
        ready(ledger, stage)
        admission_ref = save(base / 'admission.json', {'strategy': strategy, 'dataset': dataset, 'run_id': run_id,
            'status': 'canary_passed', 'errors': [], 'index_sha256': index_ref['sha256'], 'query_sha256': query_ref['sha256']})
        invocation_ref = save(base / 'invocation.json', {'argv': sys.argv, 'executor_sha256': sha256_file(Path(__file__)),
            'exit_code': 0, 'target': dataset + '/' + strategy})
        save(base / 'evidence.json', {'stage': stage, 'status': 'canary_passed', 'strategy': strategy,
            'dataset': dataset, 'exit_code': 0, 'index': index_ref, 'query': query_ref,
            'admission': admission_ref, 'invocation': invocation_ref})
        print(f'cold_native_canary_passed strategy={strategy} dataset={dataset}', flush=True)
    finally:
        if engine is not None and strategy not in {'prehop', 'naive', 'ms_graphrag'}:
            engine.close()
        if strategy in {'prehop', 'naive'}:
            from core.neo4j_service import Neo4jService
            await Neo4jService.global_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign')
    parser.add_argument('strategy')
    parser.add_argument('dataset')
    parser.add_argument('--attempt', required=True, help='Fresh attempt namespace; existing attempts are never overwritten')
    args = parser.parse_args()
    from scripts.check_paper_runtime import _load_runner_environment
    _load_runner_environment()
    asyncio.run(workflow(args.campaign, args.strategy, args.dataset, args.attempt))


if __name__ == '__main__':
    main()
