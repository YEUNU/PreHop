"""Clone frozen body properties and structural edges, then build C-only HOP links."""
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def copy_body_graph(engine, source_namespace):
    """Copy only Document/Chunk, CONTAINS and NEXT; never questions or HOP."""
    for kind, key in [('Document', 'filename'), ('Chunk', 'id')]:
        after = ''
        while True:
            rows = await engine.retry_query(
                f'MATCH (s:PR_{source_namespace}_{kind}) WHERE s.{key} > $after '
                f'WITH s ORDER BY s.{key} LIMIT $limit '
                f'CREATE (d:PR_{engine._safe_corpus}_{kind}) SET d = properties(s) '
                f'RETURN s.{key} AS id ORDER BY id', {'after': after, 'limit': 64})
            if not rows:
                break
            after = rows[-1]['id']
    for relation, source_kind, source_key in [('CONTAINS', 'Document', 'filename'), ('NEXT', 'Chunk', 'id')]:
        after = ''
        while True:
            rows = await engine.retry_query(
                f'MATCH (s:PR_{source_namespace}_{source_kind}) WHERE s.{source_key} > $after '
                f'WITH s ORDER BY s.{source_key} LIMIT $limit '
                f'OPTIONAL MATCH (s)-[:{relation}]->(t:PR_{source_namespace}_Chunk) '
                f'WITH s, collect(t.id) AS targets '
                f'MATCH (d:PR_{engine._safe_corpus}_{source_kind} {{{source_key}: s.{source_key}}}) '
                'CALL (d, targets) { UNWIND targets AS target '
                f'MATCH (t:PR_{engine._safe_corpus}_Chunk {{id: target}}) '
                f'MERGE (d)-[:{relation}]->(t) }} '
                f'RETURN s.{source_key} AS id ORDER BY id', {'after': after, 'limit': 64})
            if not rows:
                break
            after = rows[-1]['id']


async def run(source_stats, dataset):
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    from cli.index import (
        _collect_index_capacity,
        _collect_prehop_integrity,
        _index_policy_artifact,
        _load_corpus_manifest,
        _publish_neo4j_snapshot,
        _set_neo4j_snapshot_state,
        _staged_source_ids,
    )
    from core.config import RAGConfig
    from core.neo4j_service import Neo4jService
    from models.prehop.graphrag import GraphRAG
    from scripts.prehop_ablation import export_reference
    from utils.provenance import code_provenance
    source_path = Path(source_stats).resolve()
    original_bytes = source_path.read_bytes()
    source = json.loads(original_bytes)
    tag = source['corpus_tag']
    namespace = source['index_policy']['index_namespace']
    manifest = _load_corpus_manifest(dataset)
    files = sorted(p.name for p in Path(dataset).iterdir() if p.suffix in {'.txt', '.md'} and p.is_file())
    ids = _staged_source_ids(files, manifest, Path(dataset))
    target = Path('data/index_stats') / f'prehop_{tag}_{os.environ["RAG_RUN_ID"]}.json'
    destination = os.environ['RAG_INDEX_NAMESPACE']
    try:
        # Resolve the source namespace without altering source nodes.
        os.environ['RAG_INDEX_NAMESPACE'] = namespace
        GraphRAG(strategy='prehop', corpus_tag=tag)
    finally:
        os.environ['RAG_INDEX_NAMESPACE'] = destination
    reference = Path(RAGConfig.BODY_LINK_REFERENCE)
    reference.parent.mkdir(parents=True, exist_ok=True)
    await export_reference(namespace, reference)
    engine = GraphRAG(strategy='prehop', corpus_tag=tag)
    started = time.perf_counter()
    try:
        await _set_neo4j_snapshot_state(engine, 'prehop', tag, manifest, 'in_progress')
        await engine.setup_index()
        await copy_body_graph(engine, namespace)
        copied = time.perf_counter()
        await engine.build_all_hop_edges()  # Records identity/degree observations.
        linked = time.perf_counter()
        quality = await _collect_prehop_integrity(engine)
        snapshot = await _publish_neo4j_snapshot(engine, 'prehop', tag, ids, manifest)
        elapsed = time.perf_counter() - started
        stats = {'status': 'complete', 'strategy': 'prehop', 'corpus_tag': tag,
            'run_id': os.environ['RAG_RUN_ID'], 'dataset_path': str(Path(dataset).resolve()),
            'index_code_provenance': code_provenance(),
            **_index_policy_artifact('prehop', 'default', tag),
            'corpus_manifest_fingerprint': manifest['fingerprint'],
            'corpus_manifest_paragraph_count': manifest['paragraph_count'],
            'active_snapshot': snapshot, 'index_quality': quality,
            'total_documents': len(ids),
            'total_chunks': len(json.loads(reference.read_text())['nodes']),
            'total_hop_edges': quality['diagnostics']['hop_edges'],
            'index_capacity': await _collect_index_capacity('prehop', tag, engine.neo4j),
            'timing_seconds': {'body_clone_seconds': copied-started,
                'hop_construction_seconds': linked-copied, 'total_elapsed_seconds': elapsed},
            'body_link_diagnostics': getattr(engine,'body_link_diagnostics',{}),
            'body_reuse': {'source_stats_path': str(source_path),
                'source_stats_sha256': hashlib.sha256(original_bytes).hexdigest(),
                'source_namespace': namespace, 'source_timing_seconds': source.get('timing_seconds'),
                'reference_path': str(reference.resolve()),
                'embedding_requests': 0, 'generation_requests': 0,
                'cost_scope': 'clone_and_body_links_only; source construction cost recorded separately'}}
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x') as handle:
            json.dump(stats, handle, indent=2)
        print(json.dumps({'status': 'complete', 'index_stats': str(target)}))
    except BaseException:
        await _set_neo4j_snapshot_state(engine, 'prehop', tag, manifest, 'failed')
        raise
    finally:
        await Neo4jService.global_close()


if __name__ == '__main__':
    asyncio.run(run(sys.argv[1], sys.argv[2]))
