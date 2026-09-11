"""Primary component runs must not inherit representation-ablation policies."""
import gzip
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.run_primary_hop_ablation import environment, payload


def test_environment_preserves_primary_and_removes_only_hop(tmp_path):
    stats = tmp_path / 'stats.json'
    stats.write_text(json.dumps({'run_id': 'original-index', 'index_policy': {'index_namespace': 'original_graph'}}))
    reference = {
        'index_manifest_stats_path': str(stats),
        'models': {'default': 'original-generator', 'embedding': 'original-embedding'},
        'ablation': {
            'q_minus': True, 'q_plus': True, 'sentence_channel_enabled': False,
            'graph_hop_depth': 1, 'graph_path_decay': 0.5, 'hop_edge_filter': 'none',
            'qplus_hop_activation': 'owner', 'continuation_edges_enabled': False,
            'continuation_anchor_policy': 'named_only', 'hop_semantic_variant': 'body_bridge_min',
            'question_schema': 'legacy', 'candidate_pool_multiplier': 1,
            'hypo_channel_variant': 'full', 'source_selection_variant': 'role_body_list_ranking',
            'candidate_order_input_order': 'search', 'candidate_order_shuffle_seed': 0,
            'final_rank_variant': 'fused',
        },
    }
    from scripts.run_primary_hop_ablation import ROOT
    env = environment(reference, ROOT / 'data/results/test-primary-removal', tmp_path / 'inputs')
    assert env['RAG_GRAPH_EDGE_VARIANT'] == 'next_only'
    assert env['RAG_GRAPH_HOP_DEPTH'] == '1'
    assert env['RAG_HOP_SEED_POLICY'] == 'qplus'
    assert env['RAG_HOP_SEMANTIC_VARIANT'] == 'body_bridge_min'
    assert env['RAG_HYPO_CHANNEL_VARIANT'] == 'full'
    assert env['RAG_QPLUS_HOP_ACTIVATION'] == 'owner'
    assert env['RAG_GENERATION_MODEL'] == 'original-generator'
    assert env['RAG_LLM_SEED'] == ''
    assert env['RAG_INDEX_NAMESPACE'] == 'original_graph'


def test_trace_payload_is_verified_without_modifying_source(tmp_path):
    raw = b'{"query_embedding": [0.1, 0.2], "matched_qplus_ids": ["q1"]}'
    compressed = gzip.compress(raw)
    (tmp_path / 'payload.gz').write_bytes(compressed)
    event = {'payload': 'payload.gz', 'payload_sha256': hashlib.sha256(raw).hexdigest()}
    assert payload(tmp_path, event)['matched_qplus_ids'] == ['q1']
    assert (tmp_path / 'payload.gz').read_bytes() == compressed
    event['payload_sha256'] = 'incorrect'
    with pytest.raises(ValueError, match='hash mismatch'):
        payload(tmp_path, event)


@pytest.mark.asyncio
async def test_next_only_reads_bidirectional_next_and_no_hop(monkeypatch):
    from core.config import RAGConfig
    from models.prehop.retrieval.traversal import TraversalMixin

    async def empty_records():
        for row in []:
            yield row

    monkeypatch.setattr(RAGConfig, 'GRAPH_EDGE_VARIANT', 'next_only')
    monkeypatch.setattr(RAGConfig, 'CONNECTION_TIMING_MODE', '')
    session = AsyncMock()
    session.run.return_value = empty_records()
    context = AsyncMock()
    context.__aenter__.return_value = session
    reader = TraversalMixin()
    reader.chunk_label = 'OriginalChunk'
    reader.q_plus_vector_index = 'original_qplus'
    reader.neo4j = SimpleNamespace(driver=SimpleNamespace(session=MagicMock(return_value=context)))
    assert await reader._expand_frontier(['seed'], set(), {'seed': {'q1'}}) == []
    query, parameters = session.run.await_args.args
    assert 'MATCH (src)-[:NEXT]-(related:OriginalChunk)' in query
    assert 'HOP_ANSWER' not in query
    assert parameters['frontier_ids'] == ['seed']


def test_primary_inputs_keep_duplicate_release_occurrences_separate(tmp_path):
    from models.prehop.ablation_inputs import read_input, write_input
    from models.prehop.tracing import trace_identity
    for qid, score in [('original', .1), ('original__2', .9)]:
        write_input(tmp_path / 'occurrences', qid, {'score': score})
    with trace_identity(query_id='original'):
        assert read_input(tmp_path, 'same question')['score'] == .1
    with trace_identity(query_id='original__2'):
        assert read_input(tmp_path, 'same question')['score'] == .9


def test_hotpot_primary_comparison_uses_official_support_metrics():
    from scripts.run_primary_hop_ablation import comparison_metrics
    metrics = comparison_metrics('hotpotqa')
    assert metrics[0] == 'hotpot_sp_f1'
    assert set(metrics) == {'hotpot_sp_f1', 'hotpot_sp_em', 'hotpot_f1', 'hotpot_em',
                            'hotpot_joint_f1', 'hotpot_joint_em'}


def test_prepare_accepts_hotpot_population_and_preserves_each_prefix(tmp_path, monkeypatch):
    from models.prehop.ablation_inputs import read_input
    from models.prehop.tracing import trace_identity
    from scripts import run_primary_hop_ablation as run
    events=tmp_path/'events.jsonl'
    event_rows=[]
    for qid,vector in [('q1',[.1]),('q2',[.9])]:
        for name,value in [
            ('RetrieveMixin._retrieve_with_candidate_pool.start',
             {'query':'same','query_embedding':vector,'select_final':False,'top_k':12}),
            ('RetrieveMixin._retrieve_with_candidate_pool.result',[[],[{'id':qid}]])]:
            raw=json.dumps(value).encode()
            file=f'{qid}-{len(event_rows)}.gz'
            (tmp_path/file).write_bytes(gzip.compress(raw))
            event_rows.append({'event':name,'identity':{'query_id':qid},'payload':file,
                               'payload_sha256':hashlib.sha256(raw).hexdigest()})
    events.write_text('\n'.join(json.dumps(e) for e in event_rows))
    reference=tmp_path/'reference.json'
    reference.write_text(json.dumps({'query_failure_count':0,'details':[
        {'query_id':qid,'prehop_trace':{'events_path':str(events)}} for qid in ['q1','q2']],
        'ablation':{'prompt_configuration_sha256':'prompt','generation_profiles_sha256':'generation'}}))
    queries=tmp_path/'queries.json'
    queries.write_text(json.dumps([{'_id':qid,'original_query_id':'original'} for qid in ['q1','q2']]))
    monkeypatch.setattr(run,'environment',lambda *a: {})
    output=tmp_path/'prepared'
    run.prepare(SimpleNamespace(reference=reference,queries=queries,output=output))
    for qid,vector in [('q1',[.1]),('q2',[.9])]:
        with trace_identity(query_id=qid):
            assert read_input(output/'primary-inputs','same')['query_embedding']==vector
