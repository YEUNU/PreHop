"""Selection contrasts must retain candidate identity and reference ordering."""
import pytest

from scripts.compare_prehop_selectors import assign_raw_scores, indexed, select


def test_raw_scores_preserve_overlap_and_max_path_inheritance():
    nodes = [
        {'id': 'a', 'representation_scores': {'body': 1, 'q_minus': .5}},
        {'id': 'b', 'representation_scores': {'q_minus': 1}},
        {'id': 'c', 'retrieval_paths': [
            {'kind': 'hop', 'source_chunk_id': 'a'},
            {'kind': 'next', 'source_chunk_id': 'b'}]},
    ]
    channels = {'body': [{'id': 'a', 'rrf_score': 1.5}],
                'q_minus': [{'id': 'b', 'rrf_score': 1.8}, {'id': 'a', 'rrf_score': .8}]}
    result = assign_raw_scores(nodes, channels, .5)
    assert [r['raw_channel_rrf_score'] for r in result] == pytest.approx([2.3, 1.8, 1.15])
    assert 'raw_channel_rrf_score' not in nodes[0]
    channels['q_minus'].reverse()
    with pytest.raises(ValueError, match='owner order differs'):
        assign_raw_scores(nodes, channels, .5)


def test_selector_ties_and_fused_reference_order_are_distinct():
    nodes = [{'id': name, 'source': name, 'text': name, 'rank_fusion_score': 1,
              'representation_score': .5} for name in ['a', 'b']]
    assert select(nodes, 'fused', 1)[0]['chunk_id'] == 'a'
    assert select(nodes, 'representation', 1)[0]['chunk_id'] == 'b'
    nodes[1]['rank_fusion_score'] = 2
    with pytest.raises(ValueError, match='not monotonic'):
        select(nodes, 'fused', 1)


def test_duplicate_occurrences_are_not_silently_dropped():
    with pytest.raises(ValueError, match='Duplicate'):
        indexed([{'query_id': 'q'}, {'query_id': 'q'}], 'query_id')


@pytest.mark.asyncio
async def test_raw_replay_uses_recorded_query_and_closes_connection(monkeypatch):
    from core import neo4j_service
    from models.prehop.retrieval.hybrid import HybridSearchMixin
    from scripts.compare_prehop_selectors import replay_channels

    calls = []

    class Database:
        closed = False

        async def close(self):
            self.closed = True

    db = Database()
    monkeypatch.setattr(neo4j_service, 'Neo4jService', lambda: db)

    async def search(self, query, embedding, limit, channel):
        calls.append((query, embedding, limit, channel, self._channel_index_names(channel)))
        return [{'id': 'passage', 'rrf_score': 1.5, 'embedding': [999]}]

    monkeypatch.setattr(HybridSearchMixin, '_hybrid_rrf_candidates', search)
    result = await replay_channels({'query_text': 'original', 'query_embedding': [.2]}, 'test_index', 12)
    assert db.closed
    assert {c[3] for c in calls} == {'body', 'q_minus', 'q_plus'}
    assert all(c[:3] == ('original', [.2], 12) for c in calls)
    assert calls[0][4][0] == 'prehop_test_index_qminus_vector_idx'
    assert result['body'] == [{'id': 'passage', 'rrf_score': 1.5}]
