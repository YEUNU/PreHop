"""Compare final selectors on recorded complete candidate pools.

No generation or embedding requests are made. Fused and representation ordering
use trace payloads. Raw hybrid scores can come from saved replay records or an
explicit read-only replay against the original Neo4j index.
Run from the repository root with python -m scripts.compare_prehop_selectors.
"""
import argparse
import asyncio
import gzip
import hashlib
import json
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

from models.prehop.graphrag import GraphRAG
from scripts.ablation_statistics import cluster_interval
from scripts.evaluate_saved_retrieval import METRICS, hotpot_metrics, multihop_metrics
from utils.hotpotqa import project_sentences, score


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(directory, event):
    raw = gzip.decompress((directory / event['payload']).read_bytes())
    if hashlib.sha256(raw).hexdigest() != event['payload_sha256']:
        raise ValueError('Trace payload hash mismatch')
    return json.loads(raw)


def indexed(rows, key):
    out = {r[key]: r for r in rows}
    if len(out) != len(rows):
        raise ValueError(f'Duplicate {key}')
    return out


def select(nodes, variant, limit):
    if variant == 'fused':
        if any(a['rank_fusion_score'] < b['rank_fusion_score'] for a, b in pairwise(nodes)):
            raise ValueError('Recorded fused order is not monotonic')
        ordered = nodes
    else:
        field = {'representation': 'representation_score', 'raw': 'raw_channel_rrf_score'}[variant]
        # Preserve the evaluated descending-identity tie breaker.
        ordered = sorted(nodes, key=lambda n: (float(n[field]), n['id']), reverse=True)
    return GraphRAG._build_unique_sources(ordered[:limit])


def assign_raw_scores(nodes, channels, decay):
    """Require identical owner orders before propagating replayed channel scores."""
    direct = {n['id']: n for n in nodes if n.get('representation_scores')}
    scores = dict.fromkeys(direct, 0.0)
    expected_channels = {c for n in direct.values() for c in n['representation_scores']}
    if expected_channels - channels.keys():
        raise ValueError('Missing replay channel')
    for channel, found in channels.items():
        expected = sorted((n for n in direct.values() if channel in n['representation_scores']),
                          key=lambda n: (-n['representation_scores'][channel], n['id']))
        if [n['id'] for n in found] != [n['id'] for n in expected]:
            raise ValueError(f'Replayed owner order differs: {channel}')
        for node in found:
            scores[node['id']] += float(node['rrf_score'])
    result = []
    for n in nodes:
        if n['id'] in direct:
            value = scores[n['id']]
        else:
            paths = n.get('retrieval_paths', [])
            if not paths or any(p['kind'] not in ('hop', 'next') for p in paths):
                raise ValueError('Missing graph-only candidate paths')
            value = max(scores[p['source_chunk_id']] * decay for p in paths)
        result.append({**n, 'raw_channel_rrf_score': value})
    return result


async def replay_channels(data, namespace, limit):
    from core.neo4j_service import Neo4jService
    from models.prehop.retrieval.hybrid import HybridSearchMixin
    from models.prehop.retrieval.text_utils import TextUtilsMixin

    class Search(HybridSearchMixin, TextUtilsMixin):
        def __init__(self, db):
            self.neo4j = db
            self.chunk_label = f'PR_{namespace}_Chunk'
            for role, suffix in [('body', ''), ('q_minus', 'qminus_'), ('q_plus', 'qplus_')]:
                setattr(self, f'{role}_vector_index', f'prehop_{namespace}_{suffix}vector_idx')
                setattr(self, f'{role}_text_index', f'prehop_{namespace}_{suffix}text_idx')

        async def _run_channel_query(self, query, params):
            query = query.replace('owner.embedding AS embedding', 'null AS embedding')
            query = query.replace('node.embedding AS embedding', 'null AS embedding')
            return await self.neo4j.execute_query(query, params)

    db = Neo4jService()
    try:
        engine = Search(db)
        return {c: [{k: n[k] for k in ('id', 'rrf_score')} for n in
                    await engine._hybrid_rrf_candidates(data['query_text'], data['query_embedding'], limit, c)]
                for c in ('q_minus', 'body', 'q_plus')}
    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--dataset', choices=['multihoprag', 'hotpotqa'], required=True)
    parser.add_argument('--sentence-store', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--trace-events', type=Path, nargs='+', help='Override moved trace event paths')
    raw = parser.add_mutually_exclusive_group()
    raw.add_argument('--raw-score-records', type=Path, help='Saved raw replay JSONL with channel_scores and trace_sha256')
    raw.add_argument('--replay-raw', action='store_true', help='Read original Neo4j index; no generation')
    parser.add_argument('--index-stats', type=Path, help='Required with --replay-raw')
    args = parser.parse_args()
    if args.dataset == 'hotpotqa' and not args.sentence_store:
        parser.error('--sentence-store is required for HotpotQA')
    if args.replay_raw and not args.index_stats:
        parser.error('--index-stats is required with --replay-raw')
    # A new directory prevents accidental replacement of saved evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    ref = json.loads(args.reference.read_text())
    rows = indexed(ref['details'], 'query_id')
    queries = indexed(json.loads(args.queries.read_text()), '_id')
    if rows.keys() != queries.keys() or any(r.get('error') for r in rows.values()):
        raise ValueError('Reference must cover the complete query population without failures')
    if ref['ablation']['final_rank_variant'] != 'fused':
        raise ValueError('Reference does not use the evaluated fused candidate order')
    paths = args.trace_events or sorted({r['prehop_trace']['events_path'] for r in rows.values()})
    events = {}
    wanted = {'SimilarityScoringMixin._role_body_list_ranking.start',
              'SimilarityScoringMixin._score_and_select.start'}
    for path in map(Path, paths):
        with path.open() as f:
            for line in f:
                e = json.loads(line)
                qid = e.get('identity', {}).get('query_id')
                if qid not in rows or e.get('event') not in wanted:
                    continue
                key = (qid, e['event'])
                if key in events:
                    raise ValueError('Duplicate selection trace event')
                events[key] = (path.parent, e)
    recorded_raw = None
    if args.raw_score_records:
        with args.raw_score_records.open() as f:
            recorded_raw = indexed([json.loads(line) for line in f], 'query_id')
        if recorded_raw.keys() != rows.keys():
            raise ValueError('Raw-score records must cover the same query population')
    namespace = None
    if args.replay_raw:
        from core.config import RAGConfig
        stats = json.loads(args.index_stats.read_text())
        namespace = stats['index_policy']['index_namespace']
        RAGConfig.QUESTIONS_PER_DIRECTION = ref['ablation']['questions_per_direction']
    use_raw = recorded_raw is not None or args.replay_raw
    variants = ['llm', 'fused', 'representation'] + (['raw'] if use_raw else [])
    outputs = {v: [] for v in variants}

    @lru_cache(maxsize=100000)
    def project(serialized):
        return project_sentences([json.loads(serialized)], args.sentence_store)

    raw_out = (args.output / 'raw-score-records.jsonl').open('w') if args.replay_raw else None
    try:
        for qid, row in rows.items():
            directory, event = events[(qid, 'SimilarityScoringMixin._role_body_list_ranking.start')]
            data = payload(directory, event)
            limit = data['top_k']
            nodes = data['ordered']
            selected = {'llm': row['retrieved_sources'],
                        'fused': select(nodes, 'fused', limit),
                        'representation': select(nodes, 'representation', limit)}
            if use_raw:
                directory, event = events[(qid, 'SimilarityScoringMixin._score_and_select.start')]
                score_input = payload(directory, event)
                if {n['id'] for n in score_input['candidates']} != {n['id'] for n in nodes}:
                    raise ValueError('Candidate pools differ between scoring and selection')
                if recorded_raw is not None:
                    record = recorded_raw[qid]
                    if record['trace_sha256'] != event['payload_sha256']:
                        raise ValueError('Raw scores belong to a different candidate trace')
                    channels = record['channel_scores']
                else:
                    channels = asyncio.run(replay_channels(score_input, namespace, ref['ablation']['default_top_k']))
                raw_nodes = assign_raw_scores(score_input['candidates'], channels, ref['ablation']['graph_path_decay'])
                selected['raw'] = select(raw_nodes, 'raw', limit)
                if raw_out:
                    raw_out.write(json.dumps({'query_id': qid, 'channel_scores': channels,
                                              'trace_sha256': event['payload_sha256']}) + '\n')
                    raw_out.flush()
            q = queries[qid]
            for variant, sources in selected.items():
                metrics = (multihop_metrics(sources, q.get('evidence_facts', [])) if args.dataset == 'multihoprag'
                           else hotpot_metrics(sources, q['supporting_facts'], lambda s: project(json.dumps(s, sort_keys=True))))
                support = {}
                if args.dataset == 'hotpotqa':
                    predicted = project_sentences(sources, args.sentence_store)
                    support = {k: v for k, v in score('', predicted, '', q['supporting_facts']).items() if k.startswith('sp_')}
                outputs[variant].append({'query_id': qid, 'original_query_id': q.get('original_query_id') or qid,
                                         'retrieved_sources': sources, 'metrics': metrics, 'support': support})
    finally:
        if raw_out:
            raw_out.close()
    report = {'scope': 'Fixed complete candidates; retrieval and support only; no answer or full-query timing',
              'dataset': args.dataset, 'reference_sha256': sha(args.reference), 'queries_sha256': sha(args.queries),
              'rows': len(rows), 'resamples': 10000, 'seed': 42, 'conditions': {}}
    if args.sentence_store:
        report['sentence_store_sha256'] = sha(args.sentence_store)
    if args.raw_score_records:
        report['raw_score_records_sha256'] = sha(args.raw_score_records)
    for variant, details in outputs.items():
        result = {'metrics': {}, 'support': {}}
        for field, metrics in [('metrics', METRICS), ('support', ('sp_em', 'sp_prec', 'sp_recall', 'sp_f1'))]:
            for metric in metrics:
                pairs = [(a, b) for a, b in zip(details, outputs['llm'], strict=True)
                         if a[field] is not None and metric in a[field]]
                if not pairs:
                    continue
                groups = [a['original_query_id'] for a, _ in pairs]
                result[field][metric] = {'condition': cluster_interval([a[field][metric] for a, _ in pairs], groups),
                                        'minus_llm': cluster_interval([a[field][metric] - b[field][metric] for a, b in pairs], groups)}
        report['conditions'][variant] = result
        (args.output / f'{variant}.json').write_text(json.dumps({'details': details}) + '\n')
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'rows': len(rows), 'conditions': variants, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
