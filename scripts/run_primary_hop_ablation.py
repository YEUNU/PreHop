"""Replay primary retrieval with controlled HOP/NEXT expansion.

The source result and trace payloads are read-only. This launcher deliberately
does not use the representation-ablation COMMON configuration.
"""
import argparse
import asyncio
import gzip
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Stable identities retain compatibility with completed component runs.
EXPANSIONS = {
    'next_only': ('without_hop', 'primary-hop-removal-v1',
                  'HOP traversal disabled; direct retrieval and NEXT retained'),
    'none': ('direct_only', 'primary-direct-only-v1',
             'HOP and NEXT traversal disabled; direct retrieval retained'),
    'hop_only': ('hop_only', 'primary-hop-only-v1',
                 'NEXT traversal disabled; direct retrieval and primary HOP retained'),
}


def comparison_metrics(tag):
    if tag == 'hotpotqa':
        return ['hotpot_sp_f1', 'hotpot_sp_em', 'hotpot_f1', 'hotpot_em',
                'hotpot_joint_f1', 'hotpot_joint_em']
    return ['official_hits@4', 'official_hits@10', 'official_mrr@10',
            'official_map@10', 'exact_fact_recall@10', 'all_facts@10']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload(directory, event):
    raw = gzip.decompress((directory / event['payload']).read_bytes())
    if hashlib.sha256(raw).hexdigest() != event['payload_sha256']:
        raise ValueError('Source trace payload hash mismatch')
    return json.loads(raw)


def environment(reference, output, inputs):
    """Copy scientific settings from the primary result, then change one edge flag."""
    stats = json.loads(Path(reference['index_manifest_stats_path']).read_text())
    fields = ('q_minus', 'q_plus', 'sentence_channel_enabled', 'graph_hop_depth',
              'graph_path_decay', 'hop_edge_filter', 'qplus_hop_activation',
              'continuation_edges_enabled', 'continuation_anchor_policy',
              'hop_semantic_variant', 'question_schema', 'candidate_pool_multiplier',
              'hypo_channel_variant', 'source_selection_variant',
              'candidate_order_input_order', 'candidate_order_shuffle_seed',
              'final_rank_variant')
    env = {}
    for key in fields:
        value = reference['ablation'][key]
        name = 'ABLATION_' + key.upper() if key in ('q_minus', 'q_plus') else key.upper()
        env['RAG_' + name] = str(value).lower() if isinstance(value, bool) else str(value)
    env.update({
        'RAG_GRAPH_EDGE_VARIANT': 'next_only',
        'RAG_HOP_SEED_POLICY': reference['ablation'].get('hop_seed_policy', 'qplus'),
        'RAG_HOP_LINK_VARIANT': reference['ablation'].get('hop_link_variant', 'question'),
        'RAG_PREHOP_ABLATION_PROFILE': 'primary_without_hop',
        'RAG_ABLATION_DIRECT_INPUTS': str(inputs.resolve()),
        'RAG_ABLATION_REUSE_EXISTING_INDEX': 'true',
        'RAG_CONNECTION_TIMING_MODE': '', 'RAG_CONNECTION_TIMING_STORE': '',
        'RAG_BODY_LINK_REFERENCE': '',
        'RAG_PAPER_MODE': 'false', 'RAG_JUDGE_ENABLED': 'false',
        'RAG_LLM_SEED': str(seed) if (seed := reference.get('execution_profile', {}).get('generation_seed')) is not None else '', 'RAG_BENCHMARK_SEEDS': '42',
        'RAG_BENCHMARK_CONCURRENCY': '8',
        'RAG_GENERATION_MODEL': reference['models']['default'],
        'RAG_EMBEDDING_MODEL': reference['models']['embedding'],
        'RAG_INDEX_NAMESPACE': stats['index_policy']['index_namespace'],
        'RAG_INDEX_STATS_PATH': reference['index_manifest_stats_path'],
        'RAG_RUN_ID': stats['run_id'],
        'RAG_BENCHMARK_TIMESTAMP': str(output.resolve().relative_to(ROOT / 'data/results')),
        'RAG_PREHOP_TRACE': 'true',
    })
    return env


def prepare(args):
    from models.prehop.ablation_inputs import write_input
    reference = json.loads(args.reference.read_text())
    rows = {row['query_id']: row for row in reference['details']}
    if not rows or reference['query_failure_count']:
        raise ValueError('Expected a successful primary result')
    queries = json.loads(args.queries.read_text())
    if {q['_id'] for q in queries} != rows.keys():
        raise ValueError('Query population differs from the primary result')
    events = Path(reference['details'][0]['prehop_trace']['events_path'])
    names = ('RetrieveMixin._retrieve_with_candidate_pool.start',
             'RetrieveMixin._retrieve_with_candidate_pool.result',
             'SimilarityScoringMixin._score_and_select.start',
             'SimilarityScoringMixin._role_body_list_ranking.start')
    indexed = {qid: {} for qid in rows}
    for line in events.open():
        event = json.loads(line)
        qid = event.get('identity', {}).get('query_id')
        if qid in indexed and event['event'] in names:
            if event['event'] in indexed[qid]:
                raise ValueError('Primary prefix contains repeated retrieval')
            indexed[qid][event['event']] = event
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = args.output / 'primary-inputs'
    checked = []
    for number, (qid, stages) in enumerate(sorted(indexed.items()), 1):
        start = payload(events.parent, stages[names[0]])
        returned = payload(events.parent, stages[names[1]])
        if start['select_final'] or start['top_k'] != 12 or len(returned) != 2:
            raise ValueError('Unexpected primary retrieval prefix')
        nodes = returned[1]
        if not start['query_embedding'] or len({n['id'] for n in nodes}) != len(nodes):
            raise ValueError('Invalid lossless primary prefix')
        write_input(inputs / 'occurrences', qid, {
            'query': start['query'], 'query_embedding': start['query_embedding'],
            'base_candidates': nodes, 'source_query_id': qid,
            'source_start_sha256': stages[names[0]]['payload_sha256'],
            'source_result_sha256': stages[names[1]]['payload_sha256'],
        })
        checked.append({'query_id': qid, 'query': start['query'], 'candidates': len(nodes),
                        'source_events': stages})
        if number % 250 == 0:
            print(f'Lossless primary prefixes: {number}/{len(rows)}', flush=True)
    manifest = {
        'contract': 'primary-hop-removal-v1', 'reference': str(args.reference.resolve()),
        'reference_sha256': digest(args.reference), 'source_events': str(events),
        'source_events_sha256': digest(events), 'queries': len(checked),
        'source_prompt_identity': reference['ablation']['prompt_configuration_sha256'],
        'source_generation_identity': reference['ablation']['generation_profiles_sha256'],
        'latency_scope': 'frozen_prefix_downstream_only',
        'changed_component': 'HOP traversal disabled; direct retrieval and NEXT retained',
        'rows': checked,
    }
    (args.output / 'prefix-manifest.json').write_text(json.dumps(manifest, indent=2))
    env = environment(reference, args.output, inputs)
    expansion = getattr(args, 'expansion', 'next_only')
    component, contract, changed = EXPANSIONS[expansion]
    env['RAG_GRAPH_EDGE_VARIANT'] = expansion
    env['RAG_PREHOP_ABLATION_PROFILE'] = 'primary_' + component
    manifest['contract'] = contract
    manifest['changed_component'] = changed
    (args.output / 'prefix-manifest.json').write_text(json.dumps(manifest, indent=2))
    command = [sys.executable, str(Path(__file__).resolve()), 'worker',
               '--reference', str(args.reference.resolve()), '--queries', str(args.queries.resolve()),
               '--output', str(args.output.resolve())]
    (args.output / 'plan.json').write_text(json.dumps({'environment': env, 'command': command}, indent=2))
    print(json.dumps({'prepared': len(checked), 'output': str(args.output)}, indent=2))


def configure(args):
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    for key in ('RAG_INDEX_REUSE_LINK', 'RAG_BENCHMARK_RESUME'):
        os.environ.pop(key, None)
    plan = json.loads((args.output / 'plan.json').read_text())
    os.environ.update(plan['environment'])
    return json.loads(args.reference.read_text())


async def verify(args):
    """Check preserved scoring and current NEXT edges against original traces."""
    reference = configure(args)
    from core.paper_compatibility import method_identity
    from models.prehop.retrieval.scoring import SimilarityScoringMixin
    manifest = json.loads((args.output / 'prefix-manifest.json').read_text())
    identity = method_identity('prehop')
    assert identity['prompt_configuration_sha256'] == manifest['source_prompt_identity']
    assert identity['generation_profiles_sha256'] == manifest['source_generation_identity']
    directory = Path(manifest['source_events']).parent
    from core.neo4j_service import Neo4jService
    stats = json.loads(Path(reference['index_manifest_stats_path']).read_text())
    label = 'PR_' + stats['index_policy']['index_namespace'] + '_Chunk'
    database = Neo4jService()
    try:
        next_rows = await database.execute_query(
            f'MATCH (a:{label})-[:NEXT]-(b:{label}) RETURN a.id AS source_id, b.id AS id')
    finally:
        await database.close()
    next_pairs = {(row['source_id'], row['id']) for row in next_rows}

    class Scorer(SimilarityScoringMixin):
        trace_recorder = None

        @staticmethod
        def _node_identity(node):
            return node['id']

        async def _role_body_list_ranking(self, *values, **kwargs):
            self.actual = kwargs or values
            return []

    checked = 0
    # Cover the entire population at regular intervals without any new inference.
    for row in manifest['rows'][::128]:
        source = payload(directory, row['source_events']['SimilarityScoringMixin._score_and_select.start'])
        expected = payload(directory, row['source_events']['SimilarityScoringMixin._role_body_list_ranking.start'])
        scorer = Scorer()
        await scorer._score_and_select(source['query_embedding'], source['candidates'],
                                       source['top_k'], query_text=source['query_text'])
        # The production method calls the ranker positionally.
        actual = scorer.actual
        actual_nodes = actual.get('ordered') if isinstance(actual, dict) else actual[1]
        expected_nodes = expected['ordered']
        assert actual_nodes == expected_nodes, 'Deterministic scoring/order differs from primary'
        prefix = payload(directory, row['source_events']['RetrieveMixin._retrieve_with_candidate_pool.result'])[1]
        starts = {node['id'] for node in prefix if not node.get('role_body_owner_only')}
        recorded_next = {(path['source_chunk_id'], node['id']) for node in source['candidates']
                         for path in node.get('retrieval_paths', []) if path['kind'] == 'next'}
        assert {(a, b) for a, b in next_pairs if a in starts} == recorded_next, 'NEXT graph differs from primary'
        checked += 1
    report = {'source_prefixes_verified': manifest['queries'], 'score_replay_queries': checked,
              'prompt_identity_equal': True, 'generation_identity_equal': True,
              'next_graph_replay_queries': checked,
              'new_llm_calls': 0, 'new_db_queries': 1,
              'generation_model_equal': os.environ['RAG_GENERATION_MODEL'] == reference['models']['default'],
              'generation_revision_equal': os.environ.get('RAG_GENERATION_REVISION') == reference['models']['generation_revision']}
    (args.output / 'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


async def worker(args):
    reference = configure(args)
    import core.prehop_ablation

    def component_identity(config=None):
        from core.config import RAGConfig
        config = config or RAGConfig
        return {'component_ablation': EXPANSIONS[config.GRAPH_EDGE_VARIANT][0],
                'method_contract': EXPANSIONS[config.GRAPH_EDGE_VARIANT][1],
                'primary_reference_sha256': digest(args.reference),
                'hop_seed_policy': config.HOP_SEED_POLICY, 'hop_link_variant': config.HOP_LINK_VARIANT,
                'comparison_scope': 'primary_component_ablation',
                'direct_inputs': os.environ['RAG_ABLATION_DIRECT_INPUTS'],
                'latency_scope': 'frozen_prefix_downstream_only'}

    # Replace only run-local provenance labeling; retrieval runtime is unchanged.
    core.prehop_ablation.ablation_identity = component_identity
    stats = json.loads(Path(reference['index_manifest_stats_path']).read_text())
    sys.argv = ['main.py', '--mode', 'benchmark', '--strategy', 'prehop',
                '--corpus-tag', stats['corpus_tag'], '--dataset', stats['dataset_path'],
                '--queries_file', str(args.queries.resolve())]
    import main as entrypoint
    await entrypoint.main()


def supervise(args):
    plan = json.loads((args.output / 'plan.json').read_text())
    status = args.output / 'status.json'
    started = datetime.now(UTC).isoformat()
    status.write_text(json.dumps({'state': 'running', 'pid': os.getpid(), 'started': started}))
    try:
        subprocess.run(plan['command'], cwd=ROOT, check=True)
        primary = json.loads(args.reference.read_text())
        tag = json.loads(Path(primary['index_manifest_stats_path']).read_text())['corpus_tag']
        result = args.output / f'prehop/{tag}/seed_42/prehop_{tag}.json'
        completed = json.loads(result.read_text())
        primary = json.loads(args.reference.read_text())
        if (completed.get('query_failure_count') or len(completed['details']) != len(primary['details'])
                or {r['query_id'] for r in completed['details']} != {r['query_id'] for r in primary['details']}):
            raise ValueError('Incomplete or failed queries: primary component comparison not ready')
        subprocess.run([sys.executable, str(ROOT / 'scripts/ablation_statistics.py'),
                        '--left', str(args.reference), '--right', str(result),
                        '--queries', str(args.queries), '--metrics', *comparison_metrics(tag),
                        '--output', str(args.output / ('primary-minus-' + EXPANSIONS[plan['environment']['RAG_GRAPH_EDGE_VARIANT']][0].replace('_', '-') + '.json'))],
                       cwd=ROOT, check=True)
        if getattr(args, 'next_reference', None):
            subprocess.run([sys.executable, str(ROOT / 'scripts/ablation_statistics.py'),
                            '--left', str(args.next_reference), '--right', str(result),
                            '--queries', str(args.queries), '--metrics', *comparison_metrics(tag),
                            '--output', str(args.output / ('next-minus-' + EXPANSIONS[plan['environment']['RAG_GRAPH_EDGE_VARIANT']][0].replace('_', '-') + '.json'))],
                           cwd=ROOT, check=True)
        status.write_text(json.dumps({'state': 'completed', 'result': str(result), 'started': started,
                                     'completed': datetime.now(UTC).isoformat()}))
    except BaseException as error:
        status.write_text(json.dumps({'state': 'failed', 'error': repr(error)}))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'verify', 'worker', 'supervise'))
    parser.add_argument('--next-reference', type=Path)
    parser.add_argument('--expansion', choices=tuple(EXPANSIONS), default='next_only')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args)
    elif args.mode == 'verify':
        asyncio.run(verify(args))
    elif args.mode == 'worker':
        asyncio.run(worker(args))
    else:
        supervise(args)


if __name__ == '__main__':
    main()
