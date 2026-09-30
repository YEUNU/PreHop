"""Compare frozen Prehop and direct selections with a common reader.

Selection outputs are frozen and audited, not regenerated or chosen by quality.
Reader requests preserve every selected passage and use the shared transport.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from scripts.campaign_runtime import atomic_json, close_owned, drain


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def observed_prompt_tokens(record):
    """Measure one selector input, separately from total usage across retries."""
    trace = record.get('trace')
    if trace:
        if sha(trace['path']) != trace['sha256']:
            raise ValueError('Committed selector trace changed')
        for attempt in reversed(read_json(trace['path'])['attempts']):
            tokens = (attempt.get('usage') or {}).get('prompt_tokens')
            if isinstance(tokens, int) and not isinstance(tokens, bool):
                return tokens
    usage = record['usage']
    if usage['token_usage_complete'] and usage['generation_calls'] == 1:
        return usage['prompt_tokens']
    return None


ARMS = ('prehop_replay', 'direct_only')
FIXED_START = 'fixed-start-common-reader-v1'


def comparison_arms(protocol):
    arms = tuple(protocol.get('arms', ARMS))
    if arms != ARMS:
        raise ValueError('Only fixed-start expansion QA is supported')
    return arms


def auxiliary_rows(rows, dataset):
    # MultiHop-RAG null questions have no answer EM/F1; official QA still includes them.
    return [r for r in rows if dataset != 'multihoprag' or r['question_type'] != 'null_query']


def verify_sources(protocol):
    for path, digest in protocol['source_hashes'].items():
        if sha(path) != digest:
            raise ValueError(f'Experiment source changed: {path}')


def initialize_fixed_start(reference_path, direct_path, queries_path, output, profile_path):
    """Specify the QA follow-up before generation, keeping original selections."""
    from utils.official_results import dataset_key

    reference, direct = read_json(reference_path), read_json(direct_path)
    if any(dataset_key(r['dataset']) != 'multihoprag' for r in (reference, direct)):
        raise ValueError('Fixed-start QA currently requires MultiHop-RAG')
    if (direct['ablation'].get('component_ablation') != 'direct_only'
            or direct['ablation'].get('primary_reference_sha256') != sha(reference_path)):
        raise ValueError('Direct-only result does not identify this primary reference')
    for key in ('index_manifest_stats_sha256', 'models'):
        if reference[key] != direct[key]:
            raise ValueError(f'Fixed-start conditions differ in {key}')
    for key, value in reference['ablation'].items():
        if direct['ablation'].get(key) != value:
            raise ValueError(f'Fixed-start conditions differ in {key}')
    prefix_path = Path(direct['ablation']['direct_inputs']).parent / 'prefix-manifest.json'
    manifest = read_json(prefix_path)
    if (manifest['reference_sha256'] != sha(reference_path)
            or manifest['source_events_sha256'] != sha(manifest['source_events'])):
        raise ValueError('Primary prefix identity differs')
    direct_events = sorted({r['prehop_trace']['events_path'] for r in direct['details']})
    sources = [reference_path, direct_path, queries_path, prefix_path,
               Path(reference['index_manifest_stats_path']), Path(manifest['source_events']),
               *(Path(p) for p in direct_events)]
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(profile_path, output / 'execution-profile.json')
    atomic_json(output / 'protocol.json', {
        'experiment': FIXED_START, 'created_at': time.time(),
        'reference': str(reference_path.resolve()), 'queries': str(queries_path.resolve()),
        'frozen_results': {'prehop_replay': str(reference_path.resolve()), 'direct_only': str(direct_path.resolve())},
        'prefix_manifest': str(prefix_path.resolve()), 'direct_events': direct_events,
        'source_hashes': {str(p.resolve()): sha(p) for p in sources},
        'arms': ['prehop_replay', 'direct_only'], 'dataset': 'multihoprag',
        'analysis_status': 'QA follow-up after observing fixed-start retrieval and both-dataset budget controls; '
                           'historical QA is known. No claim of blind replication.',
        'hypothesis': 'Determine whether one-step HOP/NEXT expansion from fixed direct inputs improves common-reader QA.',
        'fixed': ['corpus/index', 'original query and query embedding', 'direct passage identities and scores',
                  'selector settings', 'final selection limit'],
        'selection': 'Reuse every original selected passage and its order in both completed conditions; no reruns.',
        'reader': 'Same common-reader prompt/model, temperature 0, max 256 output tokens, thinking disabled; '
                  'all selected passages, no truncation, retrieval, selection, gold or previous answers in requests.',
        'population': {'qa': len(reference['details']),
                       'original_questions': len({r['original_query_id'] for r in reference['details']}),
                       'null_qa_rows': sum(r['question_type'] == 'null_query' for r in reference['details'])},
        'primary_metric': 'official_qa_accuracy (official any-token overlap; not exact match)',
        'primary_contrast': 'prehop_replay minus direct_only',
        'secondary_metrics': ['answer_em', 'answer_f1', 'QA by question type', 'reader tokens', 'failures'],
        'uncertainty': 'Paired original-question bootstrap, 10000 resamples, seed 42; conditional on the saved '
                       'index/selections and new reader outputs.',
        'reader_order_seed': 42, 'workers': 32,
        'segments': [{'scope': 'first eight queries, both arms; retained canary', 'workers': 8},
                     {'scope': 'all remaining queries, both arms interleaved', 'workers': 32}],
        'failure_policy': 'Terminal failures remain committed and score zero; no quality-based regeneration.',
        'timing_scope': 'Reader-only; candidate/selector budgets differ, no end-to-end latency comparison.',
        'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'dirty_status': subprocess.check_output(['git', 'status', '--short'], text=True),
    })


def prepare_fixed_start(protocol, output):
    """Audit the executed direct inputs, then build gold-free reader contexts."""
    from core.synthesis_replay import returned_passage_context
    from models.prehop.graphrag import GraphRAG
    from scripts.run_primary_hop_ablation import payload

    verify_sources(protocol)
    manifest = read_json(protocol['prefix_manifest'])
    prefixes = {r['query_id']: r for r in manifest['rows']}
    source_files = {arm: read_json(path) for arm, path in protocol['frozen_results'].items()}
    sources = {arm: {r['query_id']: r for r in data['details']} for arm, data in source_files.items()}
    arms = comparison_arms(protocol)
    queries = read_json(protocol['queries'])
    ids = {q['_id'] for q in queries}
    if len(ids) != len(queries) or ids != set(prefixes) or len(prefixes) != len(manifest['rows']):
        raise ValueError('Frozen-prefix population mismatch')
    for arm in arms:
        if set(sources[arm]) != ids or len(sources[arm]) != len(source_files[arm]['details']):
            raise ValueError('Frozen-result population mismatch')
    direct_scores = {}
    for event_path in protocol['direct_events']:
        with Path(event_path).open() as stream:
            for line in stream:
                event = json.loads(line)
                qid = event.get('identity', {}).get('query_id')
                if qid in ids and event['event'] == 'SimilarityScoringMixin._score_and_select.start':
                    if qid in direct_scores:
                        raise ValueError('Repeated Direct-only scoring input')
                    direct_scores[qid] = (Path(event_path).parent, event)
    if set(direct_scores) != ids:
        raise ValueError('Missing executed Direct-only inputs')
    records, audits = {arm: [] for arm in arms}, []
    directory = Path(manifest['source_events']).parent
    annotation_fields = ('query_id', 'original_query_id', 'query', 'question_type', 'ground_truth',
                         'expected_sources', 'answer_aliases')
    for number, primary in enumerate(source_files[arms[0]]['details'], 1):
        qid = primary['query_id']
        secondary = sources[arms[1]][qid]
        if any(primary.get(k) != secondary.get(k) for k in annotation_fields):
            raise ValueError('Paired question or annotation mismatch')
        events = prefixes[qid]['source_events']
        start = payload(directory, events['RetrieveMixin._retrieve_with_candidate_pool.start'])
        returned = payload(directory, events['RetrieveMixin._retrieve_with_candidate_pool.result'])
        score_dir, score_event = direct_scores[qid]
        scored = payload(score_dir, score_event)
        query = GraphRAG._strip_format_instruction(primary['query'])
        normalized_query = GraphRAG._normalize_entity_term(query) or query.strip()
        if (scored['candidates'] != returned[1] or scored['query_embedding'] != start['query_embedding']
                or scored['query_text'] != start['query'] or scored['query_text'] != normalized_query
                or scored['top_k'] != start['top_k']):
            raise ValueError(f'Executed Direct-only input differs from the original search: {qid}')
        conditions = {}
        for arm in arms:
            row = sources[arm][qid]
            selected = row['retrieved_sources']
            conditions[arm] = {'context': returned_passage_context(selected), 'selection_error': row.get('error'),
                               'selection_source': protocol['frozen_results'][arm]}
            records[arm].append({**{k: row[k] for k in annotation_fields if k in row},
                                 'retrieved_sources': selected, 'error': row.get('error')})
        atomic_json(output / 'inputs' / f'{qid}.json',
                    {'query_id': qid, 'query': primary['query'], 'conditions': conditions})
        audits.append({'query_id': qid, 'direct_candidates': len(returned[1]),
                       'primary_start_sha256': events['RetrieveMixin._retrieve_with_candidate_pool.start']['payload_sha256'],
                       'primary_return_sha256': events['RetrieveMixin._retrieve_with_candidate_pool.result']['payload_sha256'],
                       'direct_scoring_sha256': score_event['payload_sha256']})
        if number % 100 == 0:
            print(f'Audited fixed-start query pairs: {number}/{len(ids)}', flush=True)
    for arm in arms:
        atomic_json(output / 'sources' / f'{arm}.json', {
            'dataset': 'multihoprag', 'strategy': arm, 'dataset_protocol': source_files[arm]['dataset_protocol'],
            'status': 'completed', 'evaluation_scope': 'full_benchmark',
            'total_queries': len(records[arm]), 'details': records[arm],
        })
    atomic_json(output / 'input-audit.json', {'rows': len(audits), 'details': audits})


async def generate(protocol, output, workers, query_limit=None):
    from cli.synthesis import reader_settings
    from core import inference_telemetry
    from core.inference_transport import InferenceTransport
    from core.structured_diagnostics import json_sha256, structured_request_budget
    from core.vllm_client import VLLMClient
    from utils.parsers import clean_and_unwrap_json
    from utils.prompts.prehop_answer import build_answer_messages

    rows = read_json(protocol['reference'])['details']
    if query_limit:
        rows = rows[:query_limit]
    jobs = [(row['query_id'], arm) for row in rows for arm in comparison_arms(protocol)]
    random.Random(protocol['reader_order_seed']).shuffle(jobs)
    client = VLLMClient()
    transport = InferenceTransport.resolve('prehop')
    semaphore = asyncio.Semaphore(workers)
    completed = 0

    async def one(qid, arm):
        nonlocal completed
        async with semaphore:
            source_path = output / 'inputs' / f'{qid}.json'
            source = read_json(source_path)
            condition = source['conditions'][arm]
            params = {'model': client.model_name,
                      'messages': build_answer_messages(condition['context'], source['query']), **reader_settings()}
            digest = json_sha256(params)
            target = output / 'reader_calls' / arm / f'{qid}.json'
            if target.exists():
                record = read_json(target)
                if (record['input_sha256'] != sha(source_path) or record['request_sha256'] != digest
                        or record['trace']['sha256'] != sha(record['trace']['path'])):
                    raise ValueError('Committed reader input or trace changed')
            else:
                record = {'query_id': qid, 'condition': arm, 'model': client.model_name,
                          'input_sha256': sha(source_path), 'request_sha256': digest,
                          'reader_transport': transport.policy_dict()}
                trace = {'query_id': qid, 'condition': arm, 'request': params}
                started = time.perf_counter()
                token = inference_telemetry.begin()
                try:
                    if condition['selection_error']:
                        raise ValueError('Upstream selection failed')
                    # Use the common transport directly: never invoke heuristic message truncation.
                    # One shared attempt budget also disables hidden SDK retries and records failed usage.
                    with structured_request_budget(transport.retry_attempts):
                        response = await client._create_generation_request(client.client, params)
                    trace['response'] = response.model_dump(mode='json')
                    answer = client.think_strip(clean_and_unwrap_json(response.choices[0].message.content or ''))
                    record.update(status='completed', answer=str(answer or ''))
                except Exception as exc:  # noqa: BLE001 -- terminal failures retain their population weight
                    record.update(status='failed', error=type(exc).__name__, answer='')
                finally:
                    record['reader_seconds'] = time.perf_counter() - started
                    record['usage'] = inference_telemetry.finish(token)
                trace_path = output / 'reader_traces' / arm / f'{qid}.json'
                await asyncio.to_thread(atomic_json, trace_path, trace)
                record['trace'] = {'path': str(trace_path), 'sha256': sha(trace_path)}
                await asyncio.to_thread(atomic_json, target, record)
            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                atomic_json(output / 'reader-progress.json', {
                    'completed': completed, 'total': len(jobs), 'updated_at': time.time(),
                })
                print(f'Reader calls committed: {completed}/{len(jobs)}', flush=True)

    try:
        await drain([asyncio.create_task(one(*job)) for job in jobs])
    finally:
        await close_owned(client.global_close, primary_error=sys.exc_info()[1])
    return sum(read_json(output / 'reader_calls' / arm / f'{qid}.json')['status'] != 'completed'
               for qid, arm in jobs)


async def evaluate(protocol, output):
    from core.structured_diagnostics import json_sha256
    from core.synthesis_replay import returned_passage_context
    from scripts.ablation_statistics import cluster_interval
    from scripts.export_official_results import export_comparison
    from utils.metrics import evaluate_multihoprag_response
    from utils.official_results import build_reports, dataset_key

    verify_sources(protocol)
    if protocol.get('experiment') != FIXED_START:
        raise ValueError('Expected the fixed-start common-reader protocol')
    arms = comparison_arms(protocol)
    results, summaries = {}, {}
    dataset = None
    for arm in arms:
        source = read_json(output / 'sources' / f'{arm}.json')
        current_dataset = dataset_key(source['dataset'])
        if current_dataset != 'multihoprag':
            raise ValueError('Fixed-start QA requires MultiHop-RAG')
        if dataset is not None and current_dataset != dataset:
            raise ValueError('Paired arms use different datasets')
        dataset = current_dataset
        rows, calls, reader_tokens = [], [], []
        for original in source['details']:
            qid = original['query_id']
            call = read_json(output / 'reader_calls' / arm / f'{qid}.json')
            prepared = read_json(output / 'inputs' / f'{qid}.json')
            trace = read_json(call['trace']['path'])
            prompt_tokens = ((trace.get('response') or {}).get('usage') or {}).get('prompt_tokens')
            if type(prompt_tokens) is int:
                reader_tokens.append(prompt_tokens)
            if (call['input_sha256'] != sha(output / 'inputs' / f'{qid}.json')
                    or call['trace']['sha256'] != sha(call['trace']['path'])
                    or call['request_sha256'] != json_sha256(trace['request'])
                    or prepared['conditions'][arm]['context'] != returned_passage_context(original['retrieved_sources'])
                    or prepared['query'] != original['query']):
                raise ValueError('Reader result integrity mismatch')
            expected = original['expected_sources']
            metrics = await evaluate_multihoprag_response(
                query=original['query'], response=call['answer'], ground_truth=original['ground_truth'],
                retrieved_sources=original['retrieved_sources'], evidence_facts=expected['facts'],
                evidence_docs=expected.get('docs', []), question_type=original['question_type'],
                dataset=dataset, answer_aliases=original.get('answer_aliases', []),
                supporting_facts=expected.get('supporting_facts'),
                hotpot_sentence_store=protocol.get('sentence_store'),
            )
            error = original.get('error') or call.get('error')
            if error:
                from core.benchmark_failures import QUALITY_METRICS
                metrics.update({key: 0.0 for key in QUALITY_METRICS})
            rows.append({**original, **metrics, 'answer': call['answer'], 'error': error,
                         'reader_status': call['status'], 'reader_seconds': call['reader_seconds']})
            calls.append(call)
        result = {**source, 'details': rows, 'common_reader': {
            'context_policy': 'all selected passages in recorded order; no truncation',
            'protocol': str(output / 'protocol.json'), 'protocol_sha256': sha(output / 'protocol.json'),
            'model': calls[0]['model'], 'reader_trace_directory': str(output / 'reader_traces' / arm),
        }}
        atomic_json(output / f'{arm}.json', result)
        official, _ = build_reports(result)
        auxiliary = auxiliary_rows(rows, dataset)
        summaries[arm] = {
            'qa': official['qa'],
            'failed_reader': sum(c['status'] != 'completed' for c in calls),
            'auxiliary_answer_em': statistics.mean(r['answer_em'] for r in auxiliary) if auxiliary else None,
            'auxiliary_answer_f1': statistics.mean(r['answer_f1'] for r in auxiliary) if auxiliary else None,
            'auxiliary_answer_rows': len(auxiliary),
            'retrieval_from_saved_selection': official['retrieval'],
            'reader_generation_attempts': sum(c['usage']['generation_calls'] for c in calls),
            'reader_incomplete_usage_rows': sum(not c['usage']['token_usage_complete'] for c in calls),
            'reader_observed_prompt_tokens_mean': statistics.mean(reader_tokens) if reader_tokens else None,
            'reader_observed_prompt_token_rows': len(reader_tokens),
        }
        results[arm] = {r['query_id']: r for r in rows}
    left, right = (results[arm] for arm in arms)
    if set(left) != set(right):
        raise ValueError('QA pairing mismatch')
    contrasts = {}
    fields = ['official_qa_accuracy', 'answer_em', 'answer_f1']

    def metric_rows(rows, field):
        return auxiliary_rows(rows, dataset) if field in {'answer_em', 'answer_f1'} else rows

    def paired_metric(field, qids):
        eligible = metric_rows([left[qid] for qid in qids], field)
        return cluster_interval([r[field] - right[r['query_id']][field] for r in eligible],
                                [r['original_query_id'] for r in eligible]) if eligible else None

    for field in fields:
        contrasts[field] = paired_metric(field, left)
    report = {'conditions': summaries, 'prehop_minus_direct': contrasts,
              'retrieval_contrast': None,
              'matching': {'fixed_direct_inputs': True, 'selector_tokens_matched': False}, 'protocol_sha256': sha(output / 'protocol.json')}
    report['by_question_type'] = {}
    for kind in sorted({row['question_type'] for row in left.values()}):
        qids = [qid for qid in left if left[qid]['question_type'] == kind]
        eligible_by_field = {field: [r['query_id'] for r in metric_rows([left[qid] for qid in qids], field)]
                             for field in fields}
        report['by_question_type'][kind] = {
            'conditions': {arm: {field: statistics.mean(results[arm][qid][field]
                                                      for qid in eligible_by_field[field])
                                        if eligible_by_field[field] else None
                                  for field in fields} for arm in arms},
            'prehop_minus_direct': {field: paired_metric(field, qids) for field in fields},
        }
    atomic_json(output / 'comparison.json', report)
    export_comparison([output / f'{arm}.json' for arm in arms], output / 'official')
    print({arm: summaries[arm]['qa'] for arm in arms}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['init-fixed-start', 'prepare', 'canary', 'generate',
                                          'evaluate', 'supervise'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--direct-reference', type=Path)
    parser.add_argument('--queries', type=Path)
    parser.add_argument('--execution-profile', type=Path)
    parser.add_argument('--stages', nargs='+', default=['prepare', 'canary', 'generate', 'evaluate'])
    args = parser.parse_args()
    from scripts.runner_environment import _load_runner_environment
    _load_runner_environment()
    if args.stage == 'init-fixed-start':
        if not all((args.reference, args.direct_reference, args.queries, args.execution_profile)):
            parser.error('init-fixed-start requires --reference, --direct-reference, --queries, --execution-profile')
        initialize_fixed_start(args.reference.resolve(), args.direct_reference.resolve(), args.queries.resolve(),
                               args.output.resolve(), args.execution_profile.resolve())
        return
    output = args.output.resolve()
    protocol = read_json(output / 'protocol.json')
    if args.stage == 'supervise':
        from scripts.campaign_runtime import lock, resource_lock_path, run_child
        from scripts.paper_detached_runtime import register_owner
        owner = register_owner(output / 'protocol.json')
        status = {'supervisor': owner, 'started_at': time.time(), 'state': 'running'}

        def update(values):
            status.update(values)
            atomic_json(output / 'supervisor-status.json', status)

        with lock(resource_lock_path()) as handle:
            for stage in args.stages:
                update({'stage': stage})
                argv = [sys.executable, '-u', '-m', 'scripts.compare_prehop_direct_qa', stage,
                        '--output', str(output)]
                code = run_child(argv, os.environ.copy(), output / f'{time.time_ns()}-{stage}', handle, update)
                if code:
                    update({'state': 'failed', 'exit_code': code, 'finished_at': time.time()})
                    raise SystemExit(code)
            update({'state': 'completed', 'finished_at': time.time()})
        return
    from scripts.run_primary_hop_ablation import environment
    env = environment(read_json(protocol['reference']), output, output / 'inputs')
    env.update(RAG_RUN_ID=output.name, RAG_PREHOP_TRACE='false', RAG_ABLATION_DIRECT_INPUTS='',
               RAG_EXECUTION_PROFILE=str(output / 'execution-profile.json'))
    os.environ.update(env)
    from core.inference_transport import InferenceTransport
    atomic_json(output / f'phase-{args.stage}-{time.time_ns()}.json', {
        'stage': args.stage, 'code_sha256': sha(__file__),
        'transport': InferenceTransport.resolve('prehop').policy_dict(),
        'source_hashes': {path: sha(path) for path in [
            'cli/synthesis.py', 'utils/prompts/prehop_answer.py', 'core/vllm_client.py',
            'utils/metrics.py', 'scripts/campaign_runtime.py', 'core/generation_profiles.py',
        ]}, 'served_model_revision': 'not independently verified',
    })
    if args.stage == 'prepare':
        prepare_fixed_start(protocol, output)
    elif args.stage in {'generate', 'canary'}:
        failures = asyncio.run(generate(protocol, output, 8 if args.stage == 'canary' else protocol['workers'],
                                        8 if args.stage == 'canary' else None))
        if failures and args.stage == 'canary':
            raise SystemExit(f'{failures} terminal canary failures retained; inspect before continuing')
    else:
        asyncio.run(evaluate(protocol, output))


if __name__ == '__main__':
    main()
