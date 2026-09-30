"""Reader follow-ups retain evidence, failures, and exact request identities."""
from types import SimpleNamespace

import pytest

from scripts import compare_prehop_direct_qa as qa


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [False, True])
@pytest.mark.parametrize('arms', [qa.ARMS])
async def test_reader_preserves_context_and_resumes_terminal_rows(tmp_path, monkeypatch, failure, arms):
    from core import inference_telemetry
    from core.vllm_client import VLLMClient
    from utils.prompts.prehop_answer import build_answer_messages

    reference = tmp_path / 'reference.json'
    qa.atomic_json(reference, {'details': [{'query_id': 'q', 'ground_truth': 'SECRET_GOLD'}]})
    context = '[[Article, Page 0, Chunk 0]]\nComplete passage.\n\nSecond complete passage.'
    qa.atomic_json(tmp_path / 'inputs/q.json', {
        'query_id': 'q', 'query': 'Question?',
        'conditions': {arm: {'context': context, 'selection_error': None} for arm in arms},
    })
    protocol = {'reference': str(reference), 'reader_order_seed': 42, 'arms': list(arms)}
    requests = []

    async def request(self, client, params):
        assert params['messages'] == build_answer_messages(context, 'Question?')
        assert 'SECRET_GOLD' not in str(params)
        assert params['max_tokens'] == 256
        requests.append(params)
        if failure:
            inference_telemetry.record_generation_transport_failure({'usage': None})
            raise OSError('terminal transport failure')
        inference_telemetry.record('generation', SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5, total_tokens=105),
        ))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='answer'))],
                               model_dump=lambda **kwargs: {'choices': [{'message': {'content': 'answer'}}]})

    monkeypatch.setattr(VLLMClient, '_create_generation_request', request)
    monkeypatch.setattr(VLLMClient, 'client', property(lambda self: None))
    assert await qa.generate(protocol, tmp_path, 2) == (2 if failure else 0)
    assert len(requests) == 2
    assert await qa.generate(protocol, tmp_path, 2) == (2 if failure else 0)
    assert len(requests) == 2
    row = qa.read_json(tmp_path / 'reader_calls/prehop_replay/q.json')
    assert row['status'] == ('failed' if failure else 'completed')
    assert row['usage']['token_usage_complete'] is not failure
    (tmp_path / 'reader_traces/prehop_replay/q.json').write_text('{}')
    with pytest.raises(ValueError, match='trace changed'):
        await qa.generate(protocol, tmp_path, 2)


@pytest.mark.asyncio
async def test_failed_trace_save_does_not_commit_success(tmp_path, monkeypatch):
    from core.vllm_client import VLLMClient

    reference = tmp_path / 'reference.json'
    qa.atomic_json(reference, {'details': [{'query_id': 'q'}]})
    qa.atomic_json(tmp_path / 'inputs/q.json', {
        'query': 'Question?', 'conditions': {arm: {'context': 'Evidence', 'selection_error': 'upstream'}
                                            for arm in qa.ARMS},
    })
    real_save = qa.atomic_json

    def save(path, value):
        if 'reader_traces' in str(path):
            raise OSError('disk failure')
        return real_save(path, value)

    monkeypatch.setattr(qa, 'atomic_json', save)
    monkeypatch.setattr(VLLMClient, 'client', property(lambda self: None))
    with pytest.raises(OSError, match='disk failure'):
        await qa.generate({'reference': str(reference), 'reader_order_seed': 42}, tmp_path, 2)
    assert not list((tmp_path / 'reader_calls').glob('*/*.json'))


@pytest.mark.asyncio
@pytest.mark.parametrize('fixed_start', [True])
@pytest.mark.parametrize('failed_null', [False, True])
async def test_evaluation_keeps_failed_and_null_questions_in_qa(tmp_path, fixed_start, failed_null):
    from core.structured_diagnostics import json_sha256
    from core.synthesis_replay import returned_passage_context

    previous = tmp_path / 'previous'
    qa.atomic_json(previous / 'comparison.json', {
        'conditions': {arm: {} for arm in qa.ARMS}, 'matching': {},
        'contrasts': {'prehop_replay_minus_direct_tokens': {}},
    })
    arms = ('prehop_replay', 'direct_only') if fixed_start else qa.ARMS
    protocol = {'source_hashes': {}, 'source_experiment': str(previous), 'arms': list(arms)}
    if fixed_start:
        protocol['experiment'] = qa.FIXED_START
        protocol.pop('source_experiment')
    qa.atomic_json(tmp_path / 'protocol.json', protocol)
    originals = []
    for qid in ['first', 'second', 'null']:
        null = qid == 'null'
        node = {'text': 'Alpha', 'title': 'Article'}
        original = {'query_id': qid, 'original_query_id': qid, 'query': qid + '?',
                    'question_type': 'null_query' if null else 'comparison_query',
                    'ground_truth': 'Insufficient information' if null else 'Alpha',
                    'expected_sources': {'facts': [] if null else ['Alpha']},
                    'retrieved_sources': [node], 'error': None}
        originals.append(original)
        qa.atomic_json(tmp_path / 'inputs' / f'{qid}.json', {
            'query': original['query'], 'conditions': {arm: {'context': returned_passage_context([node])}
                                                     for arm in arms},
        })
        for arm in arms:
            failed = (qid == 'first' or (null and failed_null)) and arm == 'prehop_replay'
            trace_path = tmp_path / 'reader_traces' / arm / f'{qid}.json'
            qa.atomic_json(trace_path, {'request': {'fixture': True}, 'response': {'usage': None}})
            qa.atomic_json(tmp_path / 'reader_calls' / arm / f'{qid}.json', {
                'query_id': qid, 'model': 'test-model', 'status': 'failed' if failed else 'completed',
                'answer': '' if failed else original['ground_truth'], 'error': 'transport' if failed else None,
                'input_sha256': qa.sha(tmp_path / 'inputs' / f'{qid}.json'),
                'request_sha256': json_sha256({'fixture': True}),
                'trace': {'path': str(trace_path), 'sha256': qa.sha(trace_path)}, 'reader_seconds': 1,
                'usage': {'generation_calls': 1, 'token_usage_complete': False},
            })
    for arm in arms:
        qa.atomic_json(tmp_path / 'sources' / f'{arm}.json', {
            'dataset': 'multihoprag', 'dataset_protocol': 'fixture', 'strategy': arm,
            'status': 'completed', 'evaluation_scope': 'full_benchmark', 'total_queries': 3, 'details': originals,
        })
    await qa.evaluate(protocol, tmp_path)
    report = qa.read_json(tmp_path / 'comparison.json')
    assert report['conditions']['prehop_replay']['qa']['overall']['rows'] == 3
    assert report['conditions']['prehop_replay']['qa']['overall']['accuracy'] == pytest.approx((2 - failed_null) / 3)
    assert report['conditions']['prehop_replay']['failed_reader'] == 1 + failed_null
    assert report['conditions']['prehop_replay']['reader_observed_prompt_tokens_mean'] is None
    assert report['prehop_minus_direct']['official_qa_accuracy']['mean'] == pytest.approx(-(1 + failed_null) / 3)
    assert report['conditions']['prehop_replay']['auxiliary_answer_rows'] == 2
    assert report['conditions']['prehop_replay']['auxiliary_answer_em'] == .5
    assert report['conditions']['prehop_replay']['auxiliary_answer_f1'] == .5
    assert report['prehop_minus_direct']['answer_em']['mean'] == -.5
    assert report['prehop_minus_direct']['answer_em']['rows'] == 2
    assert report['prehop_minus_direct']['answer_f1']['mean'] == -.5
    result = qa.read_json(tmp_path / 'prehop_replay.json')
    assert result['details'][0]['answer_em'] == 0
    assert result['details'][0]['reader_status'] == 'failed'
    if fixed_start:
        assert report['matching'] == {'fixed_direct_inputs': True, 'selector_tokens_matched': False}
        null = report['by_question_type']['null_query']
        assert null['prehop_minus_direct']['official_qa_accuracy']['mean'] == -int(failed_null)
        assert null['prehop_minus_direct']['answer_em'] is None
        assert null['prehop_minus_direct']['answer_f1'] is None
        assert null['conditions']['prehop_replay']['answer_em'] is None
        assert null['conditions']['prehop_replay']['answer_f1'] is None


@pytest.fixture
def fixed_start_sources(tmp_path):
    import gzip
    import hashlib
    import json

    def event(directory, name, qid, value):
        raw = json.dumps(value).encode()
        digest = hashlib.sha256(raw).hexdigest()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'{digest}.json.gz').write_bytes(gzip.compress(raw))
        return {'event': name, 'identity': {'query_id': qid},
                'payload': f'{digest}.json.gz', 'payload_sha256': digest}

    start_dir, direct_dir = tmp_path / 'primary-traces', tmp_path / 'direct-traces'
    rows, direct_rows, prefixes, starts, scores = [], [], [], [], []
    for qid in ['q1', 'null']:
        node = {'id': qid + '-direct', 'title': 'Title', 'text': 'Full direct passage.', 'rank_score': 1.0}
        extra = {'id': qid + '-hop', 'title': 'Other title', 'text': 'Full expanded passage.'}
        start = {'query': qid + '?', 'query_embedding': [1.0, 0.0], 'top_k': 12}
        source_events = {
            name: event(start_dir, name, qid, value) for name, value in [
                ('RetrieveMixin._retrieve_with_candidate_pool.start', start),
                ('RetrieveMixin._retrieve_with_candidate_pool.result', [[], [node]]),
            ]
        }
        starts.extend(source_events.values())
        prefixes.append({'query_id': qid, 'source_events': source_events})
        scores.append(event(direct_dir, 'SimilarityScoringMixin._score_and_select.start', qid, {
            'query_text': start['query'], 'query_embedding': start['query_embedding'],
            'top_k': 12, 'candidates': [node],
        }))
        row = {'query_id': qid, 'original_query_id': qid, 'query': start['query'].upper(),
               'question_type': 'null_query' if qid == 'null' else 'comparison_query',
               'ground_truth': 'SECRET_GOLD', 'expected_sources': {'facts': []},
               'answer': 'OLD_ANSWER', 'retrieved_sources': [extra, node],
               'prehop_trace': {'events_path': str(start_dir / 'events.jsonl')}}
        rows.append(row)
        direct_rows.append({**row, 'retrieved_sources': [node],
                            'prehop_trace': {'events_path': str(direct_dir / 'events.jsonl')}})
    for directory, events in [(start_dir, starts), (direct_dir, scores)]:
        (directory / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    index = tmp_path / 'index.json'
    qa.atomic_json(index, {'fixture': True})
    reference = tmp_path / 'reference.json'
    common = {'dataset': 'multihoprag', 'dataset_protocol': None, 'models': {'default': 'fixture-model'},
              'index_manifest_stats_path': str(index), 'index_manifest_stats_sha256': qa.sha(index),
              'ablation': {'graph_path_decay': 0.5}}
    qa.atomic_json(reference, {**common, 'details': rows})
    direct = tmp_path / 'direct.json'
    qa.atomic_json(direct, {**common, 'details': direct_rows, 'ablation': {
        **common['ablation'], 'component_ablation': 'direct_only',
        'primary_reference_sha256': qa.sha(reference), 'direct_inputs': str(tmp_path / 'primary-inputs'),
    }})
    qa.atomic_json(tmp_path / 'prefix-manifest.json', {
        'reference_sha256': qa.sha(reference), 'source_events': str(start_dir / 'events.jsonl'),
        'source_events_sha256': qa.sha(start_dir / 'events.jsonl'), 'rows': prefixes,
    })
    queries, profile, output = tmp_path / 'queries.json', tmp_path / 'profile.json', tmp_path / 'output'
    qa.atomic_json(queries, [{'_id': qid} for qid in ['q1', 'null']])
    qa.atomic_json(profile, {'fixture': True})
    return reference, direct, queries, output, profile


def test_fixed_start_preparation_checks_executed_inputs_and_preserves_context(fixed_start_sources):
    from core.synthesis_replay import returned_passage_context

    qa.initialize_fixed_start(*fixed_start_sources)
    reference, direct, _, output, _ = fixed_start_sources
    protocol = qa.read_json(output / 'protocol.json')
    qa.prepare_fixed_start(protocol, output)
    assert qa.read_json(output / 'input-audit.json')['rows'] == 2
    for arm, path in [('prehop_replay', reference), ('direct_only', direct)]:
        original = qa.read_json(path)['details'][0]
        prepared = qa.read_json(output / 'inputs/q1.json')['conditions'][arm]
        assert prepared['context'] == returned_passage_context(original['retrieved_sources'])
        assert 'SECRET_GOLD' not in prepared['context'] and 'OLD_ANSWER' not in prepared['context']
        saved = qa.read_json(output / 'sources' / f'{arm}.json')['details']
        assert len(saved) == 2 and saved[1]['question_type'] == 'null_query'
        assert 'answer' not in saved[0]


@pytest.mark.parametrize('change', ['embedding', 'score', 'annotation', 'missing_question'])
def test_fixed_start_rejects_changed_control(fixed_start_sources, change):
    import gzip
    import hashlib
    import json

    _, direct, _, output, _ = fixed_start_sources
    if change in {'embedding', 'score'}:
        events_path = direct.parent / 'direct-traces/events.jsonl'
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        e = events[0]
        data = json.loads(gzip.decompress((events_path.parent / e['payload']).read_bytes()))
        if change == 'embedding':
            data['query_embedding'] = [0.0, 1.0]
        else:
            data['candidates'][0]['rank_score'] = 0.9
        raw = json.dumps(data).encode()
        e['payload_sha256'] = hashlib.sha256(raw).hexdigest()
        e['payload'] = e['payload_sha256'] + '.json.gz'
        (events_path.parent / e['payload']).write_bytes(gzip.compress(raw))
        events_path.write_text(''.join(json.dumps(e) + '\n' for e in events))
    else:
        result = qa.read_json(direct)
        if change == 'annotation':
            result['details'][0]['ground_truth'] = 'Changed'
        else:
            result['details'].pop()
        qa.atomic_json(direct, result)
    qa.initialize_fixed_start(*fixed_start_sources)
    with pytest.raises(ValueError, match='differs|mismatch'):
        qa.prepare_fixed_start(qa.read_json(output / 'protocol.json'), output)
