import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cli.synthesis import run_synthesis
from core.synthesis_replay import SynthesisInput, TraceContextProvider
from scripts.prepare_synthesis_inputs import prepare_inputs


def record(method, qid, context):
    return SynthesisInput('multihoprag', method, qid, 'Question ' + qid, context,
                          {'kind': 'saved_returned_passages'}, {'included_passages': 20})


def write_inputs(path, rows):
    path.write_text(''.join(json.dumps(r.as_record()) + '\n' for r in rows))


class FakeClient:
    def __init__(self, fail_once=False):
        self.chat = SimpleNamespace(completions=self)
        self.active = self.peak = 0
        self.requests = []
        self.fail_once = fail_once

    async def create(self, **request):
        self.requests.append(request)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(.005)
            if self.fail_once:
                self.fail_once = False
                error = RuntimeError('rate limited')
                error.status_code = 429
                error.response = SimpleNamespace(headers={'retry-after': '0'})
                raise error
            text = '' if 'Question empty' in request['messages'][-1]['content'] else 'Final Answer: Yes'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                                   model_dump=lambda **_: {'model': request['model'], 'choices': [{'message': {'content': text}}]})
        finally:
            self.active -= 1


def test_prepare_preserves_all_returned_text_and_order_without_answers(tmp_path):
    sources = [{'doc': f'D{i}', 'page': 0, 'sent_id': i, 'text': ('word ' * 1000) + f'FINAL FACT {i}'} for i in range(20)]
    source = tmp_path/'source.json'
    source.write_text(json.dumps({'dataset': 'MultiHop-RAG', 'strategy': 'gfm_rag', 'details': [
        {'query_id': 'q', 'query': 'target', 'question_type': 'comparison_query',
         'retrieved_sources': sources, 'answer': 'old answer', 'ground_truth': 'gold label'}]}))
    output = tmp_path/'inputs.jsonl'
    assert prepare_inputs([source], output) == 1
    row = next(iter(TraceContextProvider(output)))
    assert row.dataset == 'multihoprag'
    assert row.metadata['included_passages'] == 20
    assert all(s['text'] in row.context for s in sources)
    assert [row.context.index(s['text']) for s in sources] == sorted(row.context.index(s['text']) for s in sources)
    assert 'gold label' not in row.context and 'old answer' not in row.context
    assert 'FINAL FACT 19' in row.messages()[1]['content']


@pytest.mark.asyncio
async def test_synthesis_only_keeps_full_context_and_runs_groups_sequentially(tmp_path):
    huge = 'preserved evidence ' * 20000
    rows = [record('prehop', '1', huge), record('prehop', '2', 'second'),
            record('hoprag', '3', 'third'), record('hoprag', 'empty', 'fourth')]
    inputs = tmp_path/'inputs.jsonl'
    write_inputs(inputs, rows)
    client = FakeClient()
    output = tmp_path/'out'
    await run_synthesis(inputs, output, concurrency=2, client=client, model='fixture', interval=0)
    assert client.peak == 2
    assert [r['messages'] for r in client.requests] == [r.messages() for r in rows]
    assert all(r['max_tokens'] == 256 and r['temperature'] == 0 for r in client.requests)
    assert json.loads((output/'status.json').read_text())['state'] == 'generation_complete'
    responses = [json.loads(s) for s in (output/'responses.jsonl').read_text().splitlines()]
    assert len(responses) == 4 and responses[-1]['empty_output']
    assert len(client.requests) == 4  # Empty success is not regenerated.


@pytest.mark.asyncio
async def test_resume_reuses_identical_requests_but_drops_changed_context(tmp_path):
    rows = [record('prehop', '1', 'first'), record('prehop', '2', 'second')]
    inputs = tmp_path/'inputs.jsonl'
    write_inputs(inputs, rows)
    client = FakeClient()
    output = tmp_path/'out'
    await run_synthesis(inputs, output, concurrency=2, client=client, model='fixture', interval=0)
    await run_synthesis(inputs, output, concurrency=2, client=client, model='fixture', interval=0)
    assert len(client.requests) == 2
    rows[1] = record('prehop', '2', 'new evidence')
    write_inputs(inputs, rows)
    await run_synthesis(inputs, output, concurrency=2, client=client, model='fixture', interval=0)
    assert len(client.requests) == 3
    results = [json.loads(s) for s in (output/'responses.jsonl').read_text().splitlines()]
    assert len(results) == 2
    assert results[1]['context_sha256'] == rows[1].context_sha256


@pytest.mark.asyncio
async def test_reader_wire_settings_and_resume_identity_share_generation_profile(tmp_path, monkeypatch):
    import hashlib

    from cli.synthesis import request_identity
    from core.config import RAGConfig
    from utils.prompts.prehop_answer import SYNTHESIS_PROMPT_VERSION

    row = record('prehop', '1', 'same evidence')
    monkeypatch.setattr(RAGConfig, 'PREHOP_SYNTHESIS_MAX_OUTPUT_TOKENS', 256)
    previous = hashlib.sha256(json.dumps({'messages': row.messages(), 'prompt_version': SYNTHESIS_PROMPT_VERSION,
        'temperature': 0, 'max_tokens': 256, 'enable_thinking': False}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert request_identity(row) == previous
    monkeypatch.setattr(RAGConfig, 'PREHOP_SYNTHESIS_MAX_OUTPUT_TOKENS', 37)
    assert request_identity(row) != previous
    inputs = tmp_path / 'inputs.jsonl'
    write_inputs(inputs, [row])
    client = FakeClient()
    await run_synthesis(inputs, tmp_path / 'out', client=client, interval=0)
    assert client.requests[0]['max_tokens'] == 37


@pytest.mark.asyncio
async def test_429_retries_same_request_without_changing_evidence(tmp_path):
    inputs = tmp_path/'inputs.jsonl'
    row = record('prehop', '1', 'immutable context')
    write_inputs(inputs, [row])
    output = tmp_path/'out'
    output.mkdir()
    (output/'execution_config.json').write_text(json.dumps({'concurrency': 1,
        'minimum_request_interval_seconds': 0, 'retry_429_initial_seconds': 0, 'retry_429_max_seconds': 0}))
    client = FakeClient(fail_once=True)
    await run_synthesis(inputs, output, client=client, model='fixture')
    assert len(client.requests) == 2 and client.requests[0] == client.requests[1]
    assert json.loads((output/'status.json').read_text())['rate_limit_events'] == 1


@pytest.mark.asyncio
async def test_failed_request_cancels_pending_peers_before_returning(tmp_path):
    inputs = tmp_path/'inputs.jsonl'
    write_inputs(inputs, [record('prehop', 'bad', 'first'), record('prehop', 'pending', 'second')])
    peer_started = asyncio.Event()
    peer_cancelled = asyncio.Event()

    class FailingClient:
        async def create(self, **request):
            if 'Question bad' in request['messages'][1]['content']:
                await peer_started.wait()
                error = RuntimeError('nonrecoverable request')
                error.status_code = 400
                raise error
            peer_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                peer_cancelled.set()
                raise

    client = SimpleNamespace(chat=SimpleNamespace(completions=FailingClient()))
    output = tmp_path/'out'
    with pytest.raises(RuntimeError, match='HTTP 400'):
        await run_synthesis(inputs, output, concurrency=2, client=client, model='fixture', interval=0)
    assert peer_cancelled.is_set()
    assert json.loads((output/'status.json').read_text())['state'] == 'stopped'
    assert not (output/'responses.jsonl').read_text()
    events = [json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
    assert events[-1]['type'] == 'request_failed'
    assert events[-1]['attempts'][0]['status_code'] == 400


@pytest.mark.asyncio
async def test_main_synthesis_path_does_not_dispatch_benchmark(tmp_path, monkeypatch):
    import main
    called = []
    async def replay(inputs, output, **kwargs):
        called.append((inputs, output, kwargs))
    monkeypatch.setattr('cli.synthesis.run_synthesis', replay)
    monkeypatch.setattr('sys.argv', ['main.py', '--mode', 'synthesize', '--trace-inputs', 'saved.jsonl', '--output-dir', str(tmp_path)])
    monkeypatch.setattr(main, 'run_benchmark_multi_seed', lambda *a, **kw: pytest.fail('Live benchmark dispatched'))
    monkeypatch.setattr(main, 'run_indexing', lambda *a, **kw: pytest.fail('Indexing dispatched'))
    await main.main()
    assert called == [(Path('saved.jsonl'), tmp_path, {'concurrency': 24})]


@pytest.mark.asyncio
@pytest.mark.parametrize('close_fails', [False, True])
async def test_owned_reader_client_closes_on_input_error(tmp_path, monkeypatch, close_fails):
    client = SimpleNamespace(close=AsyncMock(side_effect=RuntimeError('close failed') if close_fails else None))
    monkeypatch.setattr('openai.AsyncOpenAI', lambda **kwargs: client)
    with pytest.raises(FileNotFoundError):
        await run_synthesis(tmp_path / 'missing.jsonl', tmp_path / 'out')
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_reader_rate_limits_use_shared_attempt_budget(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_INFERENCE_RETRY_ATTEMPTS', '2')
    inputs = tmp_path / 'inputs.jsonl'
    write_inputs(inputs, [record('prehop', '1', 'evidence')])
    output = tmp_path / 'out'
    output.mkdir()
    (output / 'execution_config.json').write_text(json.dumps({'concurrency': 1,
        'minimum_request_interval_seconds': 0, 'retry_429_initial_seconds': 0, 'retry_429_max_seconds': 0}))
    error = RuntimeError('rate limited')
    error.status_code = 429
    calls = AsyncMock(side_effect=error)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=calls)))
    with pytest.raises(RuntimeError, match='HTTP 429'):
        await run_synthesis(inputs, output, client=client)
    assert calls.await_count == 2
    assert calls.await_args_list[0] == calls.await_args_list[1]
    status = json.loads((output / 'status.json').read_text())
    assert status['transport']['inference_retry_attempts'] == 2
    assert status['state'] == 'stopped'


async def replay_fixture(tmp_path, dataset='multihoprag', *, original_failure=False):
    from scripts.datasets.prepare_hotpotqa_hipporag import prepare
    from utils.metrics import evaluate_multihoprag_response

    store = tmp_path / 'corpus/sentences.sqlite3'
    if dataset == 'hotpotqa':
        raw = tmp_path / 'raw'
        raw.mkdir()
        (raw / 'hotpotqa_corpus.json').write_text(json.dumps({'Article': ['Yes is the answer.']}))
        (raw / 'hotpotqa.json').write_text(json.dumps([{'_id': 'q', 'question': 'Is it?', 'answer': 'Yes',
            'type': 'bridge', 'supporting_facts': [['Article', 0]]}]))
        prepare(raw, store.parent, tmp_path / 'queries.json')
    sources = [{'doc': 'Article', 'text': 'Yes is the answer.'}]
    scores = await evaluate_multihoprag_response('Is it?', 'No', 'Yes', sources,
        evidence_facts=['Yes is the answer.'], evidence_docs=['Article'], dataset=dataset,
        supporting_facts=[['Article', 0]], hotpot_sentence_store=str(store))
    row = {'query_id': 'q', 'original_query_id': 'original-q', 'query': 'Is it?',
           'ground_truth': 'Yes', 'answer': 'No', 'question_type': 'inference_query',
           'retrieved_sources': sources, 'expected_sources': {'facts': ['Yes is the answer.'],
               'docs': ['Article'], 'supporting_facts': [['Article', 0]]}, **scores}
    if original_failure:
        row.update(error='original retrieval failure', failure_scope='query')
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({'dataset': dataset, 'strategy': 'prehop', 'status': 'completed_unadmitted',
        'evaluation_scope': 'released_benchmark' if dataset == 'hotpotqa' else 'full_benchmark',
        'total_queries': 1, 'avg_latency': 999, 'details': [row]}))
    inputs = tmp_path / 'inputs.jsonl'
    prepare_inputs([source], inputs)
    await run_synthesis(inputs, tmp_path / 'reader', client=FakeClient(), model='fixture', interval=0)
    return source, tmp_path / 'reader/responses.jsonl', store


@pytest.mark.asyncio
@pytest.mark.parametrize('dataset', ['multihoprag', 'hotpotqa'])
@pytest.mark.parametrize('original_failure', [False, True])
async def test_reader_scores_preserve_retrieval_identity_failures_and_originals(tmp_path, dataset, original_failure):
    from scripts.export_official_results import export, export_synthesis_results

    source, responses, store = await replay_fixture(tmp_path, dataset, original_failure=original_failure)
    before = source.read_bytes()
    paths = await export_synthesis_results([source], responses, tmp_path / 'scores', store)
    result = json.loads(paths[0].read_text())
    assert source.read_bytes() == before
    original_row = json.loads(before)['details'][0]
    row = result['details'][0]
    assert row['retrieved_sources'] == original_row['retrieved_sources']
    assert row['expected_sources'] == original_row['expected_sources']
    assert row['original_query_id'] == original_row['original_query_id']
    assert row['answer'] == 'Final Answer: Yes'
    assert 'avg_latency' not in result and 'latency' not in row
    official_path, _ = export(paths[0], tmp_path / 'scores')
    report = json.loads(official_path.read_text())
    assert report['failed_rows'] == int(original_failure)
    quality = report['qa']['overall']['accuracy'] if dataset == 'multihoprag' else report['metrics']['joint_em']
    assert quality == (0 if original_failure else 1)
    if not original_failure and dataset == 'hotpotqa':
        assert row['predicted_supporting_facts'] == original_row['predicted_supporting_facts']
    assert report['common_reader']['source_sha256'] == result['common_reader']['source_sha256']


@pytest.mark.asyncio
@pytest.mark.parametrize('mismatch', ['missing', 'duplicate', 'source', 'context', 'incomplete'])
async def test_reader_export_rejects_mismatched_or_incomplete_evidence(tmp_path, mismatch):
    from scripts.export_official_results import export_synthesis_results

    source, responses, store = await replay_fixture(tmp_path)
    rows = [json.loads(line) for line in responses.read_text().splitlines()]
    if mismatch == 'missing':
        rows = []
    elif mismatch == 'duplicate':
        rows += rows
    elif mismatch == 'source':
        rows[0]['trace_provenance']['source_sha256'] = 'stale'
    elif mismatch == 'context':
        rows[0]['context_sha256'] = 'truncated'
    else:
        rows[0]['status'] = 'stopped'
    responses.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        await export_synthesis_results([source], responses, tmp_path / 'scores', store)
    assert not (tmp_path / 'scores').exists()
