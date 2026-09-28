import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

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
            text = '' if 'Question empty' in request['messages'][1]['content'] else 'Final Answer: Yes'
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
