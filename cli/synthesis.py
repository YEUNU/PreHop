"""Sequential dataset/system replay with concurrent final-answer requests only."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from core.synthesis_replay import TraceContextProvider
from utils.io import atomic_text_writer
from utils.prompts.prehop_answer import SYNTHESIS_PROMPT_VERSION


def atomic_json(path, value):
    with atomic_text_writer(path) as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def timestamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def reader_settings():
    from core.generation_profiles import request_settings
    settings = request_settings('prehop_answer')
    # Preserve the saved request identity for numerically identical defaults.
    if float(settings['temperature']).is_integer():
        settings['temperature'] = int(settings['temperature'])
    return {**settings, 'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}


def request_identity(row):
    settings = reader_settings()
    payload = {'messages': row.messages(), 'prompt_version': SYNTHESIS_PROMPT_VERSION,
               'temperature': settings['temperature'], 'max_tokens': settings['max_tokens'],
               'enable_thinking': settings['extra_body']['chat_template_kwargs']['enable_thinking']}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


async def run_synthesis(inputs: Path, output: Path, *, concurrency=24, client=None,
                        model=None, interval=1.0):
    """Consume the trace hook; no retrieval, ranking, embedding, or truncation occurs here."""
    from core.inference_transport import InferenceTransport
    transport = InferenceTransport.resolve('prehop')
    model = model or transport.generation_model
    owned_client = client is None
    if owned_client:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=transport.generation_base_url, api_key=transport.api_key,
                             timeout=transport.timeout_seconds, max_retries=0)
    failed = False
    try:
        return await _run_synthesis(inputs, output, concurrency=concurrency, client=client,
                                    model=model, interval=interval, transport=transport)
    except BaseException:
        failed = True
        raise
    finally:
        if owned_client:
            try:
                await client.close()
            except Exception:
                if not failed:
                    raise
                logging.getLogger('Prehop').warning('Reader cleanup failed after replay failure', exc_info=True)


async def _run_synthesis(inputs, output, *, concurrency, client, model, interval, transport):
    from core.vllm_client import VLLMClient
    from utils.parsers import clean_and_unwrap_json

    output.mkdir(parents=True, exist_ok=True)
    config = output / 'execution_config.json'
    if not config.exists():
        atomic_json(config, {'concurrency': concurrency, 'minimum_request_interval_seconds': interval,
                             'retry_429_initial_seconds': 30, 'retry_429_max_seconds': 300})
    groups = list(TraceContextProvider(inputs).groups())
    rows_by_key = {r.key: r for _, rows in groups for r in rows}
    responses = output / 'responses.jsonl'
    done = {}
    if responses.exists():
        for line in responses.read_text().splitlines():
            r = json.loads(line)
            key = (r['dataset'], r['method'], r['query_id'])
            source = rows_by_key.get(key)
            if (source is not None and r.get('status') == 'completed'
                    and r.get('request_sha256') == request_identity(source) and r.get('model') == model):
                done[key] = r
    # Superseded inputs/answers never remain alongside current responses.
    with atomic_text_writer(responses) as stream:
        for r in done.values():
            stream.write(json.dumps(r, ensure_ascii=False) + '\n')
    started = time.monotonic()
    status = {'state': 'running', 'pid': os.getpid(), 'started_utc': timestamp(),
              'total': len(rows_by_key), 'resumed': len(done), 'completed': len(done),
              'model': model, 'execution': 'sequential dataset/system; trace context; final synthesis only; direct gateway',
              'concurrency': concurrency, 'rate_limit_events': 0, 'transport': transport.policy_dict()}
    status_path = output / 'status.json'
    atomic_json(status_path, status)
    cooldown_until = last_start = 0.0
    lock = asyncio.Lock()
    from contextlib import ExitStack
    resources = ExitStack()
    try:
        log = resources.enter_context(responses.open('a', buffering=1))
        events = resources.enter_context((output / 'events.jsonl').open('a', buffering=1))
    except BaseException:
        resources.close()
        raise

    def event(**kwargs):
        events.write(json.dumps({'utc': timestamp(), **kwargs}) + '\n')

    async def one(row):
        nonlocal cooldown_until, last_start
        record = {k: v for k, v in row.as_record().items() if k != 'context'}
        record.update(model=model, request_sha256=request_identity(row), attempts=[],
                      reader_transport=transport.policy_dict())
        began = time.monotonic()
        rates = failures = 0
        while True:
            cfg = json.loads(config.read_text())
            async with lock:
                while (delay := max(cooldown_until, last_start + cfg['minimum_request_interval_seconds']) - time.monotonic()) > 0:
                    await asyncio.sleep(min(delay, 30))
                last_start = time.monotonic()
            try:
                response = await client.chat.completions.create(
                    model=model, messages=row.messages(), **reader_settings())
                record['attempts'].append({'response': response.model_dump(mode='json')})
                answer = VLLMClient.think_strip(None, clean_and_unwrap_json(response.choices[0].message.content or ''))
                record.update(status='completed', answer=str(answer or ''), empty_output=not bool(str(answer or '').strip()),
                              reader_seconds=time.monotonic()-began, completed_utc=timestamp())
                log.write(json.dumps(record, ensure_ascii=False)+'\n')
                log.flush()
                os.fsync(log.fileno())
                done[row.key] = record
                status.update(completed=len(done), updated_utc=timestamp(), elapsed_seconds=round(time.monotonic()-started, 1))
                atomic_json(status_path, status)
                return
            except Exception as exc:  # noqa: BLE001 - preserve typed transport errors; never rewrite model outputs
                code = getattr(exc, 'status_code', None)
                record['attempts'].append({'error_type': type(exc).__name__, 'status_code': code, 'utc': timestamp()})
                can_retry = len(record['attempts']) < transport.retry_attempts
                if (code == 429 or type(exc).__name__ == 'RateLimitError') and can_retry:
                    rates += 1
                    wait = min(cfg['retry_429_initial_seconds'] * 2**min(rates-1, 5), cfg['retry_429_max_seconds'])
                    try:
                        wait = max(wait, float(exc.response.headers.get('retry-after', 0)))
                    except (AttributeError, TypeError, ValueError):
                        pass
                    cooldown_until = max(cooldown_until, time.monotonic()+wait)
                    status['rate_limit_events'] += 1
                    event(type='rate_limit', dataset=row.dataset, method=row.method, query_id=row.query_id, wait_seconds=wait)
                elif VLLMClient._is_retryable_inference_error(exc) and can_retry:
                    failures += 1
                    cooldown_until = max(cooldown_until, time.monotonic()+min(10*2**(failures-1), 300))
                    event(type='transport_retry', error_type=type(exc).__name__, attempt=failures)
                else:
                    event(type='request_failed', dataset=row.dataset, method=row.method,
                          query_id=row.query_id, attempts=record['attempts'])
                    raise RuntimeError(f'{row.dataset}/{row.method}/{row.query_id}: {type(exc).__name__}, HTTP {code}') from None

    try:
        for (dataset, method), rows in groups:
            status.update(current_group=dataset+'/'+method, group_total=len(rows),
                          group_completed=sum(r.key in done for r in rows))
            atomic_json(status_path, status)
            event(type='group_start', group=status['current_group'])
            pending = [r for r in rows if r.key not in done]
            while pending:
                cfg = json.loads(config.read_text())
                if cfg.get('pause'):
                    status.update(state='paused', updated_utc=timestamp())
                    atomic_json(status_path, status)
                    return
                n = int(cfg['concurrency'])
                if n < 1:
                    raise ValueError('Concurrency must be positive')
                status['concurrency'] = n
                batch, pending = pending[:n], pending[n:]
                tasks = [asyncio.create_task(one(r)) for r in batch]
                try:
                    await asyncio.gather(*tasks)
                except BaseException:
                    # Finish cancellation before closing the response log or client.
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
                status['group_completed'] = sum(r.key in done for r in rows)
                atomic_json(status_path, status)
            event(type='group_complete', group=status['current_group'], rows=len(rows))
        status.update(state='generation_complete', completed=len(done), finished_utc=timestamp())
        atomic_json(status_path, status)
    except BaseException as exc:
        status.update(state='stopped', error_type=type(exc).__name__, error=str(exc)[:300], updated_utc=timestamp())
        atomic_json(status_path, status)
        raise
    finally:
        resources.close()


def main():
    from scripts.runner_environment import _load_runner_environment
    _load_runner_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True, help='Recorded synthesis-context JSONL; no live retrieval fallback')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=24)
    args = parser.parse_args()
    asyncio.run(run_synthesis(args.inputs, args.output_dir, concurrency=args.concurrency))


if __name__ == '__main__':
    main()
