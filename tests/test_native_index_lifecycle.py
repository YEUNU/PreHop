"""Native indexing retains its inputs and records success, failure, or cancellation."""
import asyncio
import time

import pytest

from cli import index


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [None, RuntimeError('native failure'), asyncio.CancelledError()])
@pytest.mark.parametrize('stats_fail', [False, True])
async def test_native_index_preserves_failure_and_records_timing(monkeypatch, failure, stats_fail):
    manifest = {'document_count': 2}
    capacity = {'bytes': 17}
    records = []

    async def native(**kwargs):
        assert kwargs == {'dataset_path': 'corpus', 'corpus_tag': 'tag', 'corpus_manifest': manifest}
        if failure is not None:
            raise failure
        return {'native_seconds': .2}

    async def collect(strategy, tag):
        assert (strategy, tag) == ('lightrag', 'tag')
        return capacity

    def write(*args):
        records.append(args)
        if stats_fail:
            raise OSError('stats disk failure')

    monkeypatch.setattr(index, '_collect_index_capacity', collect)
    monkeypatch.setattr(index, '_write_runtime_stage_stats', write)
    call = index._run_native_index(native, 'lightrag', 'corpus', 'tag', manifest, 'default', time.perf_counter())
    if failure is not None:
        with pytest.raises(type(failure)) as caught:
            await call
        assert caught.value is failure
    elif stats_fail:
        with pytest.raises(OSError, match='stats disk failure'):
            await call
    else:
        await call
    assert len(records) == 1
    strategy, tag, path, timing, status, recorded_manifest, model, recorded_capacity = records[0]
    assert (strategy, tag, path, model) == ('lightrag', 'tag', 'corpus', 'default')
    assert recorded_manifest is manifest
    assert status == ('failed' if failure is not None else 'complete')
    assert timing['total_elapsed_seconds'] >= timing['official_pipeline_seconds'] >= 0
    assert recorded_capacity == (None if failure is not None else capacity)
    if failure is None:
        assert timing['native_seconds'] == .2
