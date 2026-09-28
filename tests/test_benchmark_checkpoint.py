"""A failed checkpoint never commits results without their resume traces."""
import json

import pytest

from core import benchmark_checkpoint as checkpoint
from utils.io import _write_json


def row(query_id, idx):
    return {'query_id': query_id, 'idx': idx, 'query': query_id,
            'interaction_trace': [{'step': query_id}]}


def summary(*rows):
    return {'status': 'in_progress', 'details': list(rows)}


@pytest.fixture(autouse=True)
def no_derived_reports(monkeypatch):
    monkeypatch.setattr(checkpoint, '_write_model_report_artifacts', lambda *a, **k: None)


def test_trace_write_failure_preserves_previous_commit(tmp_path, monkeypatch):
    path = tmp_path / 'result.json'
    checkpoint.write_checkpoint(summary(row('two', 2)), path)
    before = path.read_bytes()
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(checkpoint, '_write_jsonl', fail)
    with pytest.raises(OSError, match='disk full'):
        checkpoint.write_checkpoint(summary(row('one', 1), row('two', 2)), path)
    assert path.read_bytes() == before
    retained, _ = checkpoint._resume_benchmark_rows(path, [{'_id': 'one'}, {'_id': 'two'}])
    assert retained[0]['interaction_trace'] == [{'step': 'two'}]


def test_resume_uses_query_ids_after_trace_publish_but_main_write_fails(tmp_path, monkeypatch):
    path = tmp_path / 'result.json'
    checkpoint.write_checkpoint(summary(row('two', 2)), path)
    before = path.read_bytes()
    def fail(*args):
        raise OSError('main write failed')
    monkeypatch.setattr(checkpoint, '_write_slim_main', fail)
    with pytest.raises(OSError, match='main write failed'):
        checkpoint.write_checkpoint(summary(row('one', 1), row('two', 2)), path)
    assert path.read_bytes() == before
    retained, _ = checkpoint._resume_benchmark_rows(path, [{'_id': 'one'}, {'_id': 'two'}])
    assert [r['query_id'] for r in retained] == ['two']
    assert retained[0]['interaction_trace'] == [{'step': 'two'}]


def test_missing_committed_trace_is_an_error(tmp_path):
    path = tmp_path / 'result.json'
    checkpoint.write_checkpoint(summary(row('one', 1)), path)
    path.with_suffix('.traces.jsonl').write_text('')
    with pytest.raises(ValueError, match='Missing trace.*one'):
        checkpoint._resume_benchmark_rows(path, [{'_id': 'one'}])


def test_derived_report_failure_does_not_destroy_resume_evidence(tmp_path, monkeypatch, caplog):
    path = tmp_path / 'result.json'
    def fail(*args, **kwargs):
        raise OSError('report failed')
    monkeypatch.setattr(checkpoint, '_write_model_report_artifacts', fail)
    checkpoint.write_checkpoint(summary(row('one', 1)), path)
    retained, _ = checkpoint._resume_benchmark_rows(path, [{'_id': 'one'}])
    assert retained[0]['interaction_trace'] == [{'step': 'one'}]
    assert 'Checkpoint saved; failed to write derived reports' in caplog.text


def test_json_serialization_failure_keeps_old_file_and_cleans_temp(tmp_path):
    path = tmp_path / 'result.json'
    _write_json(path, {'committed': True})
    with pytest.raises(TypeError):
        _write_json(path, {'bad': object()})
    assert json.loads(path.read_text()) == {'committed': True}
    assert list(tmp_path.iterdir()) == [path]
