"""Index-only orchestration must not silently admit or drop targets."""
import json

from scripts import index_matrix


def test_primary_index_matrix_is_complete_and_unique():
    rows = index_matrix.targets('campaign')
    assert len(rows) == 14
    assert len({row['run_id'] for row in rows}) == 14
    assert {row['dataset'] for row in rows} == {'multihoprag', 'musique'}


def test_source_digest_detects_execution_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(index_matrix, 'ROOT', tmp_path)
    main = tmp_path / 'main.py'
    main.write_text('old')
    before = index_matrix.source_digest()
    main.write_text('new')
    assert index_matrix.source_digest() != before


def test_smoke_failure_blocks_only_its_target_and_never_claims_admission(tmp_path, monkeypatch):
    from core import inference_queue, paper_compatibility
    from scripts import paper_campaign, paper_detached_runtime
    root = tmp_path / 'index-supervisor'
    root.mkdir()
    plan_path = root / 'plan.json'
    rows = index_matrix.targets('campaign')[:2]
    plan_path.write_text(json.dumps({'source_sha256': 'source', 'context': {}, 'campaign': 'campaign', 'targets': rows}))
    monkeypatch.setattr(index_matrix, 'source_digest', lambda: 'source')
    monkeypatch.setattr(paper_compatibility, 'context_configuration', dict)
    monkeypatch.setattr(index_matrix.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(paper_detached_runtime, 'register_owner', lambda _: {})
    monkeypatch.setattr(paper_detached_runtime, 'session_processes', lambda _: [])
    monkeypatch.setattr(paper_detached_runtime, 'terminate_owned', lambda _: [])
    monkeypatch.setattr(paper_campaign, 'safe_environment', dict)
    monkeypatch.setattr(paper_campaign, 'resource_lock_path', lambda: tmp_path / 'lock')
    calls = []
    def run(command, *args):
        calls.append(command)
        return int(command[-1] == 'smoke' and command[command.index('--dataset')+1] == 'musique')
    monkeypatch.setattr(paper_campaign, 'run_child', run)
    class Queue:
        def __init__(self, _): pass
        def start(self): pass
        def drain(self): pass
        def persist(self): pass
        def close(self): pass
    monkeypatch.setattr(inference_queue, 'OwnedQueue', Queue)
    assert index_matrix.supervise(plan_path) == 1
    status = json.loads((root / 'status.json').read_text())
    assert status['complete_indexes'] == 1
    assert status['benchmark_admitted'] is False
    assert status['targets']['multihoprag/prehop']['state'] == 'index_complete'
    assert status['targets']['musique/prehop']['state'] == 'blocked_by_smoke_failure'
    assert len(calls) == 3
    assert status['owned_cleanup_complete'] is True
