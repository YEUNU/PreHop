"""Immutable retry plans select one fresh failed target and retain validated successes."""
import json
import sys

import pytest

from scripts import paper_campaign as c
from scripts import paper_cold_canary as cold
from scripts import paper_gate_ledger as gate
from scripts import paper_stage_runner as stage


@pytest.fixture
def previous(tmp_path, monkeypatch):
    for module in (c, gate, stage, cold):
        monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(gate, '_context', lambda: {'configuration': 'unchanged'})
    monkeypatch.setattr(stage, 'selected_python_environment', lambda: {'PYTHON_BIN': sys.executable, 'UV_PROJECT_ENVIRONMENT': sys.prefix})
    monkeypatch.setattr(c, 'resource_lock_path', lambda: tmp_path / 'resource.lock')
    monkeypatch.setattr(c, 'ensure_no_other_campaigns', lambda *a: None)
    plan = {'schema_version': 1, 'campaign': 'fixture', 'attempt': 'a1', 'commit': 'historical',
            'python': sys.executable, 'python_prefix': sys.prefix, 'context': gate._context(),
            'steps': c.build_steps('fixture', 'a1', sys.executable)}
    failed = 'cold/multihoprag/ms_graphrag'
    ids = [row['id'] for row in plan['steps']]
    status = {'state': 'failed', 'stage': failed, 'completed_steps': ids[:ids.index(failed)], 'child': None}
    path = tmp_path / 'data/results/fixture/supervisor/plan.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan))
    path.with_name('status.json').write_text(json.dumps(status))
    return path, plan, status, failed


def test_successor_preserves_original_and_revalidates_completed(previous, monkeypatch):
    path, _plan, status, failed = previous
    original = {p: p.read_bytes() for p in (path, path.with_name('status.json'))}
    observed = []
    monkeypatch.setattr(c, 'step_evidence', lambda plan, step: observed.append(step['id']))
    successor_path = c.create_successor_plan(path, failed, 'a2', 'ms-retry')
    successor = json.loads(successor_path.read_text())
    assert observed == status['completed_steps']
    assert all(p.read_bytes() == raw for p, raw in original.items())
    assert successor_path.parent == path.parent / 'segments/ms-retry'
    assert successor['target_attempts'] == {failed: 'a2'}
    for step in successor['steps']:
        if step['id'].startswith(('cold/', 'one-query/')):
            assert step['argv'][-1] == ('a2' if step['id'] == failed else 'a1')


def test_aggregate_uses_same_attempt_map_for_actual_file_references(previous, monkeypatch):
    from core.strategy_registry import PRIMARY_STRATEGIES
    _, _, _, failed = previous
    recorded = []
    monkeypatch.setattr(gate, 'record', lambda ledger, name, path: recorded.append(json.loads(path.read_text())))
    for dataset in ('multihoprag', 'hotpotqa'):
        for method in PRIMARY_STRATEGIES:
            attempt = 'a2' if f'cold/{dataset}/{method}' == failed else 'a1'
            path = c.ROOT / 'data/results/fixture/cold_v2' / attempt / dataset / method / 'evidence.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'target': f'{dataset}/{method}', 'attempt': attempt}))
    stage.aggregate('fixture', 'cold_canary_16', 'a1', {failed: 'a2'})
    assert len(recorded[0]['targets']) == 14
    for target, ref in recorded[0]['targets'].items():
        _, actual = gate._bound_json(ref)
        assert actual['attempt'] == ('a2' if target == 'multihoprag/ms_graphrag' else 'a1')


def test_supervisor_inheritance_is_not_mutated_as_new_steps_finish(previous, monkeypatch):
    import os
    path, _, status, failed = previous
    monkeypatch.setattr(c, 'step_evidence', lambda *a: [])
    successor_path = c.create_successor_plan(path, failed, 'a2', 'owned-successor')
    initial = successor_path.read_bytes()
    inherited = list(status['completed_steps'])
    executed = []
    monkeypatch.setattr(c, 'safe_environment', dict)
    monkeypatch.setattr(c, 'unit_processes', lambda unit: [c.identity(os.getpid())])
    monkeypatch.setattr(c, 'final_admissions', lambda *a: [{'status': 'fixture-only'}])
    def child(argv, *args):
        executed.append(argv)
        args[-1]({'log_drain_complete': True})
        return 0
    monkeypatch.setattr(c, 'run_child', child)
    assert c.supervise(successor_path, unit='owned-fixture') == 0
    result = json.loads(successor_path.with_name('status.json').read_text())
    assert result['state'] == 'completed'
    assert len(executed) == len(result['completed_steps']) - len(inherited)
    assert successor_path.read_bytes() == initial
    assert json.loads(path.with_name('status.json').read_text()) == status
