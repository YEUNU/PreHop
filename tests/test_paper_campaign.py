"""Harmless subprocess proof of durable status, locking, failure and redaction."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import paper_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]


def test_registered_plan_indexes_and_benchmarks_all_targets_in_order():
    steps = campaign.build_steps('fresh', 'a1', sys.executable)
    assert len([row for row in steps if row['id'].startswith('cold/')]) == 16
    assert len([row for row in steps if row['id'].startswith('one-query/')]) == 16
    names = [row['id'] for row in steps]
    assert names.index('resume_stale_rejection') < names.index('one-query/multihoprag/prehop')
    assert names[-2:] == ['full_target_admitted', 'full_matrix']
    assert steps[-1]['argv'] == ['bash', 'scripts/run_paper_matrix.sh', 'fresh']


def test_stale_pid_start_does_not_identify_a_live_owner():
    current = campaign.identity(os.getpid())
    assert campaign.alive(current)
    assert not campaign.alive({**current, 'start': 'stale'})
    assert not campaign.alive({**current, 'boot_id': 'prior-boot'})


def test_logout_launch_requires_actual_linger(monkeypatch):
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a[0], 0, 'no\n', ''))
    with pytest.raises(RuntimeError, match='linger'):
        campaign.require_logout_persistence()


def test_detached_owned_supervisor_survives_launcher_and_persists_failure(tmp_path):
    root = tmp_path/'execution'
    base = root/'data/results/smoke/supervisor'
    base.mkdir(parents=True)
    secret = 'synthetic-secret-value'
    marker = tmp_path/'forbidden-next-step'
    plan = {'campaign': 'smoke', 'commit': 'fixture', 'steps': [
        {'id': 'slow', 'argv': [sys.executable, '-c', f'import time; print({secret!r},flush=True);time.sleep(1.2)']},
        {'id': 'fails', 'argv': [sys.executable, '-c', 'raise SystemExit(7)']},
        {'id': 'never', 'argv': [sys.executable, '-c', f'from pathlib import Path;Path({str(marker)!r}).touch()']}]}
    plan_path = base/'plan.json'
    plan_path.write_text(json.dumps(plan))
    child_code = f'''
import sys,os
from pathlib import Path
sys.path.insert(0,{str(ROOT)!r})
from scripts import paper_campaign as c
c.ROOT=Path({str(root)!r})
c.resource_lock_path=lambda:Path({str(root/'data/results/.paper_resource.lock')!r})
c.ensure_no_other_campaigns=lambda unit:None
c.unit_processes=lambda unit:[c.identity(os.getpid())]
c.check_plan=lambda plan:None
c.validate_step=lambda plan,step:[]
c.safe_environment=lambda:{{**os.environ,'RAG_INFERENCE_API_KEY':{secret!r}}}
raise SystemExit(c.supervise(Path({str(plan_path)!r}),unit='fixture'))
'''
    launcher = f'''
import subprocess,sys
subprocess.Popen([sys.executable,'-c',{child_code!r}],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
'''
    parent = subprocess.run([sys.executable, '-c', launcher], check=False, timeout=5)
    assert parent.returncode == 0
    status_path = base/'status.json'
    deadline = time.monotonic() + 10
    observed_running = False
    while time.monotonic() < deadline:
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status['state'] == 'running' and status.get('child'):
                observed_running = True
                with pytest.raises(RuntimeError, match='resource lock'):
                    campaign.lock(root/'data/results/.paper_resource.lock')
            if status['state'] == 'failed':
                break
        time.sleep(.02)
    else:
        pytest.fail('Owned harmless supervisor did not finish within its fixture deadline')
    assert observed_running and status['exit_code'] == 7
    assert status['stage'] == 'fails' and status['completed_steps'] == ['slow']
    assert not marker.exists()
    logs = ''.join(path.read_text() for path in base.glob('*.log'))
    assert secret not in logs and '[REDACTED]' in logs
    assert (base/'events.jsonl').is_file()
    handle = campaign.lock(root/'data/results/.paper_resource.lock')
    handle.close()


def test_duplicate_supervisor_preserves_live_status(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'resource_lock_path', lambda: tmp_path/'resource.lock')
    monkeypatch.setattr(campaign, 'ensure_no_other_campaigns', lambda unit: None)
    plan_path = tmp_path/'plan.json'
    plan_path.write_text(json.dumps({'campaign': 'test', 'commit': 'fixture'}))
    status = {'state': 'running', 'supervisor': campaign.identity(os.getpid())}
    path = tmp_path/'status.json'
    path.write_text(json.dumps(status))
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match='owner or child'):
        campaign.supervise(plan_path, resume=True, unit='fixture')
    assert path.read_bytes() == before


def test_lock_loser_does_not_overwrite_running_owner_status(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'resource_lock_path', lambda: tmp_path/'resource.lock')
    monkeypatch.setattr(campaign, 'ensure_no_other_campaigns', lambda unit: None)
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps({'campaign': 'test', 'commit': 'fixture'}))
    status = tmp_path/'status.json'
    status.write_text('{"state":"running","owner":"winner"}')
    before = status.read_bytes()
    handle = campaign.lock(tmp_path/'resource.lock')
    try:
        with pytest.raises(RuntimeError, match='resource lock'):
            campaign.supervise(plan, unit='fixture')
        assert status.read_bytes() == before and not (tmp_path/'events.jsonl').exists()
    finally:
        handle.close()


def test_other_failed_unit_native_descendant_blocks_new_campaign(monkeypatch):
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a[0], 0,
        'prehop-paper-old.service loaded failed failed preserved old unit\n', ''))
    monkeypatch.setattr(campaign, 'unit_processes', lambda unit: [{'pid': 123, 'start': 'preserved', 'boot_id': 'same'}])
    with pytest.raises(RuntimeError, match='native descendant'):
        campaign.ensure_no_other_campaigns()


def test_direct_supervision_cannot_skip_systemd_ownership(tmp_path):
    with pytest.raises(RuntimeError, match='actual systemd unit'):
        campaign.supervise(tmp_path/'plan.json')
    assert list(tmp_path.iterdir()) == []


def test_logs_redact_url_components_and_credentials():
    env = {'RAG_INFERENCE_API_KEY': 'synthetic-key', 'RAG_INFERENCE_BASE_URL': 'http://gateway.test:5000/v1'}
    line = 'synthetic-key http://gateway.test:5000/v1 http://gateway.test:5000 gateway.test:5000 gateway.test'
    assert 'synthetic-key' not in campaign.redact(line, env)
    assert 'gateway.test' not in campaign.redact(line, env)
