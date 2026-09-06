"""Harmless real nohup/session fixtures; no experiment processes or services."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import paper_campaign as campaign
from scripts import paper_detached_runtime as detached

ROOT = Path(__file__).resolve().parents[1]


def test_reused_leader_pid_retains_exact_previously_observed_descendant(monkeypatch, tmp_path):
    monkeypatch.setattr(detached, 'owner_directory', lambda: tmp_path)
    current = campaign.identity(os.getpid())
    previous_owner = {**current, 'start': str(int(current['start']) - 1)}
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(.5)'])
    try:
        known = campaign.identity(child.pid)
        (tmp_path / f'{current["pid"]}-{previous_owner["start"]}.json').write_text(json.dumps({
            'supervisor': previous_owner, 'observed_descendants': [known],
        }))
        assert detached.session_processes(previous_owner) == [known]
        with pytest.raises(RuntimeError, match='Another nohup campaign'):
            detached.ensure_no_detached_owners()
        with pytest.raises(RuntimeError, match='Another nohup campaign'):
            detached.ensure_no_detached_owners(os.getpid())
    finally:
        child.wait(timeout=3)


def test_real_nohup_child_survives_launcher_exit_and_hup(tmp_path):
    marker = tmp_path / 'receipt.json'
    code = f'''
import json,os,signal,time
from pathlib import Path
time.sleep(1)
os.kill(os.getpid(),signal.SIGHUP)
Path({str(marker)!r}).write_text(json.dumps({{'pid':os.getpid(),'sid':os.getsid(0),'pgid':os.getpgrp(),'hup_ignored':signal.getsignal(signal.SIGHUP)==signal.SIG_IGN}}))
'''
    launcher = f'''
import os,sys
from pathlib import Path
sys.path.insert(0,{str(ROOT)!r})
from scripts.paper_detached_runtime import spawn_nohup
spawn_nohup([sys.executable,'-c',{code!r}],env=os.environ.copy(),cwd=Path({str(tmp_path)!r}),log=Path({str(tmp_path/'nohup.log')!r}))
'''
    result = subprocess.run([sys.executable, '-c', launcher], check=False, timeout=5)
    assert result.returncode == 0
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(.05)
    row = json.loads(marker.read_text())
    assert row['pid'] == row['sid'] == row['pgid'] and row['hup_ignored']
    assert (tmp_path / 'nohup.log').stat().st_mode & 0o777 == 0o600


def test_owned_session_cleanup_terms_only_its_recorded_session(tmp_path):
    marker = tmp_path / 'child.json'
    code = f'''
import subprocess,sys,time
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'])
Path({str(marker)!r}).write_text(str(child.pid))
time.sleep(10)
'''
    child = detached.spawn_nohup([sys.executable, '-c', code], env=os.environ.copy(), cwd=tmp_path, log=tmp_path/'cleanup.log')
    owner = campaign.identity(child.pid)
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(.05)
    try:
        owned = detached.session_processes(owner)
        assert {child.pid, int(marker.read_text())}.issubset({row['pid'] for row in owned})
        assert os.getpid() not in {row['pid'] for row in owned}
        assert detached.terminate_owned(owner, timeout=3) == []
    finally:
        child.wait(timeout=5)


def test_missing_systemd_does_not_block_nohup_ownership(monkeypatch, tmp_path):
    monkeypatch.setattr(detached, 'owner_directory', lambda: tmp_path)
    def missing(*args, **kwargs):
        raise FileNotFoundError('systemctl')
    monkeypatch.setattr(campaign.subprocess, 'run', missing)
    campaign.ensure_no_other_campaigns()


def test_terminal_monitor_runs_current_admission_verifier_and_writes_receipt(monkeypatch, tmp_path):
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'campaign': 'fixture'}))
    (tmp_path / 'status.json').write_text(json.dumps({'state': 'completed', 'stage': 'full_matrix',
                                                   'owned_cleanup': 'TERM_only', 'owned_cleanup_complete': True,
                                                   'supervisor': campaign.identity(os.getpid()), 'exit_code': 0}))
    checked = []
    def verify(name):
        checked.append(name)
        return [{'target': 'fixture', 'status': 'admitted'}]
    monkeypatch.setattr(campaign, 'admission_statuses', verify)
    monkeypatch.setattr(campaign, 'checkpoint_progress', lambda name: [])
    assert detached.monitor(plan) == 0
    row = json.loads(next(tmp_path.glob('monitor-*-terminal.json')).read_text())
    assert checked == ['fixture'] and row['admissions'][0]['status'] == 'admitted'
    assert row['automatic_chat_notification'] is False and row['eta_seconds'] is None


def test_actual_supervisor_adopts_escaped_session_and_cleans_only_owned_child(tmp_path):
    base = tmp_path / 'execution'
    base.mkdir()
    owners = tmp_path / 'owners'
    owners.mkdir()
    marker = tmp_path / 'escaped.json'
    native = f'''import subprocess,sys
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'],start_new_session=True)
Path({str(marker)!r}).write_text(str(child.pid))
'''
    plan = base / 'plan.json'
    plan.write_text(json.dumps({'campaign': 'harmless-subprocess-fixture', 'commit': 'fixture',
                               'steps': [{'id': 'escaped-descendant', 'argv': [sys.executable, '-c', native]}]}))
    script = f'''
import os,sys
from pathlib import Path
sys.path.insert(0,{str(ROOT)!r})
from scripts import paper_campaign as c,paper_detached_runtime as d
c.ROOT=Path({str(base)!r})
d.owner_directory=lambda:Path({str(owners)!r})
c.resource_lock_path=lambda:Path({str(base/'resource.lock')!r})
c.ensure_no_other_campaigns=lambda unit:None
c.check_plan=lambda plan:None
c.safe_environment=lambda:os.environ.copy()
c.validate_step=lambda plan,step:[]
c.CHILD_LOG_DRAIN_SECONDS=.2
raise SystemExit(c.supervise(Path({str(plan)!r}),detached=True))
'''
    process = detached.spawn_nohup([sys.executable, '-c', script], env=os.environ.copy(), cwd=base, log=base/'nohup.log')
    assert process.wait(timeout=10) == 1
    status = json.loads((base/'status.json').read_text())
    assert status['state'] == 'failed' and status['owned_cleanup_complete']
    assert status['remaining_owned_processes'] == [] and status['session_id'] == process.pid
    assert marker.exists()
    events = [json.loads(line) for line in (base/'events.jsonl').read_text().splitlines()]
    assert any(int(marker.read_text()) in [row['pid'] for row in event.get('remaining_owned_processes', [])] for event in events)
    assert json.loads(next(owners.glob('*.json')).read_text())['subreaper']
