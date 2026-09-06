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


def test_new_unit_launch_owns_descendant_term_cleanup_without_kill_escalation(tmp_path, monkeypatch):
    from scripts import paper_stage_runner

    monkeypatch.setattr(campaign, 'ROOT', tmp_path)
    monkeypatch.setattr(paper_stage_runner, 'ROOT', tmp_path)
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps({'campaign': 'cleanup-fixture', 'python': sys.executable}))
    monkeypatch.setattr(campaign, 'check_plan', lambda plan: None)
    monkeypatch.setattr(campaign, 'require_logout_persistence', lambda: None)
    monkeypatch.setattr(campaign, 'ensure_no_other_campaigns', lambda: None)
    monkeypatch.setattr(campaign, 'resource_lock_path', lambda: tmp_path / 'resource.lock')
    monkeypatch.setattr(campaign, 'safe_environment', dict)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[0] == 'systemd-run':
            unit = command[command.index('--unit') + 1]
            (tmp_path / 'status.json').write_text(json.dumps({
                'unit': unit, 'state': 'running', 'supervisor': campaign.identity(os.getpid()),
            }))
            return subprocess.CompletedProcess(command, 0)
        assert command[:3] == ['systemctl', '--user', 'show']
        return subprocess.CompletedProcess(command, 0, f'MainPID={os.getpid()}\nActiveState=active\nResult=success\n')

    monkeypatch.setattr(campaign.subprocess, 'run', run)
    receipt = campaign.launch(plan_path, backend='systemd')
    properties = [commands[0][i + 1] for i, value in enumerate(commands[0]) if value == '--property']
    assert 'KillMode=control-group' in properties
    assert 'SendSIGKILL=no' in properties and 'TimeoutStopSec=30' in properties
    assert 'Restart=no' in properties and 'KillMode=process' not in properties
    assert len(commands) == 2
    assert receipt['supervisor']['pid'] == os.getpid()


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


@pytest.mark.parametrize("report_descendant", [True, False])
def test_child_inherited_pipes_fail_with_owned_descendant_receipt(tmp_path, monkeypatch, report_descendant):
    """A real grandchild retaining both FDs cannot stall a failed stage forever."""
    monkeypatch.setattr(campaign, 'ROOT', tmp_path)
    monkeypatch.setattr(campaign, 'CHILD_LOG_DRAIN_SECONDS', .15)
    monkeypatch.setattr(campaign, 'resource_lock_path', lambda: tmp_path/'resource.lock')
    monkeypatch.setattr(campaign, 'ensure_no_other_campaigns', lambda unit: None)
    monkeypatch.setattr(campaign, 'check_plan', lambda plan: None)
    monkeypatch.setattr(campaign, 'validate_step', lambda plan, step: pytest.fail('incomplete stage was validated'))
    secret = 'fixture-secret-split-across-writes'
    monkeypatch.setattr(campaign, 'safe_environment', lambda: {**os.environ, 'RAG_INFERENCE_API_KEY': secret})
    marker = tmp_path/'descendant.json'
    grandchild = 'import time; time.sleep(2)'
    code = f'''
import os,subprocess,sys,time,json
from pathlib import Path
sys.path.insert(0,{str(ROOT)!r})
from scripts.paper_campaign import identity
p = subprocess.Popen([sys.executable,'-c',{grandchild!r}])
Path({str(marker)!r}).write_text(json.dumps(identity(p.pid)))
os.write(1,{secret[:12].encode()!r});time.sleep(.03)
os.write(1,{(secret[12:] + ' 문서\n').encode()!r})
os.write(2,b'complete stderr line\\n')
os.write(2,b'unfinished-sensitive-fragment')
'''
    plan = {'campaign': 'smoke', 'commit': 'fixture', 'steps': [{'id': 'fd-leak', 'argv': [sys.executable, '-c', code]}]}
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(plan))

    def owned(_unit):
        rows = [campaign.identity(os.getpid())]
        if report_descendant and marker.exists():
            row = json.loads(marker.read_text())
            if campaign.alive(row):
                rows.append(row)
        return rows

    monkeypatch.setattr(campaign, 'unit_processes', owned)
    started = time.monotonic()
    assert campaign.supervise(path, unit='fixture') == 1
    assert time.monotonic() - started < 1.5
    status = json.loads((tmp_path/'status.json').read_text())
    assert status['state'] == 'failed'
    assert status['log_drain_complete'] is False
    assert status['log_drain_incomplete_streams'] == ['stderr', 'stdout']
    assert status['log_drain_unwritten_partial_bytes']['stderr'] > 0
    assert status['remaining_owned_processes'] == ([json.loads(marker.read_text())] if report_descendant else [])
    assert campaign.alive(json.loads(marker.read_text()))
    logs = ''.join(p.read_text() for p in tmp_path.glob('*.log'))
    assert '[REDACTED] 문서\n' in logs and 'complete stderr line\n' in logs
    assert secret not in logs and 'unfinished-sensitive-fragment' not in logs
    assert status['completed_steps'] == []
    # The harmless descendant exits on its own; production does not signal it.
    time.sleep(2)


def test_normal_child_drains_large_streams_and_final_unterminated_line(tmp_path):
    secret = 'fixture-final-secret'
    code = f'import os;os.write(1,b"x"*200000+b"\\n");os.write(2,{secret.encode()!r})'
    updates = []
    with (tmp_path/'lock').open('w') as handle:
        assert campaign.run_child([sys.executable, '-c', code],
            {**os.environ, 'RAG_INFERENCE_API_KEY': secret}, tmp_path/'normal', handle, updates.append) == 0
    assert (tmp_path/'normal.stdout.log').read_text() == 'x' * 200000 + '\n'
    assert (tmp_path/'normal.stderr.log').read_text() == '[REDACTED]'
    assert updates[-1]['log_drain_complete'] is True
