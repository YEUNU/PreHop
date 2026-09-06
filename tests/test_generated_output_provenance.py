"""Native output creation must preserve a clean execution worktree's plan."""
import subprocess
import sys
from pathlib import Path

from scripts.paper_campaign import build_steps, check_plan
from utils.provenance import code_provenance

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ('lightrag', 'hipporag2', 'gfm_rag', 'linear_rag', 'youtu_graphrag')


def test_actual_clean_worktree_native_outputs_preserve_provenance_and_plan(tmp_path, monkeypatch):
    repository = tmp_path/'fixture_repository'
    checkout = tmp_path/'execution_worktree'
    repository.mkdir()
    def git(*args, cwd=repository):
        return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True)
    git('init', '--quiet')
    git('config', 'user.name', 'Isolated Test Fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    (repository/'.gitignore').write_text((ROOT/'.gitignore').read_text())
    (repository/'owned.py').write_text('VALUE = 1\n')
    git('add', '.gitignore', 'owned.py')
    # This commit belongs only to the newly created temporary fixture repository.
    git('commit', '--quiet', '-m', 'Create isolated provenance fixture')
    git('worktree', 'add', '--quiet', '--detach', str(checkout), 'HEAD')
    before = code_provenance(checkout)
    assert before['dirty'] is False
    selected = {'PYTHON_BIN': sys.executable, 'UV_PROJECT_ENVIRONMENT': sys.prefix}
    monkeypatch.setattr('scripts.paper_stage_runner.selected_python_environment', lambda: selected)
    monkeypatch.setattr('scripts.paper_gate_ledger._context', lambda: {'version': 'fixture', 'targets': {'model': {'temperature': 0}}})
    plan = {'campaign': 'fixture', 'attempt': 'a1', 'python': sys.executable, 'python_prefix': sys.prefix,
            'commit': before['revision'], 'context': {'version': 'fixture', 'targets': {'model': {'temperature': 0}}},
            'steps': build_steps('fixture', 'a1', sys.executable)}
    check_plan(plan)
    for method in OUTPUTS:
        artifact = checkout/'data'/f'{method}_output/runs/fixture/multihoprag/native/index.json'
        artifact.parent.mkdir(parents=True)
        artifact.write_text('{"native_fixture_artifact":true}\n')
        assert git('check-ignore', str(artifact.relative_to(checkout)), cwd=checkout).stdout.strip()
        assert code_provenance(checkout) == before
        check_plan(plan)
    assert git('status', '--porcelain', cwd=checkout).stdout == ''
    (checkout/'owned.py').write_text('VALUE = 2\n')
    assert code_provenance(checkout)['dirty'] is True
    check_plan(plan)  # Project source state is informational; configuration is unchanged.
    git('add', 'owned.py', cwd=checkout)
    git('commit', '--quiet', '-m', 'Unrelated source change', cwd=checkout)
    assert code_provenance(checkout)['revision'] != before['revision']
    check_plan(plan)
