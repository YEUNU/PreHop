"""Shell entrypoint contracts affecting experiment reproducibility."""

import subprocess
from pathlib import Path

import pytest


def test_project_env_preserves_exported_overrides(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("RAG_GRAPH_HOP_DEPTH=1\nRAG_HYPO_CHANNEL_VARIANT=full\n", encoding="utf-8")
    project_root = Path(__file__).resolve().parents[1]
    command = (
        f". {project_root / 'scripts/lib.sh'}; "
        f"load_project_env {env_file}; "
        'printf "%s %s" "$RAG_GRAPH_HOP_DEPTH" "$RAG_HYPO_CHANNEL_VARIANT"'
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "RAG_GRAPH_HOP_DEPTH": "0",
        "RAG_HYPO_CHANNEL_VARIANT": "single_combined",
    }

    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.stdout == "0 single_combined"


@pytest.mark.parametrize('skip_file', [True, False])
def test_python_runner_loads_exports_then_profile_even_when_dotenv_is_skipped(tmp_path, monkeypatch, skip_file):
    import json
    import os

    from scripts import runner_environment

    monkeypatch.setattr(runner_environment, 'ROOT', tmp_path)
    (tmp_path / '.env').write_text('RAG_GENERATION_CONCURRENCY=12\nRAG_TEST_FROM_DOTENV=loaded\n')
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'settings': {'generation_concurrency': 3}}))
    monkeypatch.setenv('RAG_SKIP_PROJECT_ENV', str(skip_file).lower())
    monkeypatch.setenv('RAG_GENERATION_CONCURRENCY', '15')
    monkeypatch.delenv('RAG_TEST_FROM_DOTENV', raising=False)
    monkeypatch.delenv('RAG_EXECUTION_PROFILE', raising=False)
    runner_environment._load_runner_environment()
    assert os.environ['RAG_GENERATION_CONCURRENCY'] == '15'
    assert ('RAG_TEST_FROM_DOTENV' in os.environ) is not skip_file
    monkeypatch.setenv('RAG_EXECUTION_PROFILE', str(profile))
    runner_environment._load_runner_environment()
    assert os.environ['RAG_GENERATION_CONCURRENCY'] == '3'


def test_shell_exports_quote_dotenv_values(tmp_path, monkeypatch, capsys):
    import json
    import os
    import sys

    from core.inference_transport import _FORBIDDEN_AMBIENT_PROVIDER_KEYS
    from scripts import runner_environment

    for key in _FORBIDDEN_AMBIENT_PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(runner_environment, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['-c'])
    monkeypatch.setenv('RAG_SKIP_PROJECT_ENV', 'false')
    monkeypatch.setenv('RAG_INFERENCE_BASE_URL', 'http://gateway.test/v1')
    monkeypatch.setenv('RAG_INFERENCE_API_KEY', 'fixture-key')
    monkeypatch.setenv('RAG_GENERATION_MODEL', 'generation')
    monkeypatch.setenv('RAG_EMBEDDING_MODEL', 'embedding')
    monkeypatch.delenv('RAG_TEST_QUOTED', raising=False)
    value = 'a space; $(touch injected) `touch injected` "quoted"'
    (tmp_path / '.env').write_text(f"RAG_TEST_QUOTED='{value}'\n")
    runner_environment.export_runner_environment()
    assignments = capsys.readouterr().out
    env = {'PATH': os.environ['PATH'], 'ASSIGNMENTS': assignments, 'TEST_PYTHON': sys.executable}
    result = subprocess.run(['bash', '-c', '''
eval "$ASSIGNMENTS"
"$TEST_PYTHON" -c 'import json, os; print(json.dumps(os.environ["RAG_TEST_QUOTED"]))'
'''], env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == value
    assert not (tmp_path / 'injected').exists()
