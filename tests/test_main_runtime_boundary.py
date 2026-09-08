"""Main runtime validation and explicit launcher selection regressions."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import check_paper_runtime as runtime


def test_main_runtime_checks_current_interpreter_and_frozen_lock(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stderr='')
    monkeypatch.setattr(runtime.subprocess, 'run', run)
    runtime._check_main_runtime()
    assert calls[0][0] == ['uv', 'pip', 'check', '--python', str(Path(sys.executable).absolute())]
    assert calls[1][0] == ['uv', 'sync', '--check', '--frozen', '--no-install-project', '--offline']
    assert calls[1][1]['env']['UV_PROJECT_ENVIRONMENT'] == str(Path(sys.prefix).absolute())


@pytest.mark.parametrize('strategy', ['prehop', 'naive', 'ms_graphrag', 'lightrag', 'hipporag2', 'gfm_rag', 'linear_rag'])
def test_dependency_conflicts_reject_main_strategies_before_success(monkeypatch, strategy):
    monkeypatch.setattr('core.paper_policy.validate_paper_semantic_environment', lambda *args: None)
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr='incompatible packages'))
    with pytest.raises(RuntimeError, match='incompatible packages'):
        runtime.check(strategy, 'musique')


@pytest.mark.parametrize('selection', ['uv_only', 'python_only', 'both', 'mismatch', 'missing'])
def test_shell_explicit_runtime_selection_preserves_environment(tmp_path, selection):
    env_root = tmp_path / 'fresh'
    python = env_root / 'bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nexit 0\n')
    python.chmod(0o755)
    environment = os.environ.copy()
    environment.pop('PYTHON_BIN', None)
    environment.pop('UV_PROJECT_ENVIRONMENT', None)
    if selection in {'uv_only', 'both', 'mismatch'}:
        environment['UV_PROJECT_ENVIRONMENT'] = 'fresh'
    if selection in {'python_only', 'both', 'mismatch', 'missing'}:
        environment['PYTHON_BIN'] = str(python if selection in {'python_only', 'both'} else tmp_path / 'missing')
    lib = Path(__file__).resolve().parents[1] / 'scripts/lib.sh'
    result = subprocess.run(['bash', '-c', '. "$1"; resolve_python "$2"', 'test', str(lib), str(tmp_path)],
                            env=environment, capture_output=True, text=True, check=False)
    assert (result.returncode == 0) == (selection not in {'mismatch', 'missing'})
    if result.returncode == 0:
        assert result.stdout.strip() == str(python)
    assert python.read_text() == '#!/bin/sh\nexit 0\n'


def test_reuse_cannot_skip_failed_runtime_preflight(tmp_path):
    import shutil
    root = Path(__file__).resolve().parents[1]
    for relative in ('scripts/run_paper_target.sh', 'scripts/lib.sh', 'core/strategy_registry.py'):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    shutil.copytree(root / 'core', tmp_path / 'core', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    (tmp_path / '.env').write_text('# synthetic\n')
    (tmp_path / 'data/results/completed').mkdir(parents=True)
    python = tmp_path / 'fake-python'
    python.write_text('#!/bin/sh\nif [ "$1" = -c ]; then exec '+sys.executable+' "$@"; fi\nprintf "%s\\n" "$1" >> "$TRACE"\n'
                      '[ "$1" != scripts/check_paper_runtime.py ]\n')
    python.chmod(0o755)
    environment = {'PATH': os.defpath, 'PYTHON_BIN': str(python), 'TRACE': str(tmp_path / 'trace'),
                   'RAG_SKIP_PROJECT_ENV': 'true', 'RAG_INFERENCE_BASE_URL': 'http://litellm.test/v1',
                   'RAG_INFERENCE_API_KEY': 'synthetic', 'RAG_GENERATION_MODEL': 'gemma-4-31b-it',
                   'RAG_EMBEDDING_MODEL': 'qwen3-embedding-4b'}
    result = subprocess.run(['bash', 'scripts/run_paper_target.sh', 'musique', 'prehop', 'completed', '--check'],
                            cwd=tmp_path, env=environment, capture_output=True, check=False)
    assert result.returncode != 0
    assert (tmp_path / 'trace').read_text().splitlines() == ['scripts/check_paper_runtime.py']
