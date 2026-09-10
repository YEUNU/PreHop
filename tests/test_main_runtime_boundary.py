"""Main runtime validation and explicit launcher selection regressions."""
import os
import subprocess
from pathlib import Path

import pytest


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
