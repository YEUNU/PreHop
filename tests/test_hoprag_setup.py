"""Installation publishes only usable environments without mutating old ones."""
import hashlib
import io
import json
import sys
from types import SimpleNamespace

import pytest

from models.hoprag import pos_worker, runtime_paths
from scripts import setup_hoprag_runtime as setup


def test_shared_home_honors_existing_override_without_resolving_symlinks(tmp_path, monkeypatch):
    from core.runtime_requirements import method_main_python
    from models.hoprag import native_runtime
    physical = tmp_path / 'physical'
    physical.mkdir()
    home = tmp_path / 'alias'
    home.symlink_to(physical, target_is_directory=True)
    monkeypatch.setenv('RAG_OFFICIAL_BASELINE_HOME', str(home))
    assert runtime_paths.runtime_home() == home / 'hoprag'
    assert native_runtime.runtime_home() == home / 'hoprag'
    assert method_main_python('hoprag') == str(home / 'hoprag/main-env/bin/python')


def test_explicit_runtime_environment_takes_precedence(tmp_path, monkeypatch):
    from core.runtime_requirements import method_main_python, runtime_identity
    from models.official_baseline_runtime import official_root

    monkeypatch.setenv('RAG_OFFICIAL_BASELINE_HOME', str(tmp_path / 'ambient'))
    selected = tmp_path / 'selected'
    environment = {'RAG_OFFICIAL_BASELINE_HOME': str(selected)}
    assert method_main_python('hoprag', environment=environment) == str(selected / 'hoprag/main-env/bin/python')
    assert official_root('lightrag', environment) == selected / 'lightrag/source'
    freeze = selected / 'lightrag/runtime.freeze.txt'
    freeze.parent.mkdir(parents=True)
    freeze.write_text('selected-package==1.0\n')
    identity = runtime_identity('lightrag', environment)
    assert identity['runtime_freeze']['path'] == str(freeze)
    assert identity['runtime_freeze']['sha256'] == hashlib.sha256(freeze.read_bytes()).hexdigest()


def test_setup_publishes_verified_build_without_moving_venv(tmp_path, monkeypatch):
    home = tmp_path / 'hoprag'
    built = []
    def build(target, *args):
        assert not home.exists()
        (target / 'interpreter-path').write_text(str(target))
        built.append(target)
    def verify(target, *args):
        assert (target / 'interpreter-path').read_text() == str(target.resolve())
        return {'verified': True}
    monkeypatch.setattr(setup, 'build_runtime', build)
    monkeypatch.setattr(setup, 'verify_runtime', verify)
    assert setup.install_runtime(home, {}, 'fixture', 'revision') == home
    assert home.is_symlink() and home.resolve() == built[0]
    assert json.loads((home / 'setup.json').read_text()) == {'verified': True}
    setup.install_runtime(home, {}, 'fixture', 'revision')
    assert len(built) == 1


@pytest.mark.parametrize('existing', [False, True])
def test_setup_failure_preserves_unpublished_build_or_existing_runtime(tmp_path, monkeypatch, existing):
    home = tmp_path / 'hoprag'
    if existing:
        home.mkdir()
        (home / 'old').write_text('keep')
    def build(target, *args):
        (target / 'failure-evidence').write_text('keep')
    def verify(*args):
        raise RuntimeError('fixture verification failure')
    monkeypatch.setattr(setup, 'build_runtime', build)
    monkeypatch.setattr(setup, 'verify_runtime', verify)
    with pytest.raises(RuntimeError, match='fixture verification failure'):
        setup.install_runtime(home, {}, 'fixture', 'revision')
    if existing:
        assert (home / 'old').read_text() == 'keep'
        assert not (tmp_path / '.builds').exists()
    else:
        assert not home.exists() and not home.is_symlink()
        assert len(list((tmp_path / '.builds').glob('*/failure-evidence'))) == 1


def test_pos_transport_keeps_model_immutable_and_errors_native(tmp_path, monkeypatch):
    model = tmp_path / 'model'
    model.mkdir()
    expected = json.loads((setup.ROOT / 'configs/hoprag_pos_model.json').read_text())
    for name in expected:
        (model / name).write_text(name)
    before = {name: hashlib.sha256((model / name).read_bytes()).hexdigest() for name in expected}
    def taskflow(name, *, device_id, task_path):
        assert name == 'pos_tagging' and device_id == -1
        from pathlib import Path
        (Path(task_path) / 'generated.cache').write_text('generated')
        print('native initialization output')
        def tag(text):
            print('native task output')
            if not text:
                raise ValueError('native empty input')
            return text
        return tag
    monkeypatch.setitem(sys.modules, 'paddlenlp', SimpleNamespace(Taskflow=taskflow))
    monkeypatch.setattr(sys, 'stdin', io.StringIO('["first"]\n[]\n["last"]\n'))
    output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', output)
    pos_worker.serve(model, tmp_path / 'output')
    assert [json.loads(line) for line in output.getvalue().splitlines()] == [
        {'result': ['first']}, {'error': 'ValueError: native empty input'}, {'result': ['last']},
    ]
    assert {name: hashlib.sha256((model / name).read_bytes()).hexdigest() for name in expected} == before
    assert not (model / 'generated.cache').exists()
    realized = list((tmp_path / 'output').glob('*/realized_files.json'))
    assert len(realized) == 1 and 'generated.cache' in json.loads(realized[0].read_text())
