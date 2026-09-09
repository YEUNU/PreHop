"""Real packaging regression: only disposable test checkouts/artifacts are written."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.build_official_package import install_package, stage_source, verify_source


def _checkout(tmp_path):
    source = tmp_path / 'runtime/source'
    source.mkdir(parents=True)
    (source / 'setup.py').write_text(
        'from setuptools import setup\nsetup(name="prehop-fixture", version="0.0.1", py_modules=["fixture_module"])\n'
    )
    (source / 'fixture_module.py').write_text('VALUE = 1\n')
    (source / '.gitignore').write_text('build/\n*.egg-info/\n')
    for command in (
        ['git', 'init', '--quiet'], ['git', 'add', '.'],
        ['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
         '-c', 'commit.gpgsign=false', 'commit', '--quiet', '-m', 'synthetic packaging fixture'],
    ):
        subprocess.run(command, cwd=source, check=True, capture_output=True)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source, check=True,
                              capture_output=True, text=True).stdout.strip()
    return source, revision


def _tracked_bytes(source):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source.iterdir() if path.is_file()}


@pytest.mark.parametrize("entrypoint", ["imported", "system_python_cli"])
def test_real_setuptools_build_uses_export_and_preserves_checkout(tmp_path, monkeypatch, entrypoint):
    source, revision = _checkout(tmp_path)
    before = _tracked_bytes(source)
    artifacts = source.parent / 'artifacts'
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    uv = bin_dir / 'uv'
    # Exercise the helper's installation boundary using real setuptools code,
    # without downloading build dependencies or modifying any installed env.
    uv.write_text('#!' + sys.executable + '\nimport subprocess,sys\n'
                  'subprocess.run([sys.executable,"setup.py","build","egg_info"],'
                  'cwd=sys.argv[-1],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n')
    uv.chmod(0o755)
    monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ.get('PATH', ''))
    constraint = tmp_path / 'constraints.txt'
    constraint.write_text('')
    if entrypoint == "system_python_cli":
        helper = Path(__file__).resolve().parents[1] / "scripts/build_official_package.py"
        # Exercise the production system-python entrypoint, including stdlib
        # compatibility, before the fake uv boundary runs real setuptools.
        launcher = shutil.which("python3")
        assert launcher is not None
        subprocess.run([
            launcher, str(helper), "--source", str(source), "--revision", revision,
            "--artifacts", str(artifacts), "--python", sys.executable,
            "--constraint", str(constraint),
        ], check=True, capture_output=True, text=True)
        staged, = artifacts.glob("builds/*/source")
    else:
        staged = install_package(source, revision, artifacts, Path(sys.executable), constraint, [])
    manifest = json.loads((staged.parent / "build_manifest.json").read_text())
    assert manifest["source_archive_sha256"] == hashlib.sha256((staged.parent / "source.tar").read_bytes()).hexdigest()
    assert artifacts in staged.parents
    assert (staged / 'build/lib/fixture_module.py').is_file()
    assert list(staged.glob('*.egg-info/PKG-INFO'))
    assert not (source / 'build').exists()
    assert not list(source.glob('*.egg-info'))
    assert _tracked_bytes(source) == before
    verify_source(source, revision)
    again = stage_source(source, revision, artifacts)
    assert again != staged
    assert (staged / 'build/lib/fixture_module.py').is_file()


@pytest.mark.parametrize('generated', ['build/lib/cache.py', 'untracked.txt'])
def test_existing_ignored_or_untracked_source_files_are_preserved_and_rejected(tmp_path, generated):
    source, revision = _checkout(tmp_path)
    path = source / generated
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('preserve this file')
    with pytest.raises(RuntimeError, match='tracked, untracked, or ignored'):
        stage_source(source, revision, source.parent / 'artifacts')
    assert path.read_text() == 'preserve this file'
    assert not (source.parent / 'artifacts').exists()


def test_build_artifacts_cannot_be_placed_under_checkout(tmp_path):
    source, revision = _checkout(tmp_path)
    with pytest.raises(ValueError, match='outside'):
        stage_source(source, revision, source / 'artifacts')
    verify_source(source, revision)


def test_alternate_runtime_home_is_shared_by_source_and_python(tmp_path, monkeypatch):
    from models.official_baseline_runtime import official_python, official_root
    monkeypatch.delenv('RAG_LIGHTRAG_ROOT', raising=False)
    monkeypatch.delenv('RAG_LIGHTRAG_PYTHON', raising=False)
    home = tmp_path / 'fresh-runtime'
    monkeypatch.setenv('RAG_OFFICIAL_BASELINE_HOME', str(home))
    assert official_root('lightrag') == home / 'lightrag/source'
    assert official_python('lightrag') == home / 'lightrag/venv/bin/python'


def test_relative_installer_paths_remain_absolute_without_resolving_venv_symlink(tmp_path, monkeypatch):
    source, revision = _checkout(tmp_path)
    python = tmp_path / 'venv/bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    constraint = tmp_path / 'constraints.txt'
    constraint.write_text('')
    original_run = subprocess.run
    observed = []
    def record(command, **kwargs):
        if command[0] == 'uv':
            observed.append((command, kwargs['cwd']))
            return subprocess.CompletedProcess(command, 0)
        return original_run(command, **kwargs)
    monkeypatch.setattr(subprocess, 'run', record)
    monkeypatch.chdir(tmp_path)
    staged = install_package(source.relative_to(tmp_path), revision, Path('runtime/artifacts'),
                             Path('venv/bin/python'), Path('constraints.txt'), [])
    command, cwd = observed[0]
    assert command[command.index('--python') + 1] == str(python)
    assert command[command.index('--constraint') + 1] == str(constraint)
    assert cwd == staged.parent
    assert python != python.resolve()
