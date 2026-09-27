#!/usr/bin/env python3
"""Install and verify HopRAG's pinned main/POS environments and native POS model."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str | Path], **kwargs) -> subprocess.CompletedProcess:
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    try:
        return subprocess.run([str(arg) for arg in args], cwd=ROOT, env=environment, check=True, text=True, **kwargs)
    except subprocess.CalledProcessError as exc:
        if kwargs.get('capture_output'):
            sys.stderr.write((exc.stdout or '') + (exc.stderr or ''))
        raise


def constraints(requirement: dict, prefix: str = '') -> Path:
    path = ROOT / requirement[f'{prefix}constraints_file']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != requirement[f'{prefix}constraints_sha256']:
        raise RuntimeError(f'Constraint digest differs: {path}')
    return path


def requirements(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')}


def verify_runtime(target: Path, requirement: dict, revision: str) -> dict:
    """Setup-only verification; benchmark dispatch does not call this function."""
    from scripts.build_official_package import verify_source

    verify_source(target / 'source', revision)
    freezes = {}
    for name, prefix in [('main-env', ''), ('pos-env', 'native_pos_')]:
        python = target / name / 'bin/python'
        version = run([python, '-B', '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],
                      capture_output=True).stdout.strip()
        if version != requirement[f'{prefix}python_version']:
            raise RuntimeError(f'Unexpected Python version in {python}: {version}')
        frozen = run(['uv', 'pip', 'freeze', '--python', python], capture_output=True).stdout
        if set(frozen.splitlines()) != requirements(constraints(requirement, prefix)):
            raise RuntimeError(f'Installed packages differ in {target / name}; select a fresh RAG_OFFICIAL_BASELINE_HOME')
        freeze = target / f'{name}.freeze.txt'
        if frozen != freeze.read_text():
            raise RuntimeError(f'Installed packages differ from {freeze}')
        run(['uv', 'pip', 'check', '--python', python])
        freezes[name] = hashlib.sha256(freeze.read_bytes()).hexdigest()

    model_hashes = json.loads((ROOT / 'configs/hoprag_pos_model.json').read_text())
    for name, expected in model_hashes.items():
        if hashlib.sha256((target / 'pos-model' / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'POS model digest differs: {name}')

    run([target / 'main-env/bin/python', '-B', '-c',
         ('import sys; from models.hoprag.native_runtime import _install_import_bridges; '
         '_install_import_bridges(); sys.path.insert(0, sys.argv[1]); '
          'import HopBuilder, HopRetriever, HopGenerator'), target / 'source'], capture_output=True)
    verification = target / 'verification'
    reply = run([target / 'pos-env/bin/python', '-B', '-c',
                 ('import sys; from pathlib import Path; from models.hoprag.pos_worker import serve; '
                  'serve(Path(sys.argv[1]), Path(sys.argv[2]))'), target / 'pos-model', verification],
                input=json.dumps(['A researcher reads a document.']) + '\n', capture_output=True)
    payload = json.loads(reply.stdout)
    if not payload.get('result') or 'error' in payload:
        raise RuntimeError(f'Native POS verification failed: {payload}')
    verify_source(target / 'source', revision)
    return {'upstream_revision': revision, 'environment_sha256': freezes, 'pos_model_sha256': model_hashes}


def build_runtime(target: Path, requirement: dict, repository: str, revision: str) -> None:
    source = target / 'source'
    run(['git', 'init', source])
    run(['git', '-C', source, 'remote', 'add', 'origin', repository])
    run(['git', '-C', source, 'fetch', '--depth', '1', 'origin', revision])
    run(['git', '-C', source, 'checkout', '--detach', revision])
    for name, prefix in [('main-env', ''), ('pos-env', 'native_pos_')]:
        lock = constraints(requirement, prefix)
        run(['uv', 'venv', '--python', requirement[f'{prefix}python_version'], target / name])
        python = target / name / 'bin/python'
        command = ['uv', 'pip', 'install', '--python', python, '-r', lock]
        if name == 'main-env':
            command += ['--extra-index-url', 'https://download.pytorch.org/whl/cpu',
                        '--index-strategy', 'unsafe-best-match']
        run(command)
        frozen = run(['uv', 'pip', 'freeze', '--python', python], capture_output=True).stdout
        (target / f'{name}.freeze.txt').write_text(frozen)
    model_dir = target / 'pos-model'
    model_dir.mkdir()
    for name in json.loads((ROOT / 'configs/hoprag_pos_model.json').read_text()):
        with (urllib.request.urlopen(requirement['native_pos_model_base_url'] + name, timeout=120) as response,
              (model_dir / name).open('wb') as output):
            shutil.copyfileobj(response, output)


def install_runtime(home: Path, requirement: dict, repository: str, revision: str) -> Path:
    """Publish only a verified fresh build; never relocate or mutate existing venvs."""
    home.parent.mkdir(parents=True, exist_ok=True)
    with (home.parent / '.runtime.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if home.exists() or home.is_symlink():
            verify_runtime(home, requirement, revision)
            return home
        builds = home.parent / '.builds'
        builds.mkdir(exist_ok=True)
        target = Path(tempfile.mkdtemp(prefix='hoprag-', dir=builds))
        print(f'Preparing HopRAG in {target}', flush=True)
        build_runtime(target, requirement, repository, revision)
        receipt = verify_runtime(target, requirement, revision)
        (target / 'setup.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
        home.symlink_to(target.relative_to(home.parent), target_is_directory=True)
    return home


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    sys.path.insert(0, str(ROOT))
    from core.runtime_requirements import runtime_requirement
    from core.strategy_registry import get_strategy
    from models.hoprag.runtime_paths import runtime_home

    spec = get_strategy('hoprag')
    home = install_runtime(runtime_home(), runtime_requirement('hoprag'), spec.repository, spec.revision)
    print(f'HopRAG runtime ready: {home}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
