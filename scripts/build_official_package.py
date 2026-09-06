#!/usr/bin/env python3
"""Build upstream packages from revision exports outside immutable checkouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path


def verify_source(source: Path, revision: str) -> None:
    """Reject tracked, untracked, and ignored changes without cleaning anything."""
    actual = subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], check=True, text=True, capture_output=True,
    ).stdout.strip()
    if actual != revision:
        raise RuntimeError('upstream source revision differs from the approved revision')
    status = subprocess.run(
        ['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all', '--ignored'],
        check=True, text=True, capture_output=True,
    ).stdout
    if status:
        raise RuntimeError('upstream source contains tracked, untracked, or ignored changes; preserve it and select a fresh runtime home')


def stage_source(source: Path, revision: str, artifacts: Path) -> Path:
    """Export committed bytes to a new artifacts/builds directory for packaging."""
    source = source.resolve()
    artifacts = artifacts.resolve()
    if source == artifacts or source in artifacts.parents:
        raise ValueError('build artifacts must be outside the upstream source checkout')
    verify_source(source, revision)
    builds = artifacts / 'builds'
    builds.mkdir(parents=True, exist_ok=True)
    build_root = Path(tempfile.mkdtemp(prefix=revision + '-', dir=builds))
    archive = build_root / 'source.tar'
    subprocess.run(
        ['git', '-C', str(source), 'archive', '--format=tar', '--output', str(archive), revision], check=True,
    )
    target = build_root / 'source'
    target.mkdir()
    with tarfile.open(archive) as stream:
        # Reject archive links/paths that escape the fresh build directory.
        stream.extractall(target, filter='data')
    archive_hash = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            archive_hash.update(chunk)
    digest = archive_hash.hexdigest()
    manifest = {'schema_version': 1, 'official_revision': revision, 'source_checkout': str(source),
                'source_archive_sha256': digest, 'build_source': str(target)}
    (build_root / 'build_manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    verify_source(source, revision)
    return target


def install_package(source: Path, revision: str, artifacts: Path, python: Path, constraint: Path, extras: list[str]) -> Path:
    # Resolve caller-relative paths before moving the installer working dir.
    # Preserve the final venv interpreter symlink: resolving it would select
    # the base interpreter and install outside the intended environment.
    python = Path(os.path.abspath(python.expanduser()))
    constraint = constraint.expanduser().resolve()
    staged = stage_source(source, revision, artifacts)
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    command = ['uv', 'pip', 'install', '--python', str(python), '--constraint', str(constraint), str(staged), *extras]
    try:
        subprocess.run(command, cwd=staged.parent, env=environment, check=True)
    finally:
        # Includes ignored build/cache outputs, which ordinary porcelain omits.
        verify_source(source, revision)
    return staged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--python', type=Path)
    parser.add_argument('--constraint', type=Path)
    parser.add_argument('--extra', action='append', default=[])
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        verify_source(args.source, args.revision)
    else:
        if args.artifacts is None or args.python is None or args.constraint is None:
            parser.error('package installation requires --artifacts, --python, and --constraint')
        install_package(args.source, args.revision, args.artifacts, args.python, args.constraint, args.extra)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
