#!/usr/bin/env python3
"""Fail-closed, non-secret runtime preflight for paper methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.runtime_requirements import load_runtime_requirements
from core.strategy_registry import PRIMARY_STRATEGIES, get_strategy
from models.official_baseline_runtime import official_python, official_root, validate_runtime


def _load_runner_environment() -> None:
    """Load the same project file as shell runners without overriding exports."""
    from dotenv import load_dotenv

    if os.environ.get("RAG_SKIP_PROJECT_ENV") == "true":
        return
    env_path = ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


def _check_transport(strategy: str) -> None:
    from core.inference_transport import InferenceTransport

    spec = get_strategy(strategy)
    if spec.transport_profile != "openai_compatible_litellm":
        raise RuntimeError(f"{strategy} has an unsupported paper transport profile")
    transport = InferenceTransport.resolve(strategy)
    if transport.generation_base_url.rstrip("/") != transport.embedding_base_url.rstrip("/"):
        raise RuntimeError("paper generation and embedding must use the same LiteLLM gateway")
    if transport.generation_model != spec.paper_generation_model:
        raise RuntimeError(f"{strategy} generation model differs from the registry paper identity")
    if (
        spec.local_embedding_revision is None
        and strategy != "gfm_rag"
        and transport.embedding_model != spec.paper_embedding_model
    ):
        raise RuntimeError(f"{strategy} embedding model differs from the registry paper identity")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required runtime setting is missing: {name}")
    return value


def _required_sha(name: str) -> str:
    value = _required_env(name).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError(f"{name} must contain a SHA-256 digest")
    return value


def _check_python_contract(python: Path, requirement: dict[str, Any]) -> None:
    expected_python = str(requirement.get("python_version") or "")
    imports = requirement.get("required_imports") or []
    spacy_model = requirement.get("spacy_model")
    spacy_distribution = str(requirement.get("spacy_distribution") or "")
    spacy_version = spacy_distribution.rsplit("-", 1)[-1] if spacy_distribution else ""
    script = (
        "import importlib,json,sys;from importlib import metadata;"
        f"assert f'{{sys.version_info.major}}.{{sys.version_info.minor}}' == {expected_python!r}, "
        "f'Python version mismatch: {sys.version_info.major}.{sys.version_info.minor}';"
        f"[importlib.import_module(name) for name in {imports!r}];"
        + (
            "import spacy;"
            f"assert metadata.version({spacy_model!r}) == {spacy_version!r}, 'spaCy model version mismatch';"
            f"spacy.load({spacy_model!r});"
            if spacy_model
            else ""
        )
        + "print(json.dumps({'runtime':'ok'}))"
    )
    subprocess.run([str(python), "-c", script], check=True, cwd=ROOT)


def _check_local_snapshot(
    strategy: str,
    subdir: str,
    model: str,
    revision: str,
    required_files: list[str] | None = None,
    approved_file_sha256: dict[str, str] | None = None,
) -> Path:
    """Validate a strategy-local snapshot and its realized content manifest."""
    root = official_root(strategy).parent / "artifacts" / subdir
    manifest_path = root / "artifact_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{strategy} artifact manifest is missing or invalid: {manifest_path}") from exc
    expected = {"schema_version": 1, "repository_id": model, "revision": revision}
    if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{strategy} artifact manifest identifies a different snapshot: {manifest_path}")
    missing = [name for name in required_files or [] if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"{strategy} pinned model snapshot is missing files: {missing}")
    for filename, expected_sha256 in (approved_file_sha256 or {}).items():
        path = root / filename
        if not path.is_file() or _sha256(path) != expected_sha256:
            raise RuntimeError(f"{strategy} approved artifact digest differs: {path}")
    rows = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root)
        if relative.parts[0] == ".cache" or relative.as_posix() == "artifact_manifest.json":
            continue
        rows.append({"path": relative.as_posix(), "size": path.stat().st_size, "sha256": _sha256(path)})
    tree_sha256 = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (
        not rows
        or manifest.get("file_count") != len(rows)
        or manifest.get("total_bytes") != sum(row["size"] for row in rows)
        or manifest.get("tree_sha256") != tree_sha256
    ):
        raise RuntimeError(f"{strategy} local snapshot content differs from its artifact manifest: {root}")
    return root


def _check_frozen_runtime(python: Path) -> None:
    lock_path = python.parent.parent.parent / "runtime.freeze.txt"
    if not lock_path.is_file():
        raise RuntimeError(f"isolated runtime lock is missing: {lock_path}")
    actual = subprocess.run(
        ["uv", "pip", "freeze", "--python", str(python)],
        check=True,
        cwd=ROOT,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    expected = lock_path.read_text(encoding="utf-8").splitlines()
    if sorted(actual) != sorted(expected):
        raise RuntimeError(f"isolated runtime dependencies differ from {lock_path}")
    subprocess.run(["uv", "pip", "check", "--python", str(python)], check=True, cwd=ROOT, capture_output=True)


def _check_main_runtime() -> None:
    """Validate the running environment without synchronizing or repairing it."""
    python = Path(sys.executable).absolute()
    environment = os.environ.copy()
    environment["UV_PROJECT_ENVIRONMENT"] = str(Path(sys.prefix).absolute())
    for command in (
        ["uv", "pip", "check", "--python", str(python)],
        ["uv", "sync", "--check", "--frozen", "--no-install-project", "--offline"],
    ):
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError("main runtime dependency validation failed; preserve this environment and "
                               "prepare a fresh environment from uv.lock: " + result.stderr.strip())


def _check_approved_constraints(requirement: dict[str, Any]) -> None:
    relative = requirement.get("constraints_file")
    expected = requirement.get("constraints_sha256")
    if relative is None and expected is None:
        return
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError("approved runtime constraint identity is incomplete")
    path = (ROOT / relative).resolve()
    if ROOT not in path.parents or not path.is_file() or _sha256(path) != expected:
        raise RuntimeError(f"approved runtime constraints are missing or modified: {path}")


def check(strategy: str, dataset: str | None = None) -> None:
    spec = get_strategy(strategy)
    requirements = load_runtime_requirements()
    requirement = requirements.get(strategy)
    from core.paper_policy import validate_paper_semantic_environment

    if dataset not in {"multihoprag", "musique"}:
        raise RuntimeError("paper runtime preflight requires an explicit supported dataset")
    validate_paper_semantic_environment(strategy, dataset)
    _check_main_runtime()
    if strategy in {"prehop", "naive"}:
        return
    if not isinstance(requirement, dict):
        raise TypeError(f"paper runtime requirements are missing for {strategy}")
    _check_approved_constraints(requirement)

    if spec.external:
        validate_runtime(strategy)
        python = official_python(strategy)
        _check_frozen_runtime(python)
    else:
        python = Path(sys.executable)
    _check_python_contract(python, requirement)

    for distribution, expected in (requirement.get("distribution_versions") or {}).items():
        actual = metadata.version(distribution)
        if actual != expected:
            raise RuntimeError(f"{strategy} {distribution} version mismatch: expected {expected}, got {actual}")

    embedding_model = requirement.get("embedding_model")
    embedding_revision = requirement.get("embedding_revision")
    if embedding_model or embedding_revision:
        if (embedding_model, embedding_revision) != (spec.paper_embedding_model, spec.local_embedding_revision):
            raise RuntimeError(f"{strategy} runtime manifest and strategy registry embedding identity differ")
        subdir = requirement.get("embedding_snapshot_subdir")
        if not isinstance(subdir, str):
            raise RuntimeError(f"{strategy} runtime manifest lacks embedding_snapshot_subdir")
        _check_local_snapshot(
            strategy,
            subdir,
            str(embedding_model),
            str(embedding_revision),
            approved_file_sha256=dict(requirement.get("embedding_file_sha256") or {}),
        )

    entity_model = requirement.get("entity_linker_model")
    entity_revision = requirement.get("entity_linker_revision")
    if entity_model or entity_revision:
        if not entity_model or not entity_revision:
            raise RuntimeError(f"{strategy} entity-linker identity is incomplete")
        subdir = requirement.get("entity_linker_snapshot_subdir")
        if not isinstance(subdir, str):
            raise RuntimeError(f"{strategy} runtime manifest lacks entity_linker_snapshot_subdir")
        _check_local_snapshot(
            strategy,
            subdir,
            str(entity_model),
            str(entity_revision),
            approved_file_sha256=dict(requirement.get("entity_linker_file_sha256") or {}),
        )

    checkpoint_model = requirement.get("checkpoint_model")
    checkpoint_revision = requirement.get("checkpoint_revision")
    if checkpoint_model or checkpoint_revision:
        if not checkpoint_model or not checkpoint_revision:
            raise RuntimeError(f"{strategy} checkpoint identity is incomplete")
        subdir = requirement.get("checkpoint_snapshot_subdir")
        if not isinstance(subdir, str):
            raise RuntimeError(f"{strategy} runtime manifest lacks checkpoint_snapshot_subdir")
        _check_local_snapshot(
            strategy,
            subdir,
            str(checkpoint_model),
            str(checkpoint_revision),
            list(requirement.get("checkpoint_files") or []),
            dict(requirement.get("checkpoint_file_sha256") or {}),
        )

    artifact_root_env = requirement.get("artifact_root_env")
    for filename, sha_env in (requirement.get("required_files") or {}).items():
        path = Path(_required_env(str(artifact_root_env))) / filename
        if not path.is_file() or _sha256(path) != _required_sha(str(sha_env)):
            raise RuntimeError(f"{strategy} {filename} is missing or has the wrong digest")

    if strategy == "youtu_graphrag":
        from core.paper_policy import approved_youtu_schema

        if dataset not in {"multihoprag", "musique"}:
            raise RuntimeError("Youtu preflight requires a supported dataset for schema approval")
        schema = approved_youtu_schema(dataset)
        path = official_root(strategy) / schema["path"]
        if not path.is_file() or _sha256(path) != schema["sha256"]:
            raise RuntimeError(f"{strategy} checkout does not contain the approved schema for {dataset}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=PRIMARY_STRATEGIES, required=True)
    parser.add_argument("--dataset", choices=("multihoprag", "musique"))
    args = parser.parse_args()
    _load_runner_environment()
    _check_transport(args.strategy)
    check(args.strategy, args.dataset)
    print(f"paper runtime preflight passed: {args.strategy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
