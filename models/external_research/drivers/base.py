from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.semantic_config import parse_strict_bool


@runtime_checkable
class ResearchDriver(Protocol):
    def index(self) -> dict[str, Any]: ...
    def query(self, question: str) -> list[dict[str, Any]] | dict[str, Any]: ...
    def close(self) -> None: ...


def load_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows = json.loads((output_dir / "input" / "corpus.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("staged corpus must be a non-empty JSON list")
    ids = [str(row.get("source_id", "")) for row in rows if isinstance(row, dict)]
    if len(ids) != len(rows) or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("staged corpus source identities are empty or duplicated")
    return rows


def positive_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def canonical_semantic_env(name: str, expected: str | int | bool) -> str | int | bool:
    """Reject, rather than silently overwrite, paper-semantic env drift."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return expected
    if isinstance(expected, bool):
        try:
            normalized: str | int | bool = parse_strict_bool(raw, name=name)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
    elif isinstance(expected, int):
        normalized = int(raw)
    else:
        normalized = raw.strip()
    if normalized != expected:
        raise RuntimeError(f"{name}={raw!r} differs from the checked-in paper semantic policy")
    return expected


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for this official adapter")
    return value


def endpoint_env(alias: str, canonical: str) -> str:
    value = (os.environ.get(alias) or os.environ.get(canonical) or "").strip()
    if not value:
        raise RuntimeError(f"{canonical} (or compatibility alias {alias}) is required")
    return value


def model_env(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback).strip() or fallback


def validate_documents(strategy: str, documents: Any, staged_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(documents, list):
        raise TypeError(f"{strategy} documents must be a list")
    seen: set[str] = set()
    clean = []
    for row in documents:
        if not isinstance(row, dict):
            raise TypeError(f"{strategy} document must be an object")
        source_id = str(row.get("source_id", ""))
        score = row.get("score")
        if not source_id or source_id not in staged_ids or source_id in seen:
            raise ValueError(f"{strategy} returned missing, foreign, or duplicate source identity")
        if score is not None and (
            isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
        ):
            raise ValueError(f"{strategy} returned a non-finite score")
        seen.add(source_id)
        clean.append(dict(row))
    return clean
