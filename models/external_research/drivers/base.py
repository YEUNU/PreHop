from __future__ import annotations

import json
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
    return json.loads((output_dir / "input" / "corpus.json").read_text(encoding="utf-8"))


def positive_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
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
    return normalized


def document_rows(strategy: str, documents: Any, staged_ids: set[str]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    clean = []
    for row in documents:
        source_id = str(row.get("source_id", ""))
        row.get("score")
        seen.add(source_id)
        clean.append(dict(row))
    return clean
