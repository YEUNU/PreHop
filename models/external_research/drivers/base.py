from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ResearchDriver(Protocol):
    def index(self) -> dict[str, Any]: ...
    def query(self, question: str) -> list[dict[str, Any]] | dict[str, Any]: ...
    def close(self) -> None: ...


def load_rows(output_dir: Path) -> list[dict[str, Any]]:
    return json.loads((output_dir / "input" / "corpus.json").read_text(encoding="utf-8"))


def document_rows(documents: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in documents]
