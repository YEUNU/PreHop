import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


@contextmanager
def atomic_text_writer(path: Path) -> Iterator[TextIO]:
    """Publish a complete UTF-8 file or preserve the previous file on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            yield file
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, payload: Any) -> None:
    with atomic_text_writer(path) as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with atomic_text_writer(path) as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
