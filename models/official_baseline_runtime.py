"""Process boundary and snapshot helpers for externally maintained baselines."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any

from utils.io import _write_json

OFFICIAL_REVISIONS = {
    "browsenet": "ba82eeceb089104de2999d00b744cd02583fe8a4",
    "proprag": "3ec103488abd5589e569ee0fdd6e0c7067e5b783",
}
OFFICIAL_REPOSITORIES = {
    "browsenet": "https://github.com/bisect-group/BrowseNet.git",
    "proprag": "https://github.com/ReLink-Inc/PropRAG.git",
}
_RESULT_PREFIX = "__PREHOP_OFFICIAL_RESULT__="
_ROOT = Path(__file__).resolve().parents[1]


def source_set_sha256(source_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(source_ids)).encode("utf-8")).hexdigest()


def corpus_records_sha256(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def official_root(strategy: str) -> Path:
    key = f"RAG_{strategy.upper()}_ROOT"
    return Path(os.environ.get(key, f"data/official_baselines/{strategy}/source")).resolve()


def official_python(strategy: str) -> Path:
    key = f"RAG_{strategy.upper()}_PYTHON"
    path = Path(os.environ.get(key, f"data/official_baselines/{strategy}/venv/bin/python")).expanduser()
    # Do not call resolve(): venv Python executables are commonly symlinks to
    # the base interpreter. Resolving that final link bypasses site-packages.
    return path if path.is_absolute() else Path.cwd() / path


def output_root(strategy: str) -> Path:
    key = f"RAG_{strategy.upper()}_OUTPUT_ROOT"
    return Path(os.environ.get(key, f"data/{strategy}_output")).resolve()


def corpus_output_dir(strategy: str, corpus_tag: str) -> Path:
    return output_root(strategy) / corpus_tag


def snapshot_metadata_path(strategy: str, corpus_tag: str) -> Path:
    return corpus_output_dir(strategy, corpus_tag) / "index_snapshot_metadata.json"


def _parse_staged_document(path: Path) -> tuple[str, str]:
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    title = path.stem
    body_start = 0
    if lines and lines[0].startswith("Title:"):
        title = lines[0].split(":", 1)[1].strip() or title
        body_start = 1
    if body_start < len(lines) and lines[body_start].startswith("Paragraph-ID:"):
        body_start += 1
    while body_start < len(lines) and not lines[body_start].strip():
        body_start += 1
    body = "\n".join(lines[body_start:]).strip()
    if not body:
        raise ValueError(f"Official baseline document has no indexable body: {path}")
    return title, body


def stage_corpus(strategy: str, dataset_path: str | Path, corpus_tag: str) -> tuple[list[dict[str, Any]], Path]:
    target = corpus_output_dir(strategy, corpus_tag)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(
            f"{strategy} output already exists for corpus {corpus_tag}: {target}. Use a new run-specific output root."
        )
    input_dir = target / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for path in sorted(Path(dataset_path).iterdir()):
        if path.suffix not in {".txt", ".md"}:
            continue
        title, body = _parse_staged_document(path)
        records.append({"source_id": path.stem, "title": title, "text": body})
    if not records:
        raise ValueError(f"No documents were staged for {strategy}: {dataset_path}")
    _write_json(input_dir / "corpus.json", records)
    return records, target


def validate_runtime(strategy: str) -> None:
    root = official_root(strategy)
    python = official_python(strategy)
    if not root.is_dir() or not (root / ".git").exists():
        raise RuntimeError(
            f"{strategy} official source is not installed at {root}. "
            "Run scripts/setup_official_baselines.sh first."
        )
    if not python.is_file():
        raise RuntimeError(
            f"{strategy} runtime is not installed at {python}. "
            "Run scripts/setup_official_baselines.sh first."
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()
    if revision != OFFICIAL_REVISIONS[strategy]:
        raise RuntimeError(
            f"{strategy} source revision mismatch: expected {OFFICIAL_REVISIONS[strategy]}, got {revision}"
        )


def _command(strategy: str, corpus_tag: str, mode: str) -> list[str]:
    return [
        str(official_python(strategy)),
        str(_ROOT / "scripts" / "official_baseline_worker.py"),
        "--strategy",
        strategy,
        "--mode",
        mode,
        "--official-root",
        str(official_root(strategy)),
        "--output-dir",
        str(corpus_output_dir(strategy, corpus_tag)),
        "--corpus-tag",
        corpus_tag,
    ]


def _runtime_env(strategy: str) -> dict[str, str]:
    env = os.environ.copy()
    runtime_bin = str(official_python(strategy).parent)
    env["PATH"] = runtime_bin + os.pathsep + env.get("PATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_index_worker(strategy: str, corpus_tag: str, request: dict[str, Any]) -> dict[str, Any]:
    validate_runtime(strategy)
    process = subprocess.Popen(
        _command(strategy, corpus_tag, "index"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
        cwd=_ROOT,
        env=_runtime_env(strategy),
    )
    if process.stdin is None or process.stdout is None:
        process.terminate()
        raise RuntimeError(f"{strategy} official index worker pipes were not created")
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.close()

    payload = None
    output_tail: deque[str] = deque(maxlen=200)
    for line in process.stdout:
        if line.startswith(_RESULT_PREFIX):
            payload = json.loads(line[len(_RESULT_PREFIX) :])
        else:
            output_tail.append(line)
            print(line, end="", file=sys.stderr, flush=True)
    returncode = process.wait()
    if returncode or not isinstance(payload, dict) or not payload.get("ok"):
        detail = (payload or {}).get("error") or "".join(output_tail)[-4000:] or f"exit status {returncode}"
        raise RuntimeError(f"{strategy} official index worker failed: {detail}")
    return payload


class OfficialQueryWorker:
    """One persistent official process per adapter to avoid reloading models."""

    def __init__(self, strategy: str, corpus_tag: str):
        validate_runtime(strategy)
        self.strategy = strategy
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            _command(strategy, corpus_tag, "serve"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            cwd=_ROOT,
            env=_runtime_env(strategy),
        )
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()

        def _read_stdout() -> None:
            assert self._process.stdout is not None
            for line in self._process.stdout:
                self._stdout_queue.put(line)
            self._stdout_queue.put(None)

        self._reader = threading.Thread(target=_read_stdout, daemon=True)
        self._reader.start()
        ready = self.request({"operation": "ready"})
        if not ready.get("ready"):
            raise RuntimeError(f"{strategy} official query worker did not become ready")

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._process.poll() is not None or self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError(f"{self.strategy} official query worker is not running")
            self._process.stdin.write(json.dumps(payload) + "\n")
            self._process.stdin.flush()
            timeout = float(os.environ.get("RAG_OFFICIAL_QUERY_TIMEOUT", "1800"))
            deadline = time.monotonic() + timeout
            while True:
                try:
                    line = self._stdout_queue.get(timeout=max(0.01, deadline - time.monotonic()))
                except queue.Empty as exc:
                    self._process.terminate()
                    raise TimeoutError(f"{self.strategy} official query exceeded {timeout:g} seconds") from exc
                if line is None:
                    raise RuntimeError(f"{self.strategy} official query worker exited without a response")
                if not line.startswith(_RESULT_PREFIX):
                    continue
                response = json.loads(line[len(_RESULT_PREFIX) :])
                if not response.get("ok"):
                    raise RuntimeError(f"{self.strategy} official query failed: {response.get('error')}")
                return response

    def close(self) -> None:
        if self._process.poll() is None and self._process.stdin is not None:
            try:
                self._process.stdin.write(json.dumps({"operation": "shutdown"}) + "\n")
                self._process.stdin.flush()
                self._process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self._process.terminate()

    def __del__(self):  # pragma: no cover - best-effort interpreter cleanup
        with suppress(Exception):
            self.close()


def verify_snapshot(strategy: str, corpus_tag: str, expected_source_ids: list[str], corpus_manifest: dict | None) -> dict:
    path = snapshot_metadata_path(strategy, corpus_tag)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{strategy} snapshot metadata is unreadable: {path}") from exc
    if metadata.get("status") != "complete" or metadata.get("strategy") != strategy:
        raise RuntimeError(f"{strategy} snapshot is not complete")
    if metadata.get("official_revision") != OFFICIAL_REVISIONS[strategy]:
        raise RuntimeError(f"{strategy} snapshot uses a different official revision")
    expected = sorted(expected_source_ids)
    if metadata.get("source_count") != len(expected) or metadata.get("source_set_sha256") != source_set_sha256(expected):
        raise RuntimeError(f"{strategy} snapshot source set does not match the prepared corpus")
    try:
        records = json.loads((corpus_output_dir(strategy, corpus_tag) / "input" / "corpus.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{strategy} staged corpus is unreadable") from exc
    if not isinstance(records, list) or sorted(str(row.get("source_id", "")) for row in records) != expected:
        raise RuntimeError(f"{strategy} staged corpus identities do not match the prepared corpus")
    if metadata.get("corpus_records_sha256") != corpus_records_sha256(records):
        raise RuntimeError(f"{strategy} staged corpus content does not match its snapshot")
    if corpus_manifest is not None and metadata.get("corpus_manifest_fingerprint") != corpus_manifest.get("fingerprint"):
        raise RuntimeError(f"{strategy} snapshot fingerprint does not match the corpus manifest")
    return metadata
