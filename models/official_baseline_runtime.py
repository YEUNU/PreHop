"""Process boundary and snapshot helpers for externally maintained baselines."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from core.benchmark_failures import BenchmarkIntegrityError
from core.strategy_registry import BY_NAME, EXTERNAL_STRATEGIES, get_strategy
from utils.io import _write_json

OFFICIAL_REVISIONS = {name: BY_NAME[name].revision for name in EXTERNAL_STRATEGIES}
_RESULT_PREFIX = "__PREHOP_OFFICIAL_RESULT__="
_ROOT = Path(__file__).resolve().parents[1]


def source_set_sha256(source_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(source_ids)).encode("utf-8")).hexdigest()


def corpus_records_sha256(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact_inventory(root: Path) -> dict[str, Any]:
    """Digest actual method artifacts, excluding adapter-owned staged input."""
    artifacts = root / "artifacts"
    rows: list[dict[str, Any]] = []
    if artifacts.is_dir():
        for path in sorted(candidate for candidate in artifacts.rglob("*") if candidate.is_file()):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            rows.append({"path": path.relative_to(artifacts).as_posix(), "size": path.stat().st_size, "sha256": digest})
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def configured_embedding_model() -> str:
    return os.environ.get("RAG_EMBEDDING_MODEL", "embedding-model")


def configured_embedding_revision() -> str | None:
    return os.environ.get("RAG_EMBEDDING_REVISION", "").strip() or None


def official_root(strategy: str, environment: Mapping[str, str] | None = None) -> Path:
    environment = os.environ if environment is None else environment
    key = f"RAG_{strategy.upper()}_ROOT"
    home = Path(environment.get("RAG_OFFICIAL_BASELINE_HOME", "data/official_baselines")).expanduser()
    return Path(environment.get(key, str(home / strategy / "source"))).expanduser().resolve()


def official_python(strategy: str) -> Path:
    key = f"RAG_{strategy.upper()}_PYTHON"
    home = Path(os.environ.get("RAG_OFFICIAL_BASELINE_HOME", "data/official_baselines")).expanduser()
    path = Path(os.environ.get(key, str(home / strategy / "venv/bin/python"))).expanduser()
    # Do not call resolve(): venv Python executables are commonly symlinks to
    # the base interpreter. Resolving that final link bypasses site-packages.
    return path if path.is_absolute() else Path.cwd() / path


def output_root(strategy: str) -> Path:
    spec = get_strategy(strategy)
    key = spec.output_env or f"RAG_{strategy.upper()}_OUTPUT_ROOT"
    default = spec.output_default or f"data/{strategy}_output"
    return Path(os.environ.get(key, default)).resolve()


def _acquire_runtime_lock(strategy: str):
    runtime_dir = official_root(strategy).parent
    runtime_dir.mkdir(parents=True, exist_ok=True)
    handle = (runtime_dir.parent / ".runtime.lock").open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
    return handle


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
    return title, body


def stage_corpus(strategy: str, dataset_path: str | Path, corpus_tag: str) -> tuple[list[dict[str, Any]], Path]:
    target = corpus_output_dir(strategy, corpus_tag)
    input_dir = target / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for path in sorted(Path(dataset_path).iterdir()):
        if path.suffix not in {".txt", ".md"}:
            continue
        title, body = _parse_staged_document(path)
        records.append({"source_id": path.stem, "title": title, "text": body})
    _write_json(input_dir / "corpus.json", records)
    return records, target


def _command(strategy: str, corpus_tag: str, mode: str) -> list[str]:
    worker = get_strategy(strategy).worker
    return [
        str(official_python(strategy)),
        str(_ROOT / "scripts" / worker),
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
    from core.paper_policy import preserve_method_environment

    preserve_method_environment(env)
    runtime_bin = str(official_python(strategy).parent)
    env["PATH"] = runtime_bin + os.pathsep + env.get("PATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    seed = os.environ.get("RAG_LLM_SEED", "").strip()
    env["PYTHONHASHSEED"] = seed or os.environ.get("PYTHONHASHSEED", "0")
    if seed:
        env["RAG_LLM_SEED"] = seed
        env["RAG_SEED"] = seed
    from core.inference_transport import InferenceTransport

    transport = InferenceTransport.resolve(strategy)
    # Ambient provider state is never inherited across the process boundary.
    # The aliases below are derived only from the validated canonical contract
    # and exist solely for pinned upstream client constructors.
    for name in tuple(env):
        if name.startswith(("AZURE_OPENAI", "OPENAI_", "VLLM_", "LLM_")) or name in {
            "API_VERSION",
            "OPENAI_PROVIDER",
        }:
            env.pop(name, None)
    env.update(
        {
            "RAG_INFERENCE_BASE_URL": transport.generation_base_url,
            "RAG_INFERENCE_API_KEY": transport.api_key,
            "RAG_GENERATION_MODEL": transport.generation_model,
            "RAG_EMBEDDING_MODEL": transport.embedding_model,
            "OPENAI_API_KEY": transport.api_key,
            "RAG_INFERENCE_TIMEOUT": str(transport.timeout_seconds or 0),
            "RAG_INFERENCE_RETRY_ATTEMPTS": str(transport.retry_attempts),
            "RAG_GENERATION_CONCURRENCY": str(transport.generation_concurrency),
            "RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS": str(transport.embedding_concurrency),
            "RAG_EMBEDDING_BATCH_SIZE": str(transport.embedding_batch_size),
            "RAG_EMBEDDING_CONCURRENCY": str(transport.embedding_concurrency),
            "RAG_EMBEDDING_RETRY_ATTEMPTS": str(transport.retry_attempts),
            "RAG_LLM_SEED": "" if transport.generation_seed is None else str(transport.generation_seed),
            "EMBEDDING_QUERY_INSTRUCTION": transport.embedding_query_instruction,
            "MAX_EMBEDDING_LENGTH": str(transport.embedding_max_input_tokens),
            "NEO4J_VECTOR_DIMENSIONS": str(transport.embedding_dimensions),
            "RAG_EMBEDDING_TOKEN_RESERVE": str(transport.embedding_token_reserve),
            "RAG_MAX_CONTEXT_LENGTH": str(transport.generation_max_context_tokens),
        }
    )
    # Empty aliases block upstream dotenv from restoring a second contract.
    from core.inference_transport import preserve_provider_environment
    preserve_provider_environment(env)
    return env


def _terminate_owned_process(process: subprocess.Popen) -> None:
    """Reap a worker created here, escalating only if graceful termination stalls."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_index_worker(strategy: str, corpus_tag: str, request: dict[str, Any]) -> dict[str, Any]:
    runtime_lock = _acquire_runtime_lock(strategy)
    process = None
    try:
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
            detail = payload.get("error") if isinstance(payload, dict) else None
            detail = detail or "".join(output_tail)[-4000:] or f"exit status {returncode}"
            raise RuntimeError(f"{strategy} official index worker failed: {detail}")
        return payload
    finally:
        try:
            if process is not None:
                _terminate_owned_process(process)
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        with suppress(OSError):
                            stream.close()
        finally:
            runtime_lock.close()


class OfficialQueryWorker:
    """One persistent official process per adapter to avoid reloading models."""

    def __init__(self, strategy: str, corpus_tag: str):
        self._runtime_lock = _acquire_runtime_lock(strategy)
        self.strategy = strategy
        self._lock = threading.Lock()
        try:
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
        except Exception:
            self._runtime_lock.close()
            self._runtime_lock = None
            raise
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()

        def _read_stdout() -> None:
            assert self._process.stdout is not None
            try:
                for line in self._process.stdout:
                    self._stdout_queue.put(line)
            finally:
                # The reader owns stdout; closing it from another thread can
                # block while a read is in progress.
                with suppress(OSError):
                    self._process.stdout.close()
                self._stdout_queue.put(None)

        self._reader = threading.Thread(target=_read_stdout, daemon=True)
        self._reader.start()
        try:
            self.request({"operation": "ready"})
        except BaseException:
            with suppress(Exception):
                self.close()
            raise

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        queued_at = time.perf_counter()
        with self._lock:
            worker_queue_seconds = time.perf_counter() - queued_at
            try:
                self._process.stdin.write(json.dumps(payload) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise BenchmarkIntegrityError(f"{self.strategy} official query worker input is closed") from exc
            timeout = float(os.environ.get("RAG_OFFICIAL_QUERY_TIMEOUT", "1800"))
            deadline = time.monotonic() + timeout
            while True:
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    line = self._stdout_queue.get(timeout=remaining)
                except queue.Empty as exc:
                    self._process.terminate()
                    raise BenchmarkIntegrityError(f"{self.strategy} official query exceeded {timeout:g} seconds and its worker was terminated") from exc
                if line is None:
                    raise BenchmarkIntegrityError(
                        f"{self.strategy} official query worker closed stdout (exit status {self._process.poll()})"
                    )
                if not line.startswith(_RESULT_PREFIX):
                    continue
                response = json.loads(line[len(_RESULT_PREFIX) :])
                if not response.get("ok"):
                    raise RuntimeError(f"{self.strategy} official query failed: {response.get('error')}")
                response["worker_queue_seconds"] = worker_queue_seconds
                return response

    def close(self) -> None:
        process = getattr(self, '_process', None)
        try:
            if process is not None:
                if process.poll() is None:
                    try:
                        process.stdin.write(json.dumps({"operation": "shutdown"}) + "\n")
                        process.stdin.flush()
                        process.wait(timeout=10)
                    except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                        _terminate_owned_process(process)
                if process.stdin is not None:
                    with suppress(OSError):
                        process.stdin.close()
                reader = getattr(self, '_reader', None)
                if reader is not None:
                    reader.join(timeout=5)
                elif process.stdout is not None:
                    process.stdout.close()
        finally:
            if getattr(self, "_runtime_lock", None) is not None:
                self._runtime_lock.close()
                self._runtime_lock = None

    def __del__(self):  # pragma: no cover - best-effort interpreter cleanup
        with suppress(Exception):
            self.close()
