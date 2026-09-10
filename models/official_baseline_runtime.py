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
from contextlib import suppress
from pathlib import Path
from typing import Any

from core.benchmark_failures import BenchmarkIntegrityError
from core.embedding_policy import EmbeddingOperationalConfig
from core.strategy_registry import BY_NAME, EXTERNAL_STRATEGIES, get_strategy
from utils.io import _write_json

OFFICIAL_REVISIONS = {name: BY_NAME[name].revision for name in EXTERNAL_STRATEGIES}
OFFICIAL_REPOSITORIES = {name: BY_NAME[name].repository for name in EXTERNAL_STRATEGIES}
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


def official_root(strategy: str) -> Path:
    key = f"RAG_{strategy.upper()}_ROOT"
    home = Path(os.environ.get("RAG_OFFICIAL_BASELINE_HOME", "data/official_baselines")).expanduser()
    return Path(os.environ.get(key, str(home / strategy / "source"))).expanduser().resolve()


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
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--ignored"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"{strategy} official checkout is not clean; reinstall the pinned runtime")


def _command(strategy: str, corpus_tag: str, mode: str) -> list[str]:
    worker = get_strategy(strategy).worker
    if not worker:
        raise ValueError(f"{strategy} has no isolated official worker")
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
    embedding = EmbeddingOperationalConfig.resolve(strategy)
    env["RAG_EMBEDDING_BATCH_SIZE"] = str(embedding.batch_size)
    env["RAG_EMBEDDING_CONCURRENCY"] = str(embedding.concurrency)
    env["RAG_EMBEDDING_RETRY_ATTEMPTS"] = str(embedding.retry_attempts)
    env["LLM_MAX_RETRIES"] = str(embedding.retry_attempts)
    from core.inference_transport import InferenceTransport

    transport = InferenceTransport.resolve(strategy)
    if get_strategy(strategy).transport_profile != "openai_compatible_litellm":
        raise RuntimeError(f"{strategy} paper runtime has an unsupported inference transport profile")
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
            "RAG_INFERENCE_BASE_URL": os.environ["RAG_INFERENCE_BASE_URL"],
            "RAG_INFERENCE_API_KEY": os.environ["RAG_INFERENCE_API_KEY"],
            "RAG_GENERATION_MODEL": transport.generation_model,
            "RAG_EMBEDDING_MODEL": transport.embedding_model,
            "OPENAI_API_KEY": transport.api_key,
            "RAG_INFERENCE_TIMEOUT": str(transport.timeout_seconds or 0),
            "RAG_INFERENCE_RETRY_ATTEMPTS": str(transport.retry_attempts),
            "RAG_GENERATION_CONCURRENCY": str(transport.generation_concurrency),
            "RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS": str(transport.embedding_concurrency),
            "RAG_LLM_SEED": "" if transport.generation_seed is None else str(transport.generation_seed),
            "EMBEDDING_QUERY_INSTRUCTION": transport.embedding_query_instruction,
            "MAX_EMBEDDING_LENGTH": str(transport.embedding_max_input_tokens),
            "NEO4J_VECTOR_DIMENSIONS": str(transport.embedding_dimensions),
            "RAG_EMBEDDING_TOKEN_RESERVE": str(transport.embedding_token_reserve),
            "RAG_MAX_CONTEXT_LENGTH": str(transport.generation_max_context_tokens),
        }
    )
    if get_strategy(strategy).primary:
        # Upstream dotenv imports may repopulate absent aliases. Empty exports
        # block that without carrying a second endpoint or credential contract.
        from core.inference_transport import preserve_provider_environment
        preserve_provider_environment(env)
    else:
        # Legacy workers still consume these private child aliases. Primary
        # research drivers resolve the canonical contract again in the child.
        env.update({
            "VLLM_URL": transport.generation_base_url,
            "VLLM_EMBED_URL": transport.embedding_base_url,
            "VLLM_API_BASE": transport.generation_base_url,
            "VLLM_EMBED_API_BASE": transport.embedding_base_url,
            "VLLM_API_KEY": transport.api_key,
            "VLLM_SERVED_MODEL_NAME": transport.generation_model,
            "VLLM_SERVED_EMBED_MODEL_NAME": transport.embedding_model,
            "OPENAI_BASE_URL": transport.generation_base_url,
        })
    return env


def run_index_worker(strategy: str, corpus_tag: str, request: dict[str, Any]) -> dict[str, Any]:
    validate_runtime(strategy)
    runtime_lock = _acquire_runtime_lock(strategy)
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
    except Exception:
        runtime_lock.close()
        raise
    if process.stdin is None or process.stdout is None:
        process.terminate()
        runtime_lock.close()
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
        runtime_lock.close()
        raise RuntimeError(f"{strategy} official index worker failed: {detail}")
    runtime_lock.close()
    return payload


class OfficialQueryWorker:
    """One persistent official process per adapter to avoid reloading models."""

    def __init__(self, strategy: str, corpus_tag: str):
        validate_runtime(strategy)
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
            for line in self._process.stdout:
                self._stdout_queue.put(line)
            self._stdout_queue.put(None)

        self._reader = threading.Thread(target=_read_stdout, daemon=True)
        self._reader.start()
        ready = self.request({"operation": "ready"})
        if not ready.get("ready"):
            raise RuntimeError(f"{strategy} official query worker did not become ready")

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        queued_at = time.perf_counter()
        with self._lock:
            worker_queue_seconds = time.perf_counter() - queued_at
            if self._process.poll() is not None or self._process.stdin is None or self._process.stdout is None:
                raise BenchmarkIntegrityError(f"{self.strategy} official query worker is not running")
            self._process.stdin.write(json.dumps(payload) + "\n")
            self._process.stdin.flush()
            timeout = float(os.environ.get("RAG_OFFICIAL_QUERY_TIMEOUT", "1800"))
            deadline = time.monotonic() + timeout
            while True:
                try:
                    line = self._stdout_queue.get(timeout=max(0.01, deadline - time.monotonic()))
                except queue.Empty as exc:
                    self._process.terminate()
                    raise BenchmarkIntegrityError(f"{self.strategy} official query exceeded {timeout:g} seconds and its worker was terminated") from exc
                if line is None:
                    raise BenchmarkIntegrityError(f"{self.strategy} official query worker exited without a response")
                if not line.startswith(_RESULT_PREFIX):
                    continue
                response = json.loads(line[len(_RESULT_PREFIX) :])
                if not response.get("ok"):
                    if response.get("failure_scope") == "target":
                        raise BenchmarkIntegrityError(response.get("error"))
                    raise RuntimeError(f"{self.strategy} official query failed: {response.get('error')}")
                response["worker_queue_seconds"] = worker_queue_seconds
                return response

    def close(self) -> None:
        if self._process.poll() is None and self._process.stdin is not None:
            try:
                self._process.stdin.write(json.dumps({"operation": "shutdown"}) + "\n")
                self._process.stdin.flush()
                self._process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self._process.terminate()
        if getattr(self, "_runtime_lock", None) is not None:
            self._runtime_lock.close()
            self._runtime_lock = None

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
    semantic = metadata.get("semantic_config")
    research_driver = bool(get_strategy(strategy).driver)
    expected_embedding_model = semantic.get("embedding_model") if research_driver and isinstance(semantic, dict) else configured_embedding_model()
    expected_embedding_revision = semantic.get("embedding_revision") if research_driver and isinstance(semantic, dict) else configured_embedding_revision()
    if metadata.get("embedding_model") != expected_embedding_model:
        raise RuntimeError(f"{strategy} snapshot uses a different embedding model")
    if metadata.get("embedding_revision") != expected_embedding_revision:
        raise RuntimeError(f"{strategy} snapshot uses a different embedding revision")
    if isinstance(semantic, dict) and semantic.get("semantic_config_id") is not None and (
        metadata.get("semantic_config_id") != semantic["semantic_config_id"]
    ):
        raise RuntimeError(f"{strategy} snapshot semantic profile identifier differs from its policy")
    expected_semantic_hash = hashlib.sha256(
        json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest() if isinstance(semantic, dict) else None
    if get_strategy(strategy).driver and (
        not metadata.get("semantic_config_id") or metadata.get("semantic_config_sha256") != expected_semantic_hash
    ):
        raise RuntimeError(f"{strategy} snapshot semantic config fingerprint is missing or corrupt")
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
    inventory = artifact_inventory(corpus_output_dir(strategy, corpus_tag))
    if research_driver and (inventory["file_count"] < 1 or metadata.get("artifact_inventory") != inventory):
        raise RuntimeError(f"{strategy} retrieval artifact inventory is missing or corrupt")
    if not research_driver and metadata.get("artifact_inventory") is not None and metadata["artifact_inventory"] != inventory:
        raise RuntimeError(f"{strategy} retrieval artifact inventory is corrupt")
    if corpus_manifest is not None and metadata.get("corpus_manifest_fingerprint") != corpus_manifest.get("fingerprint"):
        raise RuntimeError(f"{strategy} snapshot fingerprint does not match the corpus manifest")
    return metadata
