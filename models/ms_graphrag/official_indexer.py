"""Official MS GraphRAG indexing wired to external OpenAI-compatible endpoints.

Builds a GraphRagConfig that points LiteLLM at VLLM_URL/VLLM_EMBED_URL (the
LiteLLM proxy by default — see CLAUDE.md "Model / inference infra"; override
with RAG_MS_GEN_API_BASE(S)/RAG_MS_EMBED_API_BASE for a different endpoint)
and runs the standard pipeline (extract_graph → Leiden communities →
community reports → embeddings).

Outputs parquet under data/ms_graphrag_output/<corpus_tag>/. The query-time
adapter reads these parquet files instead of expecting Neo4j Community nodes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

logger = logging.getLogger("Prehop")


# Defaults to VLLM_URL (the LiteLLM proxy by default). RAG_MS_GEN_API_BASES
# can still list multiple comma-separated external endpoints for
# round-robin — the primary base (passed as
# ModelConfig.api_base) is the first entry, but the LiteLLM Router monkey-
# patch below intercepts and shuffles across all bases per call.
_GEN_API_BASES = [
    s.strip()
    for s in os.environ.get(
        "RAG_MS_GEN_API_BASES",
        os.environ.get("VLLM_URL", ""),
    ).split(",")
    if s.strip()
]
_GEN_API_BASE = _GEN_API_BASES[0] if _GEN_API_BASES else ""
_GEN_MODEL_NAME = os.environ.get("VLLM_SERVED_MODEL_NAME", "generation-model")
_EMBED_API_BASE = os.environ.get("RAG_MS_EMBED_API_BASE", os.environ.get("VLLM_EMBED_URL", ""))
_EMBED_MODEL_NAME = os.environ.get(
    "RAG_MS_EMBED_MODEL_NAME", os.environ.get("VLLM_SERVED_EMBED_MODEL_NAME", "embedding-model")
)
_GEN_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
# Must match the configured embedding model's real output dimension or
# LanceDB rejects the embedding parquet on a FixedSizeList shape mismatch —
# same constraint as NEO4J_VECTOR_DIMENSIONS for prehop/naive/hoprag's Neo4j
# vector indexes (see CLAUDE.md "Model / inference infra"), reused here since
# it's the same one embedding model across every strategy.
_EMBED_DIM = int(os.environ.get("RAG_MS_EMBED_DIM", os.environ.get("NEO4J_VECTOR_DIMENSIONS", "1024")))

# Where parquet artifacts land. corpus_tag-scoped so different runs don't clobber.
_OUTPUT_ROOT = Path(os.environ.get("RAG_MS_OUTPUT_ROOT", "data/ms_graphrag_output"))
SNAPSHOT_METADATA_FILENAME = "index_snapshot_metadata.json"
_SNAPSHOT_VERSION = 1


def output_dir_for(corpus_tag: str) -> Path:
    return (_OUTPUT_ROOT / corpus_tag).resolve()


def cache_dir_for(corpus_tag: str) -> Path:
    return (_OUTPUT_ROOT / corpus_tag / "_cache").resolve()


def input_dir_for(corpus_tag: str) -> Path:
    return (_OUTPUT_ROOT / corpus_tag / "_input").resolve()


def snapshot_metadata_path(corpus_tag: str) -> Path:
    return output_dir_for(corpus_tag) / SNAPSHOT_METADATA_FILENAME


def _source_set_sha256(source_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(source_ids)).encode("utf-8")).hexdigest()


def _expected_source_ids(staged_input: Path, corpus_manifest: dict | None) -> list[str]:
    source_ids = sorted(path.stem for path in staged_input.iterdir() if path.suffix in (".txt", ".md"))
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("MS GraphRAG staged corpus has no files or duplicate filename stems")
    if corpus_manifest is not None and corpus_manifest.get("paragraph_count") != len(source_ids):
        raise ValueError("MS GraphRAG corpus manifest paragraph_count does not match staged file count")
    return source_ids


def _write_snapshot_metadata(corpus_tag: str, payload: dict) -> None:
    """Atomically publish state only after the on-disk snapshot is known."""
    path = snapshot_metadata_path(corpus_tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _set_snapshot_in_progress(corpus_tag: str, corpus_manifest: dict | None) -> None:
    _write_snapshot_metadata(
        corpus_tag,
        {
            "strategy": "ms_graphrag",
            "corpus_tag": corpus_tag,
            "status": "in_progress",
            "snapshot_version": _SNAPSHOT_VERSION,
            "corpus_manifest_fingerprint": (corpus_manifest or {}).get("fingerprint"),
            "corpus_manifest_paragraph_count": (corpus_manifest or {}).get("paragraph_count"),
            "updated_at_epoch": time.time(),
        },
    )


def _verify_and_publish_snapshot(
    corpus_tag: str,
    source_ids: list[str],
    corpus_manifest: dict | None,
) -> dict:
    """Compare actual ``documents.parquet`` sources to the staged corpus."""
    import pandas as pd

    documents_path = output_dir_for(corpus_tag) / "documents.parquet"
    documents = pd.read_parquet(documents_path)
    if "title" not in documents.columns:
        raise RuntimeError(f"MS GraphRAG documents artifact lacks title column: {documents_path}")
    actual_ids = sorted(
        {
            Path(str(title)).stem
            for title in documents["title"].tolist()
            if isinstance(title, str) and title.strip()
        }
    )
    expected = sorted(source_ids)
    if actual_ids != expected:
        missing = sorted(set(expected) - set(actual_ids))
        unexpected = sorted(set(actual_ids) - set(expected))
        raise RuntimeError(
            "MS GraphRAG documents.parquet source snapshot does not match staged corpus: "
            f"expected={len(expected)} actual={len(actual_ids)} "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    source_digest = _source_set_sha256(actual_ids)
    payload = {
        "strategy": "ms_graphrag",
        "corpus_tag": corpus_tag,
        "status": "complete",
        "snapshot_version": _SNAPSHOT_VERSION,
        "corpus_manifest_fingerprint": (corpus_manifest or {}).get("fingerprint"),
        "corpus_manifest_paragraph_count": (corpus_manifest or {}).get("paragraph_count"),
        "source_count": len(actual_ids),
        "source_set_sha256": source_digest,
        "completed_at_epoch": time.time(),
    }
    _write_snapshot_metadata(corpus_tag, payload)
    return payload


class _WorkflowTiming:
    """Record official GraphRAG workflow durations without changing upstream code."""

    def __init__(self) -> None:
        self._started: dict[str, float] = {}
        self.timing: dict[str, float] = {}

    def pipeline_start(self, names: list[str]) -> None:
        del names

    def pipeline_end(self, results) -> None:
        del results

    def workflow_start(self, name: str, instance: object) -> None:
        del instance
        self._started[name] = time.perf_counter()

    def workflow_end(self, name: str, instance: object) -> None:
        del instance
        started = self._started.pop(name, None)
        if started is not None:
            self.timing[f"workflow_{name}_seconds"] = time.perf_counter() - started

    def progress(self, progress) -> None:
        del progress

    def pipeline_error(self, error: BaseException) -> None:
        del error


def _stage_input_files(
    dataset_path: str,
    corpus_tag: str,
) -> Path:
    """Copy/link every corpus file into a tag-scoped input dir.

    MS pipeline reads from one directory via input_storage.base_dir. We can't
    pass a file list, so we materialize a filtered staging dir under the
    output tree (hardlinks to avoid disk waste; falls back to copy on FS that
    rejects hardlinks).
    """
    src_root = Path(dataset_path)
    if not src_root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset_path}")

    files = sorted(p for p in src_root.iterdir() if p.suffix in (".txt", ".md"))

    if not files:
        raise ValueError(f"MS GraphRAG staging selected no .txt/.md files from {dataset_path}")

    staged = input_dir_for(corpus_tag)
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)

    for fp in files:
        dest = staged / fp.name
        try:
            os.link(fp, dest)
        except OSError:
            shutil.copy2(fp, dest)

    logger.info("MS staging: %d files materialized at %s", len(files), staged)
    return staged


def _register_external_models_with_litellm() -> None:
    """LiteLLM rejects response_format/JSON-schema requests for unknown models.

    vLLM with Qwen3 actually supports structured output via guided_json, so
    we register the externally served model names with supports_response_schema=True.
    Without this, create_community_reports raises 'Model does not support
    response schemas' on every Leiden cluster.
    """
    import litellm

    base_meta = {
        "max_tokens": int(os.environ.get("RAG_MAX_CONTEXT_LENGTH", "16384")),
        "max_input_tokens": int(os.environ.get("RAG_MAX_CONTEXT_LENGTH", "16384")),
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "openai",
        "supports_response_schema": True,
    }
    litellm.register_model(
        {
            f"openai/{_GEN_MODEL_NAME}": {**base_meta, "mode": "chat"},
            f"openai/{_EMBED_MODEL_NAME}": {
                **base_meta,
                "mode": "embedding",
                "max_input_tokens": int(os.environ.get("MAX_EMBEDDING_LENGTH", "16384")),
                "output_vector_size": _EMBED_DIM,
            },
        }
    )


_ROUTER_INSTALLED = False


def _install_litellm_router_for_gen() -> None:
    """Monkey-patch litellm.acompletion to round-robin gen-chat across
    multiple endpoints when RAG_MS_GEN_API_BASES lists more than one (a no-op
    with the default single LiteLLM-proxy endpoint). graphrag-llm calls bare
    `litellm.acompletion(**args)`; we intercept only when model matches our
    configured generation model and delegate to a Router with simple-shuffle. Embedding +
    any other model passes through unchanged.
    """
    global _ROUTER_INSTALLED
    if _ROUTER_INSTALLED:
        return
    import contextvars
    import json
    import urllib.error
    import urllib.request

    import litellm
    from litellm import Router

    # Validate the OpenAI-compatible model registry rather than a proxy-specific
    # /health route. Busy external servers can legitimately queue a health
    # response near max_num_seqs, so retry with a realistic timeout.
    live_bases = []
    for base in _GEN_API_BASES:
        models_url = base.rstrip("/") + "/models"
        for attempt in range(1, 4):
            try:
                request = urllib.request.Request(
                    models_url,
                    headers={"Authorization": f"Bearer {_GEN_API_KEY}"},
                )
                with urllib.request.urlopen(request, timeout=15) as response:
                    payload = json.load(response)
                model_ids = {item.get("id") for item in payload.get("data", [])}
                if _GEN_MODEL_NAME not in model_ids:
                    raise RuntimeError(
                        f"configured model {_GEN_MODEL_NAME!r} not advertised by {models_url}: "
                        f"{sorted(model_id for model_id in model_ids if model_id)}"
                    )
                live_bases.append(base)
                break
            except (OSError, urllib.error.URLError, ValueError, RuntimeError) as exc:
                if attempt == 3:
                    logger.warning("MS GraphRAG: generation endpoint rejected after retries: %s (%s)", base, exc)
                else:
                    time.sleep(attempt)
    if not live_bases:
        raise ConnectionError(
            f"MS GraphRAG: no configured generation endpoint passed its health check: {_GEN_API_BASES}"
        )
    logger.info("MS GraphRAG: live gen endpoints for router: %s", live_bases)

    if len(live_bases) <= 1:
        return

    target = f"openai/{_GEN_MODEL_NAME}"
    model_list = [
        {
            "model_name": target,
            "litellm_params": {
                "model": target,
                "api_base": ab,
                "api_key": _GEN_API_KEY,
            },
        }
        for ab in live_bases
    ]
    router = Router(model_list=model_list, routing_strategy="simple-shuffle")

    # Capture the ORIGINAL acompletion before we replace litellm.acompletion.
    # Router internally calls `litellm.acompletion(...)`, which would re-enter
    # our wrapper and recurse forever. We use a contextvar to flag "we are
    # already inside Router" so the re-entry bypasses Router and uses the
    # original function — which is what Router actually expects to call.
    orig_acompletion = litellm.acompletion
    _in_router: contextvars.ContextVar[bool] = contextvars.ContextVar(
        "_ms_router_reentry",
        default=False,
    )

    async def _routed_acompletion(**kwargs):
        if _in_router.get():
            return await orig_acompletion(**kwargs)
        if kwargs.get("model") == target:
            kwargs.pop("api_base", None)
            token = _in_router.set(True)
            try:
                return await router.acompletion(**kwargs)
            finally:
                _in_router.reset(token)
        return await orig_acompletion(**kwargs)

    litellm.acompletion = _routed_acompletion
    _ROUTER_INSTALLED = True
    logger.info(
        "MS LiteLLM router installed for %s across %d endpoints: %s",
        target,
        len(_GEN_API_BASES),
        _GEN_API_BASES,
    )


def build_config(corpus_tag: str, staged_input_dir: Path):
    """Construct a GraphRagConfig pointing LiteLLM at external inference."""
    _register_external_models_with_litellm()
    _install_litellm_router_for_gen()

    from graphrag.config.models.graph_rag_config import GraphRagConfig
    from graphrag_cache import CacheConfig
    from graphrag_input import InputConfig
    from graphrag_llm.config.model_config import ModelConfig
    from graphrag_storage import StorageConfig, StorageType
    from graphrag_vectors import IndexSchema, VectorStoreConfig

    out_dir = output_dir_for(corpus_tag)
    cache_dir = cache_dir_for(corpus_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # vLLM 4B quirks:
    # - encoding_format="float" required (LiteLLM 1.83 sends None which vLLM 0.15 rejects).
    # - max_tokens: keep modest so a runaway entity-extraction doesn't blow chunk context.
    # - extra_body.guided_json supported by vLLM but not configured here; rely on json_repair fallback.
    completion_call_args = {
        "temperature": 0.0,
        "max_tokens": 1500,
    }
    # Community reports need a higher output cap. The default 1500 truncates the
    # report JSON mid-string (Unterminated string -> JSONDecodeError -> empty
    # report; ~14% of communities failed at 1500 on the 4B model). max_length is
    # 2000 content tokens + JSON wrapping, so give it headroom. A SEPARATE model
    # id keeps extract_graph on max_tokens=1500 so its (expensive) per-call LLM
    # cache stays valid — the cache key includes max_tokens, so bumping the
    # shared model would force a full re-extraction.
    report_call_args = {
        "temperature": 0.0,
        "max_tokens": int(os.environ.get("RAG_MS_REPORT_MAX_TOKENS", "4096")),
    }

    cfg = GraphRagConfig(
        completion_models={
            "default_completion_model": ModelConfig(
                type="litellm",
                model_provider="openai",
                model=_GEN_MODEL_NAME,
                api_base=_GEN_API_BASE,
                api_key=_GEN_API_KEY,
                call_args=completion_call_args,
            ),
            "report_completion_model": ModelConfig(
                type="litellm",
                model_provider="openai",
                model=_GEN_MODEL_NAME,
                api_base=_GEN_API_BASE,
                api_key=_GEN_API_KEY,
                call_args=report_call_args,
            ),
        },
        embedding_models={
            "default_embedding_model": ModelConfig(
                type="litellm",
                model_provider="openai",
                model=_EMBED_MODEL_NAME,
                api_base=_EMBED_API_BASE,
                api_key=_GEN_API_KEY,
                call_args={"encoding_format": "float"},
            ),
        },
        input=InputConfig(file_pattern=r".*\.txt$"),
        input_storage=StorageConfig(
            type=StorageType.File,
            base_dir=str(staged_input_dir),
        ),
        output_storage=StorageConfig(
            type=StorageType.File,
            base_dir=str(out_dir),
        ),
        cache=CacheConfig(
            storage=StorageConfig(type=StorageType.File, base_dir=str(cache_dir)),
        ),
        # Must match the configured embedding model's real dim (_EMBED_DIM
        # above); default IndexSchema assumes 3072 (text-embedding-3-large).
        # Without the override, lancedb rejects the embedding parquet on a
        # FixedSizeList shape mismatch. Keys MUST match
        # generate_text_embeddings.py's embedded_fields: entity_description /
        # community_full_content / text_unit_text (graphrag.config.embeddings
        # constants), not arbitrary names.
        vector_store=VectorStoreConfig(
            db_uri=str(out_dir / "lancedb"),
            index_schema={
                name: IndexSchema(index_name=name, vector_size=_EMBED_DIM)
                for name in (
                    "entity_description",
                    "community_full_content",
                    "text_unit_text",
                )
            },
        ),
    )
    # Route community-report generation to the higher-max_tokens model so long
    # reports don't truncate; extract_graph stays on the cached default model.
    cfg.community_reports.completion_model_id = "report_completion_model"

    # MS pipeline gates extract_graph + summarize via asyncio.Semaphore(num_threads=concurrent_requests).
    # vLLM 4B handles 30+ parallel reqs comfortably (peak observed: 14 running + 7 waiting at limit 16
    # → fully saturated). Bump to 48 to drive the queue and shave wall-clock on the 33k-text_unit corpus.
    cfg.concurrent_requests = max(1, int(os.environ.get("RAG_MS_CONCURRENT_REQUESTS", "48")))
    return cfg


async def run_official_index(
    dataset_path: str,
    corpus_tag: str,
    corpus_manifest: dict | None = None,
) -> dict[str, float]:
    """Stage inputs, build config, run the standard MS pipeline."""
    from graphrag.api.index import build_index
    from graphrag.config.enums import IndexingMethod

    staged_input = _stage_input_files(dataset_path, corpus_tag)
    source_ids = _expected_source_ids(staged_input, corpus_manifest)
    # Make an old complete marker unusable before the official pipeline starts
    # replacing output artifacts. A failed/interrupted run therefore fails the
    # later benchmark gate rather than inheriting stale provenance.
    _set_snapshot_in_progress(corpus_tag, corpus_manifest)

    config = build_config(corpus_tag, staged_input)
    out_dir = output_dir_for(corpus_tag)

    logger.info(
        "MS official indexing: corpus_tag=%s, %d input files, output=%s, gen=%s embed=%s",
        corpus_tag,
        len(list(staged_input.iterdir())),
        out_dir,
        _GEN_API_BASE,
        _EMBED_API_BASE,
    )

    workflow_timing = _WorkflowTiming()
    results = await build_index(
        config=config,
        method=IndexingMethod.Standard,
        callbacks=[workflow_timing],
        verbose=False,
    )

    failures = [r for r in results if getattr(r, "errors", None)]
    if failures:
        logger.error("MS pipeline produced %d workflow(s) with errors", len(failures))
        for r in failures:
            logger.error("  workflow=%s errors=%s", getattr(r, "workflow", "?"), r.errors)

    # Sanity: verify expected parquet artifacts.
    expected = [
        "entities.parquet",
        "relationships.parquet",
        "communities.parquet",
        "community_reports.parquet",
        "text_units.parquet",
        "documents.parquet",
    ]
    missing = [name for name in expected if not (out_dir / name).exists()]
    if failures or missing:
        parts = []
        if failures:
            parts.append(f"{len(failures)} failed workflow(s)")
        if missing:
            parts.append(f"missing artifacts: {missing}")
        raise RuntimeError("MS GraphRAG indexing incomplete: " + "; ".join(parts))
    snapshot = _verify_and_publish_snapshot(corpus_tag, source_ids, corpus_manifest)
    workflow_timing.timing["active_snapshot_verified"] = 1.0
    workflow_timing.timing["active_snapshot_source_count"] = float(snapshot["source_count"])
    logger.info("MS pipeline produced all expected parquet files at %s", out_dir)
    return workflow_timing.timing
