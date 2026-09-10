"""Official HopRAG indexing wired to external OpenAI-compatible inference.

`third_party/HopRAG` is the upstream package. It's not pip-installable: it
imports `config` and `tool` as top-level modules and bakes config-time vars
(edge_name, embed_dim, deployment_sign, cypher templates) into module
constants. We:

1. Prepend the package dir to sys.path.
2. Import `config`, override its attributes for external inference + corpus-tagged
   labels, and recompile the cypher templates that string-concatenated
   `edge_name` at module load.
3. Monkey-patch `tool.load_embed_model` / `tool.get_doc_embeds` to use the
   configured embedding endpoint (avoids loading SentenceTransformer locally).
4. Then `import HopBuilder`, which picks up the patched config.

`HopBuilder.create_edge` does pairwise question similarity, which is O(N²) per
group. We preserve the official per-problem context groups so each edge build
stays tractable without a dataset-specific entity gate.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from core.index_namespace import index_namespace

logger = logging.getLogger("Prehop")

_HOPRAG_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "HopRAG"

_GEN_API_BASE = os.environ.get("RAG_INFERENCE_BASE_URL", "").strip()
_GEN_API_BASES: list[str] = [_GEN_API_BASE] if _GEN_API_BASE else []
_GEN_MODEL_NAME = os.environ.get("RAG_GENERATION_MODEL", "")
_EMBED_API_BASE = _GEN_API_BASE
_EMBED_MODEL_NAME = os.environ.get("RAG_EMBEDDING_MODEL", "")
_EMBED_DIM = int(os.environ.get("RAG_HOP_EMBED_DIM", os.environ.get("NEO4J_VECTOR_DIMENSIONS", "2560")))
_EMBED_BATCH_SIZE = max(1, int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE", "32")))
_EMBED_REQUEST_SEMAPHORE = threading.BoundedSemaphore(
    max(1, int(os.environ.get("RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS", "2")))
)
_GEN_API_KEY = os.environ.get("RAG_INFERENCE_API_KEY", "EMPTY")
_DOC_WORKERS = max(1, int(os.environ.get("RAG_HOP_DOC_WORKERS", "10")))
_INTERNAL_RETRIES = max(1, int(os.environ.get("RAG_HOP_INTERNAL_RETRIES", "2")))
_QUESTION_RETRIES = max(1, int(os.environ.get("RAG_HOP_QUESTION_RETRIES", "3")))
_NODE_INSERT_BATCH = max(1, int(os.environ.get("RAG_HOP_NODE_BATCH", "200")))
_EDGE_INSERT_BATCH = max(1, int(os.environ.get("RAG_HOP_EDGE_BATCH", "500")))

_OUTPUT_ROOT = Path(os.environ.get("RAG_HOP_OUTPUT_ROOT", "data/hoprag_output"))
_SNAPSHOT_LABEL = "RAGIndexSnapshot"
_SNAPSHOT_VERSION = 2
_CACHE_FORMAT_VERSION = 3
_PAGE_MARKER_RE = re.compile(r"^-+\s*Page\s+\d+\s*-+$", re.IGNORECASE)


def _hoprag_indexable_text(content: str) -> str:
    """Remove repository metadata while retaining HopRAG's own chunker.

    Prepared corpora carry title, paragraph-identity, and optional page-marker
    lines for provenance. They are not evidence and must not be embedded or
    sent to HopRAG's question generator. Paragraph boundaries and all body
    text otherwise remain unchanged for the official ``\n\n`` chunker.
    """
    lines = content.removeprefix("\ufeff").splitlines()
    if lines and lines[0].startswith("Title: "):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    if lines and lines[0].startswith("Paragraph-ID: "):
        lines = lines[1:]
    lines = [line for line in lines if not _PAGE_MARKER_RE.fullmatch(line.strip())]
    body = "\n".join(lines).strip()
    return body


def _document_cache_digest(path: Path) -> str:
    """Bind a per-document cache to both source bytes and parser semantics."""
    digest = hashlib.sha256()
    digest.update(f"hoprag-cache-v{_CACHE_FORMAT_VERSION}\0".encode("ascii"))
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_pickle_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            import pickle

            pickle.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def output_dir_for(corpus_tag: str) -> Path:
    return (_OUTPUT_ROOT / corpus_tag).resolve()


def cache_dir_for(corpus_tag: str) -> Path:
    return (_OUTPUT_ROOT / corpus_tag / "_cache").resolve()


def input_dir_for(corpus_tag: str) -> Path:
    return (_OUTPUT_ROOT / corpus_tag / "_input").resolve()


def _source_set_sha256(source_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(source_ids)).encode("utf-8")).hexdigest()


def _expected_source_ids(staged_files: list[str], corpus_manifest: dict | None) -> list[str]:
    source_ids = sorted(Path(name).stem for name in staged_files)
    return source_ids


def _set_snapshot_state(config, corpus_tag: str, corpus_manifest: dict | None, status: str) -> None:
    """Keep an unconnected metadata node out of upstream HopRAG traversal."""
    from neo4j import GraphDatabase

    namespace = index_namespace(corpus_tag)
    driver = GraphDatabase.driver(
        config.neo4j_url,
        auth=(config.neo4j_user, config.neo4j_password),
        notifications_disabled_categories=["DEPRECATION"],
    )
    try:
        with driver.session(database=config.neo4j_dbname) as session:
            session.run(
                f"""
                MERGE (m:{_SNAPSHOT_LABEL} {{strategy: 'hoprag', index_namespace: $index_namespace}})
                SET m.status = $status,
                    m.corpus_tag = $corpus_tag,
                    m.snapshot_version = $_version,
                    m.corpus_manifest_fingerprint = $fingerprint,
                    m.corpus_manifest_paragraph_count = $paragraph_count,
                    m.updated_at_epoch = $updated_at
                """,
                {
                    "corpus_tag": corpus_tag,
                    "index_namespace": namespace,
                    "status": status,
                    "_version": _SNAPSHOT_VERSION,
                    "fingerprint": (corpus_manifest or {}).get("fingerprint"),
                    "paragraph_count": (corpus_manifest or {}).get("paragraph_count"),
                    "updated_at": time.time(),
                },
            ).consume()
    finally:
        driver.close()


def _publish_snapshot(builder, corpus_tag: str, source_ids: list[str], corpus_manifest: dict | None) -> dict:
    """Bind the active HopRAG representation to the complete staged corpus."""
    namespace = index_namespace(corpus_tag)
    with builder.driver.session() as session:
        rows = session.run(
            f"""
            MATCH (n:{builder.label})
            WHERE coalesce(n.source, '') <> ''
            RETURN DISTINCT n.source AS source
            """
        )
        # Stage 2 stores the filename stem in ``n.source`` already. Applying
        # Path.stem again corrupts legitimate identifiers containing periods
        # (for example ``U.S._...``) by treating the tail as another suffix.
        actual_ids = sorted({str(row["source"] or "") for row in rows})
        expected = sorted(source_ids)
        omitted = sorted(set(expected) - set(actual_ids))
        source_digest = _source_set_sha256(actual_ids)
        omitted_digest = _source_set_sha256(omitted)
        session.run(
            f"""
            MERGE (m:{_SNAPSHOT_LABEL} {{strategy: 'hoprag', index_namespace: $index_namespace}})
            SET m.status = 'complete',
                m.corpus_tag = $corpus_tag,
                m.snapshot_version = $_version,
                m.corpus_manifest_fingerprint = $fingerprint,
                m.corpus_manifest_paragraph_count = $paragraph_count,
                m.source_count = $source_count,
                m.source_set_sha256 = $source_digest,
                m.omitted_source_count = $omitted_source_count,
                m.omitted_source_set_sha256 = $omitted_source_digest,
                m.completed_at_epoch = $completed_at,
                m.updated_at_epoch = $completed_at
            """,
            {
                "corpus_tag": corpus_tag,
                "index_namespace": namespace,
                "_version": _SNAPSHOT_VERSION,
                "fingerprint": (corpus_manifest or {}).get("fingerprint"),
                "paragraph_count": (corpus_manifest or {}).get("paragraph_count"),
                "source_count": len(actual_ids),
                "source_digest": source_digest,
                "omitted_source_count": len(omitted),
                "omitted_source_digest": omitted_digest,
                "completed_at": time.time(),
            },
        ).consume()
    return {
        "status": "complete",
        "input_source_count": len(expected),
        "source_count": len(actual_ids),
        "source_set_sha256": source_digest,
        "omitted_source_count": len(omitted),
        "omitted_source_set_sha256": omitted_digest,
    }


# ---------------------------------------------------------------- file staging


def _stage_input_files(
    dataset_path: str,
    corpus_tag: str,
) -> tuple[Path, list[str]]:
    """Materialize every corpus file in a tag-scoped input directory."""
    src_root = Path(dataset_path)

    files = sorted(p for p in src_root.iterdir() if p.suffix in (".txt", ".md"))


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

    logger.info("HopRAG staging: %d files materialized at %s", len(files), staged)

    return staged, [fp.name for fp in files]


# ---------------------------------------------------------------- monkey patch


class _VLLMEmbedClient:
    """Drop-in replacement for SentenceTransformer.encode().

    Calls our vLLM /v1/embeddings (same backing model as the rest of the stack
    so HopRAG nodes/edges live in the same embedding space as prehop's,
    which keeps the architectural comparison apples-to-apples).
    """

    def __init__(self, base_url: str, model: str, dim: int):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        import requests
        from requests.adapters import HTTPAdapter

        self._sess = requests.Session()
        # Default pool_maxsize=10 overflows when DOC_WORKERS × CHUNK_THREADS > 10.
        # Set to 128 to avoid "Connection pool is full" warnings under parallel load.
        _adapter = HTTPAdapter(pool_maxsize=128, pool_connections=16)
        self._sess.mount("http://", _adapter)
        self._sess.mount("https://", _adapter)

    def encode(self, documents, normalize_embeddings: bool = True, device=None):
        _ = normalize_embeddings, device
        if documents is None:
            return np.zeros((0, self.dim), dtype=np.float32)
        if isinstance(documents, str):
            single = True
            documents = [documents]
        else:
            single = False

        if not documents:
            return np.zeros((0, self.dim), dtype=np.float32)

        # Respect the same aggregate server-capacity budget as the in-repo
        # clients. This synchronous official path can be called by several
        # document threads, so the semaphore must be thread-based.
        chunk = _EMBED_BATCH_SIZE
        out = []
        for i in range(0, len(documents), chunk):
            batch = documents[i : i + chunk]
            out.extend(self._request_batch(batch))

        arr = np.asarray(out, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / norms
        return arr[0] if single else arr

    def _request_batch(self, batch: list[str]) -> list[list[float]]:
        with _EMBED_REQUEST_SEMAPHORE:
            response = self._sess.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": batch, "encoding_format": "float"},
                headers={"Authorization": f"Bearer {_GEN_API_KEY}"},
                timeout=180,
            )
        if not response.ok:
            message = response.text or ""
            splittable = response.status_code == 413 or "messagepack data is malformed" in message.lower()
            if splittable and len(batch) > 1:
                midpoint = len(batch) // 2
                logger.warning(
                    "HopRAG embedding payload rejected (size=%d); bisecting into %d and %d",
                    len(batch),
                    midpoint,
                    len(batch) - midpoint,
                )
                return self._request_batch(batch[:midpoint]) + self._request_batch(batch[midpoint:])
            response.raise_for_status()

        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        [item.get("index") if isinstance(item, dict) else None for item in data]
        ordered = sorted(data, key=lambda item: item["index"])
        vectors: list[list[float]] = []
        for item in ordered:
            vector = item.get("embedding")
            vectors.append(vector)
        return vectors


def _setup_hoprag_modules(corpus_tag: str) -> None:
    from models.hoprag.native_runtime import setup
    setup(corpus_tag)
    # This scheduler only invokes the unchanged per-document native function.
    _patch_create_nodes_offline_parallel()


def _patch_create_nodes_offline_parallel() -> None:
    """Replace QABuilder.create_nodes_offline with a version that processes
    _DOC_WORKERS documents concurrently instead of sequentially.

    Inner per-doc chunk parallelism (max_thread_num) is preserved — the two
    levels of parallelism stack:
        total concurrent LLM calls ≈ _DOC_WORKERS × max_thread_num
        e.g. DOC_WORKERS=10 × CHUNK_THREADS=4 → at most 40 calls on the
        shared 120-sequence inference endpoint.

    Thread-safety notes:
    - Node-ID assignment uses a lock-protected counter.  IDs only need to be
      unique within the offline cache (real Neo4j IDs are assigned later in
      create_nodes_cache).
    - docid2nodes / node2questiondict are written from the main thread only
      (as_completed loop), so no lock is needed there.
    - _VLLMEmbedClient shares a requests.Session across threads, which is safe
      for concurrent POST calls (urllib3 connection pool is thread-safe).
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import as_completed as _as_completed

    import HopBuilder
    from tqdm import tqdm as _tqdm

    if getattr(HopBuilder.QABuilder.create_nodes_offline, "_patched_parallel", False):
        return

    _doc_workers = _DOC_WORKERS

    def _parallel_create_nodes_offline(self, docs_dir, start_index=0, span=100):
        import os
        import pickle
        import time as _time
        from pathlib import Path

        docs_pool = sorted(os.listdir(docs_dir))

        # Per-doc cache dir: sibling of _input/, stores one .pkl per completed doc.
        # Survives process crashes — restart resumes from last saved doc.
        per_doc_dir = Path(docs_dir).parent / "_cache" / "docs"
        per_doc_dir.mkdir(parents=True, exist_ok=True)

        # Load only cached node-ID companions to keep document embeddings and
        # question payloads out of startup memory.
        docid2nodes: dict = {}
        cached_doc_ids: set = set()

        for pkl_file in sorted(per_doc_dir.glob("*.pkl")):
            doc_id = pkl_file.stem
            source_file = Path(docs_dir) / doc_id
            digest_file = per_doc_dir / (doc_id + ".sha256")
            if not source_file.is_file() or not digest_file.is_file():
                continue
            expected_digest = _document_cache_digest(source_file)
            if digest_file.read_text(encoding="utf-8").strip() != expected_digest:
                logger.info("HopRAG cache invalidated for changed document: %s", doc_id)
                continue
            ids_file = per_doc_dir / (doc_id + ".ids")
            if ids_file.exists():
                try:
                    with open(ids_file) as fh:
                        local_nodes = json.load(fh)
                    docid2nodes[doc_id] = local_nodes
                    cached_doc_ids.add(doc_id)
                    continue
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.warning("HopRAG cache id companion is invalid (%s): %s", ids_file, exc)
            # No .ids companion yet (pkl from an older run): load pkl once,
            # write the .ids file, then discard the heavy question/embed data.
            try:
                with open(pkl_file, "rb") as fh:
                    local_nodes, _local_n2q = pickle.load(fh)
                from utils.io import _write_json

                _write_json(ids_file, local_nodes)
                docid2nodes[doc_id] = local_nodes
                cached_doc_ids.add(doc_id)
            except (OSError, EOFError, pickle.PickleError, TypeError, ValueError) as exc:
                logger.warning(
                    "HopRAG parallel: corrupt per-doc cache %s, will reprocess: %s",
                    pkl_file.name,
                    exc,
                )
                pkl_file.unlink(missing_ok=True)

        # The upstream aggregate ``docid2nodes.json`` has no parser/version
        # binding. Only a per-document cache that passed the versioned digest
        # check is safe to resume.
        docs_to_process = [d for d in docs_pool[start_index : start_index + span] if d not in cached_doc_ids]

        import config as _cfg

        logger.info(
            "HopRAG parallel node build: %d to process, %d from per-doc cache, "
            "%d in main cache | doc_workers=%d chunk_threads=%d",
            len(docs_to_process),
            len(cached_doc_ids),
            len(self.done),
            _doc_workers,
            _cfg.max_thread_num,
        )

        _id_lock = threading.Lock()
        # Start counter above any IDs already assigned in cached docs to avoid collisions.
        max_cached_id = max((max(v) for v in docid2nodes.values() if v), default=0)
        _counter = [max(start_index * 50, max_cached_id)]

        def _process_one(doc_id):
            doc_path = os.path.join(docs_dir, doc_id)
            try:
                with open(doc_path, "r") as fh:
                    doc = _hoprag_indexable_text(fh.read())
                sentence2node = self.get_single_doc_qa(doc)
                local_nodes = []
                local_n2q = {}
                for tup in sentence2node.values():
                    node = {
                        "text": tup[0],
                        "keywords": sorted(tup[1]),
                        "embed": tup[2],
                    }
                    with _id_lock:
                        _counter[0] += 1
                        node_id = _counter[0]
                    local_n2q[(node_id, doc_id)] = (node, tup[3])
                    local_nodes.append(node_id)

                # Atomic write: tmp → rename so a crash during write leaves no partial file.
                cache_file = per_doc_dir / (doc_id + ".pkl")
                _atomic_pickle_dump(cache_file, (local_nodes, local_n2q))

                # Write lightweight .ids companion (just the node-ID list) so
                # future restarts skip loading the full ~168MB pkl at startup.
                from utils.io import _write_json

                ids_file = per_doc_dir / (doc_id + ".ids")
                _write_json(ids_file, local_nodes)
                digest_file = per_doc_dir / (doc_id + ".sha256")
                digest_file.write_text(
                    _document_cache_digest(Path(doc_path)),
                    encoding="utf-8",
                )

                # Explicitly free before return: concurrent.futures caches the
                # return value inside the Future object until the executor exits.
                # Returning local_n2q (~30 MB/doc × 290 futures = ~8 GB) would
                # accumulate in memory even though the main loop discards it.
                del sentence2node, local_n2q
                return doc_id, local_nodes, None
            except Exception as exc:  # noqa: BLE001 - aggregate heterogeneous per-document failures
                import traceback as _tb

                logger.warning(
                    "HopRAG parallel: error on %s: %s\n%s",
                    doc_id,
                    exc,
                    _tb.format_exc(),
                )
                _time.sleep(1)
                return doc_id, None, None

        failed_docs: list[str] = []
        processed_docs = 0
        with ThreadPoolExecutor(max_workers=_doc_workers) as pool:
            futures = {pool.submit(_process_one, d): d for d in docs_to_process}
            for fut in _tqdm(_as_completed(futures), total=len(futures), desc="create_nodes_parallel"):
                doc_id, nodes, _n2q = fut.result()
                if nodes is not None:
                    docid2nodes[doc_id] = nodes
                    processed_docs += 1
                    # _n2q intentionally NOT accumulated — per-doc pkls hold the
                    # data; Stage 2 streams them group-by-group to avoid OOM.
                else:
                    failed_docs.append(doc_id)

        logger.info(
            "HopRAG parallel node build complete: %d docs total (%d processed + %d cached)",
            len(docid2nodes),
            processed_docs,
            len(cached_doc_ids),
        )
        if failed_docs:
            preview = ", ".join(failed_docs[:10])
            raise RuntimeError(f"HopRAG Stage 1 failed for {len(failed_docs)} document(s): {preview}")
        return docid2nodes, {}  # empty — Stage 2 streams per-doc pkls

    _parallel_create_nodes_offline._patched_parallel = True  # type: ignore[attr-defined]
    HopBuilder.QABuilder.create_nodes_offline = _parallel_create_nodes_offline
    import config as _hop_config

    logger.info(
        "HopRAG: patched create_nodes_offline for doc-level parallelism (doc_workers=%d, chunk_threads=%d)",
        _doc_workers,
        _hop_config.max_thread_num,
    )


_EDGE_CHUNKED_THRESHOLD = int(os.environ.get("RAG_HOP_EDGE_CHUNK_THRESHOLD", "400"))
_EDGE_TOP_K = int(os.environ.get("RAG_HOP_EDGE_TOP_K", "30"))
_EDGE_CHUNK_SIZE = int(os.environ.get("RAG_HOP_EDGE_CHUNK_SIZE", "1000"))


def _patch_create_edge_batched() -> None:
    """Wrap create_edge to replace row-by-row INSERT loops with UNWIND batches.

    Strategy: run the pandas2-patched create_edge with a _NullDriver that
    silently discards all session.run() calls.  This populates self.edges and
    self.abstract2chunk via the existing pandas computation without touching
    Neo4j.  Then we do the actual batched INSERTs ourselves.

    Quality: identical edges are created — same source DataFrames, same Cypher
    relationship type (config.edge_name), same properties.  numpy int64 / float32
    arrays are explicitly cast to Python int / list so UNWIND nested-dict params
    serialize correctly over Bolt.
    """
    import HopBuilder

    if getattr(HopBuilder.QABuilder.create_edge, "_patched_edge_batched", False):
        return

    _orig_create_edge = HopBuilder.QABuilder.create_edge
    _batch_size = _EDGE_INSERT_BATCH

    class _NullSession:
        def run(self, *a, **kw):
            return iter([])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class _NullDriver:
        def session(self):
            return _NullSession()

    def _batched_create_edge(self, node2questiondict, docid2nodes):
        import config as _hop_config

        # Ensure a real driver exists before we swap it.
        if self.driver is None:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                _hop_config.neo4j_url,
                auth=(_hop_config.neo4j_user, _hop_config.neo4j_password),
                database=_hop_config.neo4j_dbname,
                notifications_disabled_categories=["DEPRECATION"],
            )

        real_driver = self.driver

        if len(node2questiondict) > _EDGE_CHUNKED_THRESHOLD:
            # Exhaustive scoring in bounded blocks: no dense-only preselection.
            from models.hoprag.exact_edges import exact_edges
            self.edges, self.abstract2chunk = exact_edges(
                node2questiondict, docid2nodes,
                HopBuilder.pending_dot_answerable, HopBuilder.sparse_similarity)

        else:
            self.driver = _NullDriver()
            try:
                # Runs all pandas computation; INSERT loops hit the null driver (no-op).
                _orig_create_edge(self, node2questiondict, docid2nodes)
            finally:
                self.driver = real_driver

        # self.edges and self.abstract2chunk are now populated.
        edge_name = _hop_config.edge_name

        pending2answerable_batch = (
            f"UNWIND $rows AS row "
            f"MATCH (a) WHERE id(a) = row.id1 "
            f"WITH a, row MATCH (b) WHERE id(b) = row.id2 "
            f"CREATE (a)-[r:{edge_name} {{keywords: row.keywords, embed: row.embed, "
            f"question: row.question}}]->(b)"
        )
        abstract2answerable_batch = (
            f"UNWIND $rows AS row "
            f"MATCH (a) WHERE id(a) = row.abstract_id "
            f"WITH a, row MATCH (b) WHERE id(b) = row.id2 "
            f"CREATE (a)-[r:{edge_name} {{keywords: row.keywords, embed: row.embed, "
            f"question: row.question}}]->(b)"
        )

        if self.edges is not None and len(self.edges) > 0:
            p2a_rows = []
            for _, row in self.edges.iterrows():
                emb = row["embedding_x"]
                if hasattr(emb, "tolist"):
                    emb = emb.tolist()
                p2a_rows.append(
                    {
                        "id1": int(row["node_id_x"]),
                        "id2": int(row["node_id_y"]),
                        "keywords": sorted(row["keywords_both"]),
                        "embed": emb,
                        "question": row["question_y"],
                    }
                )
            with self.driver.session() as session:
                for i in range(0, len(p2a_rows), _batch_size):
                    session.run(pending2answerable_batch, {"rows": p2a_rows[i : i + _batch_size]})
            logger.info("HopRAG batched edges: %d pending2answerable inserted", len(p2a_rows))

        if self.abstract2chunk is not None and len(self.abstract2chunk) > 0:
            a2a_rows = []
            for _, row in self.abstract2chunk.iterrows():
                emb = row["embedding"]
                if hasattr(emb, "tolist"):
                    emb = emb.tolist()
                abstract_id = docid2nodes[row["doc_id"]][0]
                a2a_rows.append(
                    {
                        "abstract_id": int(abstract_id),
                        "id2": int(row["node_id"]),
                        "keywords": sorted(row["keywords"]),
                        "embed": emb,
                        "question": row["question"],
                    }
                )
            with self.driver.session() as session:
                for i in range(0, len(a2a_rows), _batch_size):
                    session.run(abstract2answerable_batch, {"rows": a2a_rows[i : i + _batch_size]})
            logger.info("HopRAG batched edges: %d abstract2answerable inserted", len(a2a_rows))

    _batched_create_edge._patched_edge_batched = True  # type: ignore[attr-defined]
    HopBuilder.QABuilder.create_edge = _batched_create_edge
    logger.info("HopRAG: patched create_edge (UNWIND batch_size=%d)", _batch_size)


# ---------------------------------------------------------------- driver


def _build_official_edge_groups(corpus_tag: str, staged_dir: Path, staged_files: list[str]) -> dict[str, list[str]]:
    """Pass the corpus to native create_edge without consulting query/gold files."""
    return {"whole-corpus": list(staged_files)}


def _prune_stale_hoprag_sources(builder, staged_files: list[str], edge_type: str) -> bool:
    """Remove nodes outside the active snapshot and invalidate all old edges.

    The node label and relationship type are corpus-tag scoped. When any stale
    or unowned node exists, all remaining relationships of this corpus are
    cleared so edge groups can be rebuilt without retaining partial old state.
    """
    active_sources = sorted({Path(doc_id).stem for doc_id in staged_files})
    stale_where = "n.source IS NULL OR NOT n.source IN $active_sources"
    with builder.driver.session() as session:
        result = session.run(
            f"MATCH (n:{builder.label}) WHERE {stale_where} RETURN count(n) AS stale_count",
            {"active_sources": active_sources},
        )
        record = result.single()
        stale_count = int(record["stale_count"]) if record else 0
        if not stale_count:
            return False
        session.run(
            f"MATCH (n:{builder.label}) WHERE {stale_where} DETACH DELETE n",
            {"active_sources": active_sources},
        ).consume()
        session.run(f"MATCH (a:{builder.label})-[r:{edge_type}]-(b:{builder.label}) WITH DISTINCT r DELETE r").consume()
    logger.info(
        "HopRAG snapshot reconciliation: pruned %d stale nodes and cleared corpus edges for rebuild",
        stale_count,
    )
    return True


def _run_stage2_group_streaming(
    builder,
    per_doc_dir: Path,
    staged_files: list[str],
    config,
) -> None:
    """Insert nodes per document, then build edges per official problem group.

    The split keeps memory bounded. Source of truth is the content-addressed per-doc
    cache produced by Stage 1.
    """
    import gc
    import pickle

    if builder.driver is None:
        from neo4j import GraphDatabase

        builder.driver = GraphDatabase.driver(
            config.neo4j_url,
            auth=(config.neo4j_user, config.neo4j_password),
            database=config.neo4j_dbname,
            notifications_disabled_categories=["DEPRECATION"],
        )

    unwind_insert = (
        f"UNWIND $rows AS row "
        f"CREATE (n:{builder.label} {{text: row.text, keywords: row.keywords, "
        f"embed: row.embed, title: row.title}}) "
        f"RETURN id(n)"
    )
    backfill_cypher = (
        "UNWIND $rows AS row MATCH (n) WHERE id(n) = row.id SET n.source = row.source, n.title = row.title"
    )

    # Derive the staged directory from the per-doc cache layout.
    staged_dir = per_doc_dir.parents[1] / "_input"
    groups = _build_official_edge_groups(
        getattr(config, "corpus_tag", "default"),
        staged_dir,
        staged_files,
    )
    staged_titles: dict[str, str] = {}
    for doc_id in staged_files:
        with open(staged_dir / doc_id, "r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        staged_titles[Path(doc_id).stem] = first_line.removeprefix("Title: ").strip()

    # Resume support: load tracking sets from cache dir sibling of per_doc_dir.
    cache_dir = per_doc_dir.parent
    nodes_done_path = cache_dir / "stage2_nodes_done.pkl"
    edges_done_path = cache_dir / "stage2_edges_done.pkl"

    nodes_done: set[str] = set()
    edges_done: set[str] = set()
    if nodes_done_path.exists():
        with open(nodes_done_path, "rb") as fh:
            nodes_done = pickle.load(fh)
    if edges_done_path.exists():
        with open(edges_done_path, "rb") as fh:
            edges_done = pickle.load(fh)

    nodes_done &= set(staged_files)
    if _prune_stale_hoprag_sources(builder, staged_files, config.edge_name):
        # Stale-node deletion invalidates any edge group that included it.
        # Remaining corpus edges were cleared too, so every group must rerun.
        edges_done.clear()
    _atomic_pickle_dump(nodes_done_path, nodes_done)
    _atomic_pickle_dump(edges_done_path, edges_done)

    existing_by_source: dict[str, list[int]] = {}
    with builder.driver.session() as s:
        title_rows = [{"source": source, "title": title} for source, title in sorted(staged_titles.items())]
        s.run(
            f"UNWIND $rows AS row MATCH (n:{builder.label}) WHERE n.source = row.source SET n.title = row.title",
            {"rows": title_rows},
        ).consume()
        res = s.run(
            f"MATCH (n:{builder.label}) WHERE n.source IS NOT NULL RETURN n.source AS source, collect(id(n)) AS ids"
        )
        for rec in res:
            existing_by_source[str(rec["source"])] = list(rec["ids"])

    # A graph clear invalidates every marker. Otherwise validate node markers
    # against actual sources; edge markers remain valid only while all nodes
    # still exist, because upstream edges do not carry a problem/group id.
    nodes_done &= {doc_id for doc_id in staged_files if Path(doc_id).stem in existing_by_source}
    if not existing_by_source:
        edges_done.clear()
    _atomic_pickle_dump(nodes_done_path, nodes_done)
    _atomic_pickle_dump(edges_done_path, edges_done)

    if existing_by_source:
        logger.info(
            "HopRAG streaming: %d documents already have nodes (partial resume)",
            len(existing_by_source),
        )

    total_nodes = 0
    # Stage 2a: insert each document exactly once, independent of how many
    # problem contexts reference it.
    for doc_index, doc_id in enumerate(sorted(staged_files), start=1):
        pkl_file = per_doc_dir / (doc_id + ".pkl")
        try:
            with open(pkl_file, "rb") as handle:
                _local_nodes, local_n2q = pickle.load(handle)
        except Exception as exc:
            raise RuntimeError(f"HopRAG corrupt Stage 2 cache for {doc_id}") from exc
        if not local_n2q:
            # Upstream records an empty node list when every paragraph is
            # skipped by question generation. Preserve that method outcome.
            nodes_done.add(doc_id)
            _atomic_pickle_dump(nodes_done_path, nodes_done)
            continue

        stem = Path(doc_id).stem
        existing_ids = existing_by_source.get(stem, [])
        if existing_ids:
            nodes_done.add(doc_id)
            del local_n2q
            gc.collect()
            continue

        rows = []
        for node, _questiondict in local_n2q.values():
            embed = node["embed"]
            if hasattr(embed, "tolist"):
                embed = embed.tolist()
            rows.append(
                {
                    "text": node["text"],
                    "keywords": node["keywords"],
                    "embed": embed,
                    "title": staged_titles[stem],
                }
            )

        real_ids: list[int] = []
        with builder.driver.session() as session:
            for offset in range(0, len(rows), _NODE_INSERT_BATCH):
                batch = rows[offset : offset + _NODE_INSERT_BATCH]
                result = session.run(unwind_insert, {"rows": batch})
                batch_ids = [record[0] for record in result]
                real_ids.extend(batch_ids)
            backfill = [{"id": int(real_id), "source": stem, "title": staged_titles[stem]} for real_id in real_ids]
            session.run(backfill_cypher, {"rows": backfill})
        existing_by_source[stem] = real_ids
        nodes_done.add(doc_id)
        _atomic_pickle_dump(nodes_done_path, nodes_done)
        total_nodes += len(real_ids)
        if doc_index % 100 == 0 or doc_index == len(staged_files):
            logger.info(
                "HopRAG Stage 2a nodes: %d/%d documents, %d nodes inserted this run",
                doc_index,
                len(staged_files),
                total_nodes,
            )
        del local_n2q, rows, real_ids
        gc.collect()

    # Stage 2b: stream one official problem context at a time.
    for group_index, (group_id, doc_list) in enumerate(groups.items(), start=1):
        if group_id in edges_done:
            continue
        group_n2q: dict = {}
        group_docid2nodes: dict = {}
        for doc_id in doc_list:
            pkl_file = per_doc_dir / (doc_id + ".pkl")
            try:
                with open(pkl_file, "rb") as handle:
                    _local_nodes, local_n2q = pickle.load(handle)
            except Exception as exc:
                raise RuntimeError(f"HopRAG could not load edge cache for group={group_id}, doc={doc_id}") from exc
            real_ids = existing_by_source.get(Path(doc_id).stem, [])
            for real_id, ((_fake_id, did), (_node, questiondict)) in zip(real_ids, local_n2q.items()):
                group_n2q[(real_id, did)] = questiondict
                group_docid2nodes.setdefault(did, []).append(real_id)
        if not group_n2q:
            # Upstream create_edges_musique skips an all-empty problem group.
            edges_done.add(group_id)
            _atomic_pickle_dump(edges_done_path, edges_done)
            continue
        try:
            _patch_create_edge_batched()
            builder.create_edge(group_n2q, group_docid2nodes)
        except Exception as exc:
            raise RuntimeError(f"HopRAG edge build failed for group={group_id}") from exc
        edges_done.add(group_id)
        _atomic_pickle_dump(edges_done_path, edges_done)
        if group_index % 100 == 0 or group_index == len(groups):
            logger.info("HopRAG Stage 2b edges: %d/%d groups", group_index, len(groups))
        del group_n2q, group_docid2nodes
        gc.collect()

    set(groups) - edges_done
    logger.info("HopRAG streaming Stage 2 complete: %d nodes inserted this run", total_nodes)


def _run_official_index_blocking(
    dataset_path: str,
    corpus_tag: str,
    corpus_manifest: dict | None = None,
) -> dict[str, float]:
    """Synchronous driver — HopBuilder is sync, so we call it directly and
    let the orchestrator wrap us in run_in_executor."""
    _setup_hoprag_modules(corpus_tag)

    # Now safe to import HopBuilder (it does `from config import *`).
    import config
    import HopBuilder

    staged_input, staged_files = _stage_input_files(dataset_path, corpus_tag)
    source_ids = _expected_source_ids(staged_files, corpus_manifest)
    # This invalidates any prior completion marker before mutating the active
    # corpus-tagged label.  A crash/failure therefore cannot leave a stale
    # `complete` marker behind for a full benchmark to consume.
    _set_snapshot_state(config, corpus_tag, corpus_manifest, "in_progress")
    cache_dir = cache_dir_for(corpus_tag)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: build nodes (LLM-heavy: title/keywords + answerable+pending Qs).
    # main_nodes auto-resumes via docid2nodes.json cache.
    logger.info(
        "HopRAG official indexing: corpus_tag=%s, %d input files, output=%s, node_name=%s edge_name=%s gen=%s embed=%s",
        corpus_tag,
        len(staged_files),
        output_dir_for(corpus_tag),
        config.node_name,
        config.edge_name,
        _GEN_API_BASE,
        _EMBED_API_BASE,
    )

    stage_started = time.perf_counter()
    HopBuilder.main_nodes(
        cache_dir=str(cache_dir),
        docs_dir=str(staged_input),
        label=config.node_name,
        start_index=0,
        span=len(staged_files) + 100,
        offline=True,
    )
    timing = {"stage1_node_generation_seconds": time.perf_counter() - stage_started}

    # Stage 2: insert nodes once, then build edges one official group at a time.
    # Avoids loading all 324 docs × ~168 MB = ~54 GB into RAM at once.
    per_doc_dir = cache_dir / "docs"

    builder = HopBuilder.QABuilder(done=set(), label=config.node_name)
    stage_started = time.perf_counter()
    _run_stage2_group_streaming(builder, per_doc_dir, staged_files, config)
    timing["stage2_edge_build_seconds"] = time.perf_counter() - stage_started

    # Stage 3: vector + fulltext indices.
    stage_started = time.perf_counter()
    builder.create_index()
    timing["stage3_index_creation_seconds"] = time.perf_counter() - stage_started
    snapshot = _publish_snapshot(builder, corpus_tag, source_ids, corpus_manifest)
    timing["active_snapshot_verified"] = 1.0
    timing["active_snapshot_input_source_count"] = float(snapshot["input_source_count"])
    timing["active_snapshot_source_count"] = float(snapshot["source_count"])
    timing["active_snapshot_omitted_source_count"] = float(snapshot["omitted_source_count"])
    if builder.driver is not None:
        builder.driver.close()
        builder.driver = None
    logger.info("HopRAG official indexing complete for %s", corpus_tag)
    return timing


async def run_official_index(
    dataset_path: str,
    corpus_tag: str,
    corpus_manifest: dict | None = None,
) -> dict[str, float]:
    """Async wrapper around the sync HopBuilder driver."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        _run_official_index_blocking,
        dataset_path,
        corpus_tag,
        corpus_manifest,
    )
