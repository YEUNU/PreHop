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
import contextlib
import hashlib
import importlib
import io
import itertools
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("Prehop")

_HOPRAG_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "HopRAG"

_GEN_API_BASES: list[str] = [
    s.strip()
    for s in os.environ.get(
        "RAG_HOP_GEN_API_BASES",
        os.environ.get("RAG_HOP_GEN_API_BASE", os.environ.get("VLLM_URL", "")),
    ).split(",")
    if s.strip()
]
_GEN_API_BASE = _GEN_API_BASES[0] if _GEN_API_BASES else ""
_GEN_MODEL_NAME = os.environ.get("VLLM_SERVED_MODEL_NAME", "generation-model")
_EMBED_API_BASE = os.environ.get("RAG_HOP_EMBED_API_BASE", os.environ.get("VLLM_EMBED_URL", ""))
_EMBED_MODEL_NAME = os.environ.get(
    "RAG_HOP_EMBED_MODEL_NAME", os.environ.get("VLLM_SERVED_EMBED_MODEL_NAME", "embedding-model")
)
_EMBED_DIM = int(os.environ.get("RAG_HOP_EMBED_DIM", os.environ.get("NEO4J_VECTOR_DIMENSIONS", "1024")))
_GEN_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
_DOC_WORKERS = max(1, int(os.environ.get("RAG_HOP_DOC_WORKERS", "4")))
_NODE_INSERT_BATCH = max(1, int(os.environ.get("RAG_HOP_NODE_BATCH", "200")))
_EDGE_INSERT_BATCH = max(1, int(os.environ.get("RAG_HOP_EDGE_BATCH", "500")))

_OUTPUT_ROOT = Path(os.environ.get("RAG_HOP_OUTPUT_ROOT", "data/hoprag_output"))


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


# ---------------------------------------------------------------- file staging


def _stage_input_files(
    dataset_path: str,
    corpus_tag: str,
) -> tuple[Path, list[str]]:
    """Materialize every corpus file in a tag-scoped input directory."""
    src_root = Path(dataset_path)
    if not src_root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset_path}")

    files = sorted(p for p in src_root.iterdir() if p.suffix in (".txt", ".md"))

    if not files:
        raise ValueError(f"HopRAG staging selected no .txt/.md files from {dataset_path}")

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

        # vLLM batches via continuous batching; chunk to keep payloads sane.
        chunk = 64
        out = []
        for i in range(0, len(documents), chunk):
            batch = documents[i : i + chunk]
            r = self._sess.post(
                f"{self.base_url}/embeddings",
                json={
                    "model": self.model,
                    "input": batch,
                    "encoding_format": "float",
                },
                headers={"Authorization": f"Bearer {_GEN_API_KEY}"},
                timeout=180,
            )
            r.raise_for_status()
            data = r.json()["data"]
            for d in data:
                out.append(d["embedding"])

        arr = np.asarray(out, dtype=np.float32)
        return arr[0] if single else arr


def _install_round_robin_patch(config) -> None:
    """Round-robin gen endpoints by replacing tool.OpenAI with a subclass that
    rotates base_url on each instantiation. This preserves the original
    _get_chat_completion logic (return format, JSON parsing, try_run retries)
    and only changes which server each request goes to.

    Also caps max_tokens to 1024 so long 10-K documents (28K+ tokens) fit
    within the 32768 context window (32768 - 1024 = 31744 max input).
    """
    import urllib.error
    import urllib.request

    import tool
    from openai import OpenAI as _OrigOpenAI

    # Validate the OpenAI-compatible model registry rather than a proxy-specific
    # /health route. A two-second /health probe produced false negatives while
    # a healthy external server was busy near its max_num_seqs limit.
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
                    logger.warning("HopRAG: generation endpoint rejected after retries: %s (%s)", base, exc)
                else:
                    time.sleep(attempt)
    if not live_bases:
        raise ConnectionError(f"HopRAG: no configured generation endpoint passed its health check: {_GEN_API_BASES}")
    logger.info("HopRAG: live gen endpoints for round-robin: %s", live_bases)

    _cycle = itertools.cycle(live_bases)
    _lock = threading.Lock()
    _local_bases_set = set(live_bases) | set(_GEN_API_BASES)

    class _RoundRobinOpenAI(_OrigOpenAI):
        """Drop-in replacement: rotates base_url across live gen endpoints."""

        def __init__(self, api_key=None, base_url=None, **kwargs):
            if base_url and any(base_url.startswith(b.rstrip("/v1").rstrip("/")) for b in _local_bases_set):
                with _lock:
                    base_url = next(_cycle)
            super().__init__(api_key=api_key, base_url=base_url, **kwargs)

    # tool.py does `from openai import OpenAI` at module level; replacing
    # tool.OpenAI makes all subsequent `OpenAI(...)` calls in that module use
    # our subclass while preserving every other part of _get_chat_completion.
    tool.OpenAI = _RoundRobinOpenAI

    # Cap max_tokens and truncate input per LLM call.
    # Root cause of "Unterminated string" errors: per-chunk input was unbounded,
    # so the model generated question lists longer than _MAX_OUTPUT.
    # Fix: cap input to 3000 chars (~667 tokens) → question list ~10 items ~250t
    # output, well under the 512-token budget.
    _MAX_OUTPUT = 512
    _MAX_INPUT_CHARS = 3000  # chars — tight cap on each LLM call's user message

    _orig_get_chat_completion = tool._get_chat_completion

    def _capped_get_chat_completion(chat, return_json=True, model=None, max_tokens=4096, keys=None):
        # Truncate the last user message if it exceeds the input budget.
        messages = list(chat)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                text = messages[i].get("content", "")
                if len(text) > _MAX_INPUT_CHARS:
                    messages = list(messages)
                    messages[i] = {**messages[i], "content": text[:_MAX_INPUT_CHARS]}
                    logger.debug("HopRAG: truncated user message %d→%d chars", len(text), _MAX_INPUT_CHARS)
                break
        return _orig_get_chat_completion(
            messages,
            return_json=return_json,
            model=model,
            max_tokens=min(max_tokens, _MAX_OUTPUT),
            keys=keys,
        )

    tool._get_chat_completion = _capped_get_chat_completion

    # Patch txt2obj to fix a bug where `.replace('\\"', '"')` corrupts valid
    # JSON before parsing. Root cause: vLLM json_object mode returns properly
    # escaped JSON (e.g. "term \"N/M\""), but txt2obj unescapes \" → " before
    # json.loads, producing unbalanced quotes → JSONDecodeError → returns None.
    # Fix: try json.loads directly first (handles all standard JSON escapes),
    # and only fall back to the original clean_json_str path if that fails.
    _orig_txt2obj = tool.txt2obj

    def _safe_txt2obj(text):
        if not text:
            return _orig_txt2obj(text)
        # Fast path: vLLM json_object guarantees valid JSON — parse directly.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.debug("HopRAG response required upstream non-standard JSON cleanup")
        # Slow path: original clean_json_str logic for non-standard responses.
        return _orig_txt2obj(text)

    tool.txt2obj = _safe_txt2obj

    logger.info(
        "HopRAG: round-robin OpenAI patch + max_tokens cap installed across %d endpoints: %s",
        len(live_bases),
        live_bases,
    )


def _install_optional_stubs() -> None:
    """Stub HopRAG's heavy/Chinese-NLP deps that we don't need (paddlenlp +
    sentence_transformers + modelscope). They get imported at module load by
    third_party/HopRAG/tool.py but we replace their downstream calls."""
    import types

    def _unavailable(name: str):
        def _raise(*_args, **_kwargs):
            raise RuntimeError(
                f"HopRAG optional dependency '{name}' was unexpectedly used; the local runtime hook was not installed"
            )

        return _raise

    if "paddlenlp" not in sys.modules:
        m = types.ModuleType("paddlenlp")
        m.Taskflow = _unavailable("paddlenlp")
        sys.modules["paddlenlp"] = m
    if "sentence_transformers" not in sys.modules:
        m = types.ModuleType("sentence_transformers")

        class _ST:
            def __init__(self, *_args, **_kwargs):
                _unavailable("sentence_transformers")()

        m.SentenceTransformer = _ST
        sys.modules["sentence_transformers"] = m
    if "modelscope" not in sys.modules:
        m = types.ModuleType("modelscope")

        class _Dummy:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                _unavailable("modelscope")()

        m.AutoModelForCausalLM = _Dummy
        m.AutoModelForSequenceClassification = _Dummy
        m.AutoTokenizer = _Dummy
        sys.modules["modelscope"] = m


def _setup_hoprag_modules(corpus_tag: str) -> None:
    """Prep sys.path + override config + tool BEFORE HopBuilder imports."""
    _install_optional_stubs()
    if str(_HOPRAG_ROOT) not in sys.path:
        sys.path.insert(0, str(_HOPRAG_ROOT))

    # Upstream config.py prints its baked-in demo configuration at import time
    # (hotpot + a localhost model).  It is replaced immediately below and is
    # never used, so suppress that misleading vendor-only line in run logs.
    with contextlib.redirect_stdout(io.StringIO()):
        config = importlib.import_module("config")

    config.local_base = _GEN_API_BASE
    config.local_key = _GEN_API_KEY
    config.local_model_name = _GEN_MODEL_NAME
    config.query_generator_model = _GEN_MODEL_NAME
    config.traversal_model = _GEN_MODEL_NAME
    config.default_gpt_model = _GEN_MODEL_NAME

    config.embed_model = "qwen3_embed_via_vllm"
    config.embed_model_dict = {"qwen3_embed_via_vllm": "(vllm-served)"}
    config.embed_dim = _EMBED_DIM
    config.signal = "\n\n"
    config.max_thread_num = max(1, int(os.environ.get("RAG_HOP_MAX_THREADS", "8")))

    safe = "".join(c if c.isalnum() else "_" for c in corpus_tag)
    config.dataset_name = corpus_tag
    config.corpus_tag = corpus_tag
    config.node_name = f"HO_{safe}"
    config.edge_name = f"HO_{safe}_p2a"
    config.generator_label = f"HO_{safe}_"
    config.node_dense_index_name = f"HO_{safe}_node_dense_idx"
    config.edge_dense_index_name = f"HO_{safe}_edge_dense_idx"
    config.node_sparse_index_name = f"HO_{safe}_node_sparse_idx"
    config.edge_sparse_index_name = f"HO_{safe}_edge_sparse_idx"

    # Cypher templates were string-concat'd with the OLD edge_name at module
    # load. Rebuild with the new one.
    config.create_pending2answerable = (
        "MATCH (a), (b) WHERE id(a) = $id1 AND id(b) = $id2 "
        f"CREATE (a)-[r:{config.edge_name} "
        "{keywords: $keywords, embed: $embed, question: $answerable_question}]->(b)"
    )
    config.create_abstract2answerable = (
        "MATCH (a), (b) WHERE id(a) = $abstract_id AND id(b) = $id2 "
        f"CREATE (a)-[r:{config.edge_name} "
        "{keywords: $keywords, embed: $embed, question: $answerable_question}]->(b)"
    )

    # Reflect the external model in upstream's `local_model_name` field so
    # `_get_chat_completion`
    # routes to our vLLM rather than gpt.
    config.deployment_sign = {
        "gpt": {
            "base": getattr(config, "gpt_base", ""),
            "key": getattr(config, "gpt_key", ""),
            "default_model": "gpt-4o-mini",
        },
        config.local_model_name: {"base": config.local_base, "key": config.local_key},
    }

    # Round-robin across multiple gen endpoints when RAG_HOP_GEN_API_BASES has
    # more than one URL. Health-checks each endpoint first; only live servers
    # enter the cycle. Monkey-patches tool._get_chat_completion (thread-safe).
    if len(_GEN_API_BASES) >= 1:
        _install_round_robin_patch(config)

    # Neo4j connection from our env.
    config.neo4j_url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    config.neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    config.neo4j_password = os.environ.get("NEO4J_PASSWORD", "")
    config.neo4j_dbname = os.environ.get("NEO4J_DATABASE", "neo4j")

    # Patch tool to use vLLM embeddings instead of local SentenceTransformer.
    import tool

    embed_client = _VLLMEmbedClient(_EMBED_API_BASE, _EMBED_MODEL_NAME, _EMBED_DIM)
    tool.load_embed_model = lambda _name: embed_client

    # Default get_doc_embeds calls model.encode(...).tolist() — our wrapper
    # already returns numpy with .tolist(), so the original works as-is.

    # Replace paddlenlp-based POS tagging with spaCy. Original keeps content
    # words (nouns/proper-nouns/verbs/adj) and drops function words. Without
    # this, get_ner_eng falls back to character-level splitting (we saw
    # keywords like [' ', '0', '2', 'B'] in the smoke), which trashes
    # sparse_similarity in HopBuilder.create_edge.
    tool.get_ner_eng = _spacy_ner_eng

    # Fix: try_run() returns (None, None, None) on failure, but get_question_list
    # callers unpack only 2 values → ValueError: too many values to unpack.
    # Root cause: tool.try_run hardcodes a 3-tuple on exhausted retries, but
    # _get_chat_completion with keys=["Question List"] normally returns 2 values.
    # Patch get_question_list directly so exhausted retries fail the document
    # instead of silently dropping every affected chunk from the graph.

    def _safe_get_question_list(extract_template, sentences, query_generator):
        result = tool.get_chat_completion(
            [{"role": "user", "content": extract_template.format(sentences=sentences)}],
            keys=["Question List"],
            model=query_generator,
            max_tokens=4096,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError("HopRAG question generation returned an invalid result")
        question_list, _ = result
        if not isinstance(question_list, list) or not question_list:
            raise RuntimeError("HopRAG question generation returned no questions")
        return question_list

    tool.get_question_list = _safe_get_question_list
    import HopBuilder as _HB_tmp

    _HB_tmp.get_question_list = _safe_get_question_list

    _patch_hopbuilder_for_pandas2()
    _patch_create_nodes_offline_parallel()
    _patch_create_nodes_cache_batched()
    _patch_create_edge_batched()


_SPACY_NLP = None
_KEEP_POS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}


def _spacy_ner_eng(text: str):
    """spaCy substitute for paddlenlp Taskflow('pos_tagging').

    Returns a list of unique content-word lemmas. Filters punctuation, stop
    words, and function-word POS tags. Result feeds into node 'keywords'
    sets used by sparse_similarity in HopBuilder.create_edge.
    """
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy

        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser"])
    doc = _SPACY_NLP(str(text or ""))
    seen = set()
    out = []
    for tok in doc:
        if tok.is_punct or tok.is_space or tok.is_stop:
            continue
        if tok.pos_ not in _KEEP_POS:
            continue
        lemma = tok.lemma_.lower().strip()
        if not lemma or len(lemma) < 2:
            continue
        if lemma in seen:
            continue
        seen.add(lemma)
        out.append(lemma)
    return out


def _patch_hopbuilder_for_pandas2() -> None:
    """HopBuilder.create_edge does
        df.apply(lambda x: x['kw_x'].union(x['kw_y']), axis=1)
    which returns a Series of set objects. pandas 2.x expands those into
    multiple columns when assigned back, raising
    'Cannot set a DataFrame with multiple columns to the single column'.
    Wrap the union result in a list so pandas treats it as a scalar."""
    import HopBuilder

    if getattr(HopBuilder.QABuilder.create_edge, "_patched_for_pandas2", False):
        return

    import inspect
    import textwrap

    import pandas as pd

    src = inspect.getsource(HopBuilder.QABuilder.create_edge)
    src = textwrap.dedent(src)

    # pandas 2.x expands lambda-returned set/list into multiple columns when
    # assigned. Rewrite both `apply(...)` lines to use list comprehensions,
    # which always yield a single Series of scalars.
    replacements = [
        (
            "cartesian1['keywords_both']=cartesian1.apply(lambda x:x['keywords_x'].union(x['keywords_y']),axis=1)",
            "cartesian1['keywords_both']=[set(kx).union(set(ky)) for kx,ky in zip(cartesian1['keywords_x'],cartesian1['keywords_y'])]",
        ),
        (
            "cartesian2['keywords_both']=cartesian2.apply(lambda x:x['keywords_x'].union(x['keywords_y']),axis=1)",
            "cartesian2['keywords_both']=[set(kx).union(set(ky)) for kx,ky in zip(cartesian2['keywords_x'],cartesian2['keywords_y'])]",
        ),
    ]
    for old, new in replacements:
        if old not in src:
            raise RuntimeError(f"HopBuilder.create_edge source pattern not found: {old[:60]}...")
        src = src.replace(old, new)

    # Bind into the HopBuilder module namespace so cypher templates etc resolve.
    namespace = dict(HopBuilder.__dict__)
    namespace.update({"pd": pd})
    exec(compile(src, "<hop_create_edge_patched>", "exec"), namespace)  # noqa: S102 - patch vendored method source
    patched = namespace["create_edge"]
    patched._patched_for_pandas2 = True  # type: ignore[attr-defined]
    HopBuilder.QABuilder.create_edge = patched
    logger.info("HopRAG: patched QABuilder.create_edge for pandas-2 compatibility")


def _patch_create_nodes_offline_parallel() -> None:
    """Replace QABuilder.create_nodes_offline with a version that processes
    _DOC_WORKERS documents concurrently instead of sequentially.

    Inner per-doc chunk parallelism (max_thread_num) is preserved — the two
    levels of parallelism stack:
        total concurrent LLM calls ≈ _DOC_WORKERS × max_thread_num
    e.g. DOC_WORKERS=4 × CHUNK_THREADS=8 → 32 concurrent calls across 2 endpoints.

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

        # Load per-doc caches from any previous partial run.
        # Only load the node-ID list (.ids companion) to avoid pulling ~168MB of
        # embedding/question data per doc into RAM at startup (54GB for 324 docs).
        docid2nodes: dict = {}
        cached_doc_ids: set = set()

        for pkl_file in sorted(per_doc_dir.glob("*.pkl")):
            doc_id = pkl_file.stem
            source_file = Path(docs_dir) / doc_id
            digest_file = per_doc_dir / (doc_id + ".sha256")
            if not source_file.is_file() or not digest_file.is_file():
                continue
            expected_digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
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

        # Skip docs already in the main cache (self.done) OR per-doc cache.
        docs_to_process = [
            d for d in docs_pool[start_index : start_index + span] if d not in self.done and d not in cached_doc_ids
        ]

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
                    doc = fh.read()
                sentence2node = self.get_single_doc_qa(doc)
                if not sentence2node:
                    raise RuntimeError("HopRAG produced no indexable nodes for the document")
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
                    hashlib.sha256(Path(doc_path).read_bytes()).hexdigest(),
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


def _patch_create_nodes_cache_batched() -> None:
    """Replace create_nodes_cache with UNWIND batch INSERT (no per-doc sleep).

    Correctness guarantees:
    - Neo4j UNWIND ... CREATE ... RETURN id(n) returns IDs in the same order as
      the input list — this is a stable, documented property used in all Neo4j
      production batch patterns.
    - We assert len(returned_ids) == len(batch) and raise on mismatch so a
      silent mapping error is impossible.
    - numpy embed arrays are explicitly converted to Python lists so nested-dict
      UNWIND parameters serialize correctly over Bolt.
    """
    import HopBuilder

    if getattr(HopBuilder.QABuilder.create_nodes_cache, "_patched_batched", False):
        return

    _batch_size = _NODE_INSERT_BATCH

    def _batched_create_nodes_cache(self, cache_dir="path/to/cache_dir"):
        import json
        import pickle

        logger.info("HopRAG batched nodes: label=%s from %s", self.label, cache_dir)
        if self.driver is None:
            import config as _c
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                _c.neo4j_url,
                auth=(_c.neo4j_user, _c.neo4j_password),
                database=_c.neo4j_dbname,
            )

        with open(f"{cache_dir}/node2questiondict.pkl", "rb") as fh:
            old_node2questiondict = pickle.load(fh)
        with open(f"{cache_dir}/docid2nodes.json", "r") as fh:
            old_docid2nodes = json.load(fh)

        # Flatten all nodes into a list to allow cross-doc batching.
        # Order within each doc is preserved so docid2nodes ordering is stable.
        all_items = []  # (doc_id, text, keywords, embed_list, questiondict)
        for doc_id, old_node_ids in old_docid2nodes.items():
            for old_node in old_node_ids:
                node, questiondict = old_node2questiondict[(old_node, doc_id)]
                embed = node["embed"]
                if hasattr(embed, "tolist"):
                    embed = embed.tolist()
                all_items.append((doc_id, node["text"], node["keywords"], embed, questiondict))

        logger.info(
            "HopRAG batched nodes: inserting %d nodes in batches of %d",
            len(all_items),
            _batch_size,
        )

        unwind_query = (
            f"UNWIND $rows AS row "
            f"CREATE (n:{self.label} {{text: row.text, keywords: row.keywords, embed: row.embed}}) "
            f"RETURN id(n)"
        )

        new_node2questiondict: dict = {}
        new_docid2nodes: dict = {}

        with self.driver.session() as session:
            for i in range(0, len(all_items), _batch_size):
                batch = all_items[i : i + _batch_size]
                rows = [{"text": text, "keywords": kw, "embed": emb} for _, text, kw, emb, _ in batch]
                result = session.run(unwind_query, {"rows": rows})
                new_ids = [r[0] for r in result]

                if len(new_ids) != len(batch):
                    raise RuntimeError(
                        f"HopRAG batched nodes: UNWIND returned {len(new_ids)} IDs "
                        f"for batch of {len(batch)} — aborting to prevent ID mismatch"
                    )

                for (doc_id, _, _, _, questiondict), new_id in zip(batch, new_ids):
                    new_node2questiondict[(new_id, doc_id)] = questiondict
                    new_docid2nodes.setdefault(doc_id, []).append(new_id)

                if (i // _batch_size + 1) % 20 == 0 or i + _batch_size >= len(all_items):
                    logger.info(
                        "HopRAG batched nodes: %d/%d inserted",
                        min(i + _batch_size, len(all_items)),
                        len(all_items),
                    )

        return new_docid2nodes, new_node2questiondict

    _batched_create_nodes_cache._patched_batched = True  # type: ignore[attr-defined]
    HopBuilder.QABuilder.create_nodes_cache = _batched_create_nodes_cache
    logger.info(
        "HopRAG: patched create_nodes_cache (UNWIND batch_size=%d, sleep removed)",
        _batch_size,
    )


_EDGE_CHUNKED_THRESHOLD = int(os.environ.get("RAG_HOP_EDGE_CHUNK_THRESHOLD", "400"))
_EDGE_TOP_K = int(os.environ.get("RAG_HOP_EDGE_TOP_K", "30"))
_EDGE_CHUNK_SIZE = int(os.environ.get("RAG_HOP_EDGE_CHUNK_SIZE", "1000"))


def _edges_via_chunked_topk(
    node2questiondict: dict,
    docid2nodes: dict,
    top_k: int = _EDGE_TOP_K,
    chunk_size: int = _EDGE_CHUNK_SIZE,
):
    """Memory-safe replacement for HopBuilder.create_edge's O(N²) cross join.

    The original code calls pending_df.merge(answerable_df, how='cross') which
    materialises N_pending × N_answerable rows — for 3M's 14K nodes that's
    ~140K × 28K = 3.9B rows = ~31 GB OOM.

    This function instead:
    1. Computes cosine similarities in chunks of `chunk_size` pending questions.
    2. Keeps only the top-`top_k` answerable candidates per pending question.
    3. Builds a compact edge-candidate DataFrame (~N_p × top_k rows).
    4. Applies the same selection logic as create_edge:
       cartesian1 (best per pending, intra+cross) + cartesian2 (top-2 cross-doc).

    Returns (edges_df, abstract2chunk_df) with the same column schemas as
    self.edges / self.abstract2chunk after create_edge runs.
    """
    import pandas as pd

    data: list = []
    for (node_id, doc_id), qdict in node2questiondict.items():
        for question_label, tuplelist in qdict.items():
            for qi, tup in enumerate(tuplelist):
                question, keywords, emb = tup
                data.append(
                    {
                        "doc_id": doc_id,
                        "node_id": node_id,
                        "question_label": question_label,
                        "question_id": qi,
                        "embedding": np.asarray(emb, dtype=np.float32),
                        "question": question,
                        "keywords": keywords,
                    }
                )

    if not data:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(data)
    del data

    answerable_df = df[df["question_label"] == "answerable"].reset_index(drop=True)
    pending_df = df[df["question_label"] == "pending"].reset_index(drop=True)
    del df

    if len(answerable_df) == 0 or len(pending_df) == 0:
        return pd.DataFrame(), answerable_df

    # Stack embeddings into 2-D float32 arrays for chunked matmul.
    pending_emb = np.stack(pending_df["embedding"].values).astype(np.float32)  # (N_p, dim)
    answerable_emb = np.stack(answerable_df["embedding"].values).astype(np.float32)  # (N_a, dim)

    actual_k = min(top_k, len(answerable_df))
    best_a_idx = np.empty((len(pending_df), actual_k), dtype=np.int32)
    best_a_scores = np.empty((len(pending_df), actual_k), dtype=np.float32)

    for i in range(0, len(pending_df), chunk_size):
        chunk = pending_emb[i : i + chunk_size]  # (C, dim)
        sims = (chunk @ answerable_emb.T).astype(np.float32)  # (C, N_a)
        if actual_k < sims.shape[1]:
            idx = np.argpartition(sims, -actual_k, axis=1)[:, -actual_k:]
        else:
            idx = np.tile(np.arange(sims.shape[1]), (sims.shape[0], 1))
        scores = np.take_along_axis(sims, idx, axis=1)
        best_a_idx[i : i + chunk_size] = idx
        best_a_scores[i : i + chunk_size] = scores
        del sims, idx, scores

    del pending_emb, answerable_emb

    # Pre-extract arrays for fast row-level access (avoids repeated .iloc[pi]).
    p_nids = pending_df["node_id"].values
    p_dids = pending_df["doc_id"].values
    p_qs = pending_df["question"].values
    p_kws = pending_df["keywords"].values
    p_embs = pending_df["embedding"].values

    a_nids = answerable_df["node_id"].values
    a_dids = answerable_df["doc_id"].values
    a_qs = answerable_df["question"].values
    a_kws = answerable_df["keywords"].values

    edge_rows: list = []
    for pi in range(len(pending_df)):
        p_nid = p_nids[pi]
        p_did = p_dids[pi]
        for rank in range(actual_k):
            ai = int(best_a_idx[pi, rank])
            a_nid = a_nids[ai]
            if a_nid == p_nid:
                continue  # no self-loops
            p_kw: set = p_kws[pi]
            a_kw: set = a_kws[ai]
            union_kw = (p_kw | a_kw) if (p_kw or a_kw) else set()
            inter_kw = p_kw & a_kw
            sparse_sim = len(inter_kw) / len(union_kw) if union_kw else 0.0
            edge_rows.append(
                {
                    "node_id_x": int(p_nid),
                    "node_id_y": int(a_nid),
                    "doc_id_x": p_did,
                    "doc_id_y": a_dids[ai],
                    "question_x": p_qs[pi],
                    "question_y": a_qs[ai],
                    "keywords_both": union_kw,
                    "embedding_x": p_embs[pi],
                    "similarity": float(best_a_scores[pi, rank]) + sparse_sim,
                }
            )

    del best_a_idx, best_a_scores

    if not edge_rows:
        return pd.DataFrame(), answerable_df

    cartesian = pd.DataFrame(edge_rows)
    del edge_rows

    max_edges = 1_000_000_000
    inner_ratio = 1 / 4

    # cartesian1: best answerable per pending question (intra + cross doc).
    idx1 = cartesian.groupby("question_x")["similarity"].idxmax()
    cartesian1 = (
        cartesian.loc[idx1]
        .sort_values("similarity", ascending=False)
        .drop_duplicates(subset=["node_id_x", "node_id_y"], keep="first")
    )

    # cartesian2: cross-doc only, top-2 per pending question.
    cartesian2 = cartesian[cartesian["doc_id_x"] != cartesian["doc_id_y"]].copy()
    del cartesian
    cartesian2 = (
        cartesian2.sort_values(["question_x", "similarity"], ascending=[True, False])
        .groupby("question_x")
        .head(2)
        .sort_values("similarity", ascending=False)
        .drop_duplicates(subset=["node_id_x", "node_id_y"], keep="first")
    )
    trimmed = cartesian2.iloc[max_edges:]
    cartesian2 = cartesian2.iloc[:max_edges]
    if len(trimmed) > 0:
        trimmed = trimmed[~trimmed["node_id_x"].isin(cartesian2["node_id_x"])].groupby("node_id_x").head(1)
        cartesian2 = pd.concat([cartesian2, trimmed], ignore_index=True)
    del trimmed

    cartesian1 = cartesian1.iloc[: int(max_edges * inner_ratio)]
    cartesian2 = cartesian2.iloc[: int(max_edges * (1 - inner_ratio))]

    cols = ["node_id_x", "question_y", "keywords_both", "embedding_x", "node_id_y", "similarity"]
    edges_df = pd.concat([cartesian1[cols], cartesian2[cols]], ignore_index=True).drop_duplicates(
        subset=["node_id_x", "node_id_y"], keep="first"
    )

    used_q = set(cartesian1["question_y"].tolist()) | set(cartesian2["question_y"].tolist())
    abstract2chunk_df = answerable_df[~answerable_df["question"].isin(used_q)]

    logger.info(
        "HopRAG chunked edges: %d edges, %d abstract2chunk (top_k=%d, chunk=%d)",
        len(edges_df),
        len(abstract2chunk_df),
        top_k,
        chunk_size,
    )
    return edges_df, abstract2chunk_df


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
            )

        real_driver = self.driver

        if len(node2questiondict) > _EDGE_CHUNKED_THRESHOLD:
            # Large group: avoid O(N²) cross join OOM with chunked top-K.
            self.edges, self.abstract2chunk = _edges_via_chunked_topk(node2questiondict, docid2nodes)
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
            f"MATCH (a), (b) WHERE id(a) = row.id1 AND id(b) = row.id2 "
            f"CREATE (a)-[r:{edge_name} {{keywords: row.keywords, embed: row.embed, "
            f"question: row.question}}]->(b)"
        )
        abstract2answerable_batch = (
            f"UNWIND $rows AS row "
            f"MATCH (a), (b) WHERE id(a) = row.abstract_id AND id(b) = row.id2 "
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


def _build_official_edge_groups(
    corpus_tag: str,
    staged_dir: Path,
    staged_files: list[str],
) -> dict[str, list[str]]:
    """Build the small per-problem document groups used by official HopRAG.

    Upstream HopRAG creates HotpotQA/MuSiQue edges within each problem's
    supplied context instead of taking a corpus-wide Cartesian product. The
    normalized benchmark files retain only gold evidence, so use the raw
    dataset context when available and fall back to the normalized evidence
    list for MultiHop-RAG (which upstream does not publish a loader for).
    """
    import html

    title_to_file: dict[str, str] = {}
    for path in staged_dir.iterdir():
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        title = first_line.removeprefix("Title: ").strip()
        if title:
            title_to_file[title] = path.name

    def _resolve_titles(titles) -> list[str]:
        resolved = []
        for raw_title in titles or []:
            title = html.unescape(str(raw_title or "").strip())
            filename = title_to_file.get(title)
            if filename is None:
                import re

                safe = re.sub(r'[\\/*?:"<>|]', "_", title).strip()
                safe = re.sub(r"\s+", "_", safe)[:150] or "untitled"
                candidate = f"{safe}.txt"
                if (staged_dir / candidate).is_file():
                    filename = candidate
            if filename and filename not in resolved:
                resolved.append(filename)
        return resolved

    groups: dict[str, list[str]] = {}
    tag = corpus_tag.lower()
    if tag == "hotpotqa":
        import pyarrow.parquet as pq

        raw_path = Path("data/hotpotqa_distractor_validation.parquet")
        if not raw_path.is_file():
            raise FileNotFoundError(f"HotpotQA raw context is required for official HopRAG edges: {raw_path}")
        table = pq.read_table(raw_path, columns=["id", "context"])
        for row in table.to_pylist():
            context = row.get("context") or {}
            docs = _resolve_titles(context.get("title") or [])
            if docs:
                groups[str(row.get("id"))] = docs
    elif tag == "musique":
        raw_path = Path("data/musique_ans_v1.0_dev.jsonl")
        if not raw_path.is_file():
            raise FileNotFoundError(f"MuSiQue raw context is required for official HopRAG edges: {raw_path}")
        with open(raw_path, "r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("answerable") is False:
                    continue
                docs = _resolve_titles(paragraph.get("title") for paragraph in (row.get("paragraphs") or []))
                if docs:
                    groups[str(row.get("id"))] = docs
    elif tag == "multihoprag":
        raw_path = Path("data/MultiHopRAG.json")
        if not raw_path.is_file():
            raise FileNotFoundError(f"MultiHop-RAG source is required for HopRAG edges: {raw_path}")
        with open(raw_path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
        for idx, row in enumerate(rows):
            docs = _resolve_titles(evidence.get("title") for evidence in (row.get("evidence_list") or []))
            if docs:
                groups[str(idx)] = docs
    else:
        raise ValueError(f"HopRAG has no official edge grouping for unsupported corpus={corpus_tag!r}")

    if not groups:
        raise ValueError(f"HopRAG found no edge-construction groups for corpus={corpus_tag}")
    grouped_docs = {doc for docs in groups.values() for doc in docs}
    missing_docs = sorted(set(staged_files) - grouped_docs)
    if missing_docs:
        raise RuntimeError(
            "HopRAG official edge groups do not cover every staged document: "
            f"missing={len(missing_docs)}, sample={missing_docs[:5]}"
        )
    logger.info(
        "HopRAG official edge groups: %d problems, %d/%d referenced documents",
        len(groups),
        len(grouped_docs),
        len(staged_files),
    )
    return groups


def _run_stage2_group_streaming(
    builder,
    per_doc_dir: Path,
    staged_files: list[str],
    config,
) -> None:
    """Insert nodes per document, then build edges per official problem group.

    The split keeps memory bounded and supports documents shared by many
    HotpotQA/MuSiQue problems. Source of truth is the content-addressed per-doc
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
        )

    unwind_insert = (
        f"UNWIND $rows AS row "
        f"CREATE (n:{builder.label} {{text: row.text, keywords: row.keywords, "
        f"embed: row.embed}}) "
        f"RETURN id(n)"
    )
    backfill_cypher = "UNWIND $rows AS row MATCH (n) WHERE id(n) = row.id SET n.source = row.source"

    # Derive the staged directory from the per-doc cache layout.
    staged_dir = per_doc_dir.parents[1] / "_input"
    groups = _build_official_edge_groups(
        getattr(config, "corpus_tag", "default"),
        staged_dir,
        staged_files,
    )

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

    existing_by_source: dict[str, list[int]] = {}
    with builder.driver.session() as s:
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
        if not pkl_file.is_file():
            raise FileNotFoundError(f"HopRAG Stage 2 cache missing for {doc_id}: {pkl_file}")
        try:
            with open(pkl_file, "rb") as handle:
                _local_nodes, local_n2q = pickle.load(handle)
        except Exception as exc:
            raise RuntimeError(f"HopRAG corrupt Stage 2 cache for {doc_id}") from exc
        if not local_n2q:
            raise RuntimeError(f"HopRAG Stage 2 cache has no nodes for {doc_id}")

        stem = Path(doc_id).stem
        existing_ids = existing_by_source.get(stem, [])
        if existing_ids:
            if len(existing_ids) != len(local_n2q):
                raise RuntimeError(
                    f"HopRAG resume count mismatch for {doc_id}: Neo4j={len(existing_ids)}, cache={len(local_n2q)}"
                )
            nodes_done.add(doc_id)
            del local_n2q
            gc.collect()
            continue

        rows = []
        for node, _questiondict in local_n2q.values():
            embed = node["embed"]
            if hasattr(embed, "tolist"):
                embed = embed.tolist()
            rows.append({"text": node["text"], "keywords": node["keywords"], "embed": embed})

        real_ids: list[int] = []
        with builder.driver.session() as session:
            for offset in range(0, len(rows), _NODE_INSERT_BATCH):
                batch = rows[offset : offset + _NODE_INSERT_BATCH]
                result = session.run(unwind_insert, {"rows": batch})
                batch_ids = [record[0] for record in result]
                if len(batch_ids) != len(batch):
                    raise RuntimeError(f"HopRAG UNWIND returned {len(batch_ids)} IDs for {len(batch)} nodes")
                real_ids.extend(batch_ids)
            backfill = [{"id": int(real_id), "source": stem} for real_id in real_ids]
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
            if len(real_ids) != len(local_n2q):
                raise RuntimeError(f"HopRAG edge input mismatch for group={group_id}, doc={doc_id}")
            for real_id, ((_fake_id, did), (_node, questiondict)) in zip(real_ids, local_n2q.items()):
                group_n2q[(real_id, did)] = questiondict
                group_docid2nodes.setdefault(did, []).append(real_id)
        if not group_n2q:
            raise RuntimeError(f"HopRAG edge group {group_id} has no nodes")
        try:
            builder.create_edge(group_n2q, group_docid2nodes)
        except Exception as exc:
            raise RuntimeError(f"HopRAG edge build failed for group={group_id}") from exc
        edges_done.add(group_id)
        _atomic_pickle_dump(edges_done_path, edges_done)
        if group_index % 100 == 0 or group_index == len(groups):
            logger.info("HopRAG Stage 2b edges: %d/%d groups", group_index, len(groups))
        del group_n2q, group_docid2nodes
        gc.collect()

    missing_edge_groups = set(groups) - edges_done
    if missing_edge_groups:
        raise RuntimeError(
            "HopRAG Stage 2 incomplete; missing completed edge groups: " + ", ".join(sorted(missing_edge_groups)[:10])
        )
    logger.info("HopRAG streaming Stage 2 complete: %d nodes inserted this run", total_nodes)


def _run_official_index_blocking(
    dataset_path: str,
    corpus_tag: str,
) -> None:
    """Synchronous driver — HopBuilder is sync, so we call it directly and
    let the orchestrator wrap us in run_in_executor."""
    _setup_hoprag_modules(corpus_tag)

    # Now safe to import HopBuilder (it does `from config import *`).
    import config
    import HopBuilder

    staged_input, staged_files = _stage_input_files(dataset_path, corpus_tag)
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

    HopBuilder.main_nodes(
        cache_dir=str(cache_dir),
        docs_dir=str(staged_input),
        label=config.node_name,
        start_index=0,
        span=len(staged_files) + 100,
        offline=True,
    )

    # Stage 2: insert nodes once, then build edges one official group at a time.
    # Avoids loading all 324 docs × ~168 MB = ~54 GB into RAM at once.
    per_doc_dir = cache_dir / "docs"
    if not per_doc_dir.exists() or not any(per_doc_dir.glob("*.pkl")):
        raise RuntimeError(f"HopRAG per-doc cache missing at {per_doc_dir}; Stage 1 produced no usable output")

    builder = HopBuilder.QABuilder(done=set(), label=config.node_name)
    _run_stage2_group_streaming(builder, per_doc_dir, staged_files, config)

    # Stage 3: vector + fulltext indices.
    builder.create_index()
    if builder.driver is not None:
        builder.driver.close()
        builder.driver = None
    logger.info("HopRAG official indexing complete for %s", corpus_tag)


async def run_official_index(
    dataset_path: str,
    corpus_tag: str,
) -> None:
    """Async wrapper around the sync HopBuilder driver."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        _run_official_index_blocking,
        dataset_path,
        corpus_tag,
    )
