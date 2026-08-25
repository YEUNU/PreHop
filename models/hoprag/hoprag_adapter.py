"""
[HopRAG] adapter wired to the official HopRetriever implementation.

This keeps the benchmark interface while delegating traversal logic to
`third_party/HopRAG/HopRetriever.py` and preserving its published top-k/order.
"""

import asyncio
import importlib
import logging
import os
import sys
import threading
import types
from pathlib import Path
from typing import Any

from core.config import RAGConfig
from core.neo4j_service import Neo4jService
from core.vllm_client import VLLMClient, get_llm_client
from utils.formatters import format_context_from_nodes
from utils.prompts.shared import build_answer_prompt

logger = logging.getLogger(__name__)
_SYNC_LOOP_STATE = threading.local()


# The official HopRAG repository's end-to-end HopGenerator example uses
# `--topk 20`. Keep that published baseline setting instead of forcing the
# in-repo Prehop/Naive context budget onto the external method.
OFFICIAL_HOPRAG_TOP_K = 20


def _run_coro_sync(coro):
    """Run async coroutines from synchronous official HopRAG hooks.

    A hard timeout guards against a wedged HTTP connection hanging the whole
    benchmark forever (the loop-bound httpx read timeout does not fire when a
    pooled connection is reused across event loops). On timeout the coroutine
    is cancelled and TimeoutError propagates to the caller's try/except.
    """
    from core.config import RAGConfig

    base = RAGConfig.LLM_REQUEST_TIMEOUT or 300
    hard_timeout = base + 60

    async def _guarded():
        return await asyncio.wait_for(coro, timeout=hard_timeout)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # Official HopRetriever invokes this hook repeatedly from a stable
        # asyncio.to_thread worker. Creating a fresh loop for every node
        # judgement leaves cached AsyncOpenAI transports attached to hundreds
        # of dead loops and eventually exhausts file descriptors. Reuse one
        # loop per worker thread; this changes only adapter resource ownership,
        # not the upstream call sequence or responses.
        loop = getattr(_SYNC_LOOP_STATE, "loop", None)
        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
            _SYNC_LOOP_STATE.loop = loop
        return loop.run_until_complete(_guarded())

    holder: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}

    def _runner():
        try:
            holder["value"] = asyncio.run(_guarded())
        except BaseException as e:  # noqa: BLE001  # pragma: no cover - propagate thread cancellation/system exits
            errors["error"] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    if "error" in errors:
        raise errors["error"]
    return holder.get("value")


def _install_missing_hoprag_stubs() -> None:
    """Install tiny import-time stubs for optional upstream dependencies."""

    def _unavailable(name: str):
        def _raise(*_args, **_kwargs):
            raise RuntimeError(
                f"HopRAG optional dependency '{name}' was unexpectedly used; the local runtime hook was not installed"
            )

        return _raise

    if "paddlenlp" not in sys.modules:
        paddlenlp = types.ModuleType("paddlenlp")
        paddlenlp.Taskflow = _unavailable("paddlenlp")  # type: ignore[attr-defined]
        sys.modules["paddlenlp"] = paddlenlp


class HopRAGAdapter:
    """
    HopRAG benchmark adapter that executes the official HopRetriever traversal.
    """

    def __init__(
        self,
        model_id: str = "default",
        max_hop: int = 4,
        top_k: int = OFFICIAL_HOPRAG_TOP_K,
        corpus_tag: str = "default",
    ):
        self.model_id = model_id
        self.max_hop = max_hop
        self.top_k = top_k
        self.corpus_tag = corpus_tag

        self.llm = get_llm_client(model_id)
        self.vllm = VLLMClient(model_name=model_id)
        self.neo4j = Neo4jService()

        # Match the labels, relationship type, and vector index written by the
        # official indexer.
        import re as _re

        _safe_corpus = _re.sub(r"[^A-Za-z0-9_]", "_", self.corpus_tag)
        self.prefix = "HO_"
        self.chunk_label = f"{self.prefix}{_safe_corpus}"
        self.edge_type = f"{self.prefix}{_safe_corpus}_p2a"
        self.vector_index = f"{self.prefix}{_safe_corpus}_node_dense_idx"

        self._hop_module = self._load_official_hop_module()
        self._configure_official_hop_runtime()
        self._retriever = self._build_official_retriever()

    def _load_official_hop_module(self):
        _install_missing_hoprag_stubs()

        hop_root = Path(__file__).resolve().parents[2] / "third_party" / "HopRAG"
        if not hop_root.exists():
            raise RuntimeError(f"Official HopRAG not found: {hop_root}")

        root_text = str(hop_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        return importlib.import_module("HopRetriever")

    def _configure_official_hop_runtime(self) -> None:
        """Patch official runtime hooks to use this project's infra/services."""
        hop_module = self._hop_module
        tool_module = importlib.import_module("tool")

        neo4j_url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
        neo4j_password = os.environ.get("NEO4J_PASSWORD", "")
        neo4j_dbname = os.environ.get("NEO4J_DATABASE", "neo4j")

        # We DELIBERATELY do NOT override retrieve_node_dense_query or its
        # sparse/edge variants — config.py's templates already match the
        # HopRAG-native schema (RETURN node-object, columns text/embed/keywords).
        # An earlier override returned a Cypher dict literal which collided
        # with HopRetriever's runtime `.format()` substitution (the dict's
        # `{text: ...}` got parsed as a `{text}` placeholder, raising
        # KeyError('\\n    text')).
        #
        # We only override the expand/edge-walk queries, which need the
        # corpus-tagged label + relationship type. Use string concat (not
        # f-string with `{{ }}`) to avoid format-spec collisions.
        expand_logic_query = (
            "MATCH (dense_node:"
            + self.chunk_label
            + ")-[r:"
            + self.edge_type
            + "]-(logic_node:"
            + self.chunk_label
            + ") "
            "WHERE dense_node.text=$text "
            "RETURN logic_node"
        )
        expand_node_edge_query = (
            "MATCH (dense_node:"
            + self.chunk_label
            + ")-[out_edge:"
            + self.edge_type
            + "]-(out_node:"
            + self.chunk_label
            + ") "
            "WHERE dense_node.text=$text "
            "RETURN out_node, out_edge"
        )
        get_out_edge_query = (
            "MATCH (n:" + self.chunk_label + ")-[r:" + self.edge_type + "]->(m:" + self.chunk_label + ") "
            "WHERE n.embed=$embed AND n.text=$text "
            "RETURN r as out_edge, m as out_node"
        )

        def _load_embed_model(_name):
            # Embeddings are served by the project's vLLM embedding endpoint.
            return object()

        def _get_doc_embeds(documents, _model):
            if isinstance(documents, str):
                emb = _run_coro_sync(self.vllm.get_embedding(documents))
                if not emb:
                    raise ValueError("HopRAG query embedding is missing")
                return emb
            docs = [str(d) for d in documents]
            embs = _run_coro_sync(self.vllm.get_embeddings(docs))
            if len(embs) != len(docs) or any(not emb for emb in embs):
                raise ValueError(f"HopRAG embedding count mismatch: expected {len(docs)}, got {len(embs)}")
            return embs

        def _load_language_model(model_name):
            # Use model id as opaque identifier; chat completion is patched below.
            return model_name

        def _get_chat_completion(chat, return_json=True, model=None, max_tokens=4096, keys=None):
            messages = chat if isinstance(chat, list) else [{"role": "user", "content": str(chat)}]
            _ = model, max_tokens  # keep official signature compatibility
            if not return_json:
                response = _run_coro_sync(self.llm.generate_response(messages, temperature=0.0))
                if not str(response or "").strip():
                    raise ValueError("HopRAG LLM hook returned an empty response")
                return response, messages

            generated = _run_coro_sync(self.llm.generate_json(messages, temperature=0.0))
            if not isinstance(generated, dict):
                raise TypeError(f"HopRAG LLM hook expected dict, got {type(generated).__name__}")
            payload = generated

            missing_keys = [key for key in (keys or []) if key not in payload]
            if missing_keys:
                raise ValueError(f"HopRAG LLM hook missing required keys: {missing_keys}")
            values = [payload[key] for key in (keys or [])]
            if any(value is None or (isinstance(value, str) and not value.strip()) for value in values):
                raise ValueError("HopRAG LLM hook returned an empty required value")
            return (*values, messages)

        # All schema-tagged Neo4j index names (sparse + dense, node + edge)
        # must point at the corpus-tagged indices created by the indexer.
        sparse_node_index = self.vector_index.replace("_node_dense_idx", "_node_sparse_idx")
        sparse_edge_index = self.vector_index.replace("_node_dense_idx", "_edge_sparse_idx")
        dense_edge_index = self.vector_index.replace("_node_dense_idx", "_edge_dense_idx")

        patch_targets = [hop_module, tool_module]
        for target in patch_targets:
            target.neo4j_url = neo4j_url
            target.neo4j_user = neo4j_user
            target.neo4j_password = neo4j_password
            target.neo4j_dbname = neo4j_dbname
            target.node_dense_index_name = self.vector_index
            target.node_sparse_index_name = sparse_node_index
            target.edge_sparse_index_name = sparse_edge_index
            target.edge_dense_index_name = dense_edge_index
            target.expand_logic_query = expand_logic_query
            target.expand_node_edge_query = expand_node_edge_query
            target.get_out_edge_query = get_out_edge_query
            target.load_embed_model = _load_embed_model
            target.get_doc_embeds = _get_doc_embeds
            target.load_language_model = _load_language_model
            target.get_chat_completion = _get_chat_completion

    def _build_official_retriever(self):
        hop_cls = self._hop_module.HopRetriever
        return hop_cls(
            llm=self.model_id,
            max_hop=self.max_hop,
            entry_type="node",
            if_hybrid=False,
            if_trim=False,
            tol=2,
            topk=self.top_k,
            traversal="bfs_node",
            mock_dense=False,
            mock_sparse=False,
        )

    async def _run_official_retrieval(self, query: str) -> list[str]:
        context_texts, _ = await asyncio.to_thread(self._retriever.search_docs, query)
        if not isinstance(context_texts, list):
            raise TypeError(f"Official HopRetriever returned {type(context_texts).__name__}, expected list")
        for idx, text in enumerate(context_texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Official HopRetriever result at index {idx} is {type(text).__name__}, expected str"
                )
            if not text.strip():
                raise ValueError(f"Official HopRetriever result at index {idx} is blank")
        return context_texts

    async def _lookup_nodes_by_text(self, texts: list[str]) -> list[dict[str, Any]]:
        if not texts:
            return []
        # Official HopRetriever returns only text and internally keys its
        # traversal state by that text. Exact duplicate text from different
        # documents is therefore an upstream equivalence class, not a node
        # whose single source can be recovered. Keep the retrieved evidence
        # rank but expose ambiguous provenance explicitly; never choose one
        # of the matching documents and accidentally award title-level credit.
        # HopRAG-native nodes have no page or chunk index, so those stay 0.
        query = f"""
            UNWIND range(0, size($texts) - 1) AS idx
            WITH idx, $texts[idx] AS target_text
            MATCH (n:{self.chunk_label})
            WHERE n.text = target_text
            RETURN idx, elementId(n) AS id, coalesce(n.title, '') AS title,
                   0 AS sent_id, 0 AS page,
                   n.text AS text, n.embed AS embedding,
                   coalesce(n.source, '') AS source
            ORDER BY idx ASC, id ASC
        """
        async with self.neo4j.driver.session() as session:
            result = await session.run(query, {"texts": texts})  # type: ignore
            rows = [dict(r) async for r in result]

        by_idx: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            idx = int(row.get("idx", -1))
            if idx < 0:
                continue
            by_idx.setdefault(idx, []).append(row)

        ordered: list[dict[str, Any]] = []
        for idx, text in enumerate(texts):
            matches = by_idx.get(idx, [])
            if not matches:
                raise RuntimeError(f"HopRAG provenance lookup found no node for official result at index {idx}")
            source_groups = {
                (str(row.get("source") or ""), str(row.get("title") or "")) for row in matches
            }
            if len(source_groups) == 1:
                node = matches[0]
            else:
                node = {
                    "id": f"hoprag:ambiguous:{idx}",
                    "title": "Ambiguous exact-text provenance",
                    "source": "",
                    "sent_id": 0,
                    "page": 0,
                    "text": text,
                    "embedding": matches[0].get("embedding"),
                    "provenance_status": "ambiguous_exact_text",
                    "provenance_candidates": [
                        {
                            "id": row.get("id"),
                            "title": row.get("title", ""),
                            "source": row.get("source", ""),
                        }
                        for row in matches
                    ],
                }
            node.pop("idx", None)
            ordered.append(node)
        return ordered

    async def retrieve(self, query: str, top_k: int | None = None) -> tuple[str, list[dict[str, Any]]]:
        top_k = self.top_k if top_k is None else top_k
        context_texts = await self._run_official_retrieval(query)
        candidates = await self._lookup_nodes_by_text(context_texts)
        if not candidates:
            return "", []

        # Preserve the official HopRetriever ordering and published top-k.
        # Do not add adapter-only candidate widening or extra scoring.
        nodes = candidates[:top_k]
        context = format_context_from_nodes(nodes)
        return context, nodes

    async def run_workflow(self, query: str, history: list[dict] | None = None) -> tuple[str, list, list]:
        _ = history
        context, nodes = await self.retrieve(query, top_k=self.top_k)
        if not context:
            return (
                "Insufficient evidence.",
                [],
                [{"step": "hoprag_official_hopretriever_qa", "output": "empty_context"}],
            )

        prompt = build_answer_prompt(context, query)
        messages = [{"role": "user", "content": prompt}]
        answer = await self.llm.generate_response(
            messages,
            temperature=0.0,
            max_tokens=RAGConfig.SYNTHESIS_MAX_OUTPUT_TOKENS,
        )
        if not str(answer or "").strip():
            raise ValueError("Answer synthesis returned an empty response")
        trace = [{"step": "hoprag_official_hopretriever_qa", "input": messages, "output": answer}]
        sources = [
            {
                "doc": n.get("title") or n.get("source", ""),
                "source": n.get("source", ""),
                "page": n.get("page", 0),
                "text": n.get("text", ""),
                "sent_id": n.get("sent_id", 0),
                **(
                    {
                        "provenance_status": n["provenance_status"],
                        "provenance_candidates": n["provenance_candidates"],
                    }
                    if n.get("provenance_status")
                    else {}
                ),
            }
            for n in nodes
        ]
        return answer, sources, trace
