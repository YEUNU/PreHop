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
from core.index_namespace import index_namespace
from core.neo4j_service import Neo4jService
from core.vllm_client import VLLMClient, get_llm_client
from utils.formatters import format_context_from_nodes
from utils.prompts.shared import build_answer_prompt, mark_answer_boundary

logger = logging.getLogger(__name__)
_SYNC_LOOP_STATE = threading.local()


# Match the pinned public HopGenerator CLI default.
OFFICIAL_HOPRAG_TOP_K = 8


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

    def __init__(self, model_id="default", max_hop=5, top_k=8, corpus_tag="default"):
        if (max_hop, top_k) != (5, 8):
            raise ValueError("Paper HopRAG requires native CLI defaults: max_hop=5, topk=8")
        from models.hoprag.native_runtime import setup, native_pipeline
        setup(corpus_tag)
        self.model_id, self.corpus_tag = model_id, corpus_tag
        self.max_hop, self.top_k = max_hop, top_k
        self.neo4j = Neo4jService()
        self.chunk_label = f"HO_{index_namespace(corpus_tag)}"
        self._pipeline = native_pipeline()
        self._retriever = self._pipeline.retriever

    async def _run_official_retrieval(self, query):
        context, _ = await asyncio.to_thread(self._retriever.search_docs, query)
        return context

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
            source_groups = {(str(row.get("source") or ""), str(row.get("title") or "")) for row in matches}
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

    async def run_workflow(self, query: str, history=None):
        from models.hoprag.native_runtime import QUERY
        token = QUERY.set(query)
        try:
            answer, context, scores = await asyncio.to_thread(self._pipeline.rag, query)
        finally:
            QUERY.reset(token)
        nodes = await self._lookup_nodes_by_text(context)
        sources = [{"doc": n.get("title") or n.get("source", ""), "source": n.get("source", ""),
                    "text": n.get("text", ""), "page": 0, "sent_id": 0,
                    **({"provenance_status": n["provenance_status"], "provenance_candidates": n["provenance_candidates"]}
                       if n.get("provenance_status") else {})} for n in nodes]
        return mark_answer_boundary(str(answer)), sources, [{"step": "hoprag_native_rag", "input": query,
                                                            "output": answer, "context": context, "scores": scores}]
