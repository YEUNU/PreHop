"""Compose offline graph indexing and deterministic query-time retrieval.

All Neo4j labels and index names are derived from (strategy, corpus_tag) so
multiple corpora and strategies coexist in the same database without
collision.
"""

import asyncio
import logging
import os
import re
import time
from typing import Any

from core.config import RAGConfig
from core.neo4j_service import Neo4jService
from core.vllm_client import VLLMClient, get_llm_client
from models.prehop.indexing import IndexingPipeline
from models.prehop.retrieval import RetrievalPipeline
from utils.prompts.shared import build_answer_prompt as build_shared_answer_prompt

_ANSWER_PREFIX = "@@ANSWER:"


logger = logging.getLogger(__name__)


class GraphRAG(IndexingPipeline, RetrievalPipeline):
    def __init__(
        self,
        strategy: str = "prehop",
        indexing_model_id: str | None = None,
        corpus_tag: str | None = None,
        save_intermediate: bool = False,
    ):
        self.strategy = strategy.lower()
        self.corpus_tag = corpus_tag or "default"
        self.prefix = self.strategy[:2].upper() + "_"
        self._safe_corpus = re.sub(r"[^A-Za-z0-9_]", "_", self.corpus_tag)

        self.chunk_label = f"{self.prefix}{self._safe_corpus}_Chunk"
        self.doc_label = f"{self.prefix}{self._safe_corpus}_Document"
        self.q_minus_label = f"{self.prefix}{self._safe_corpus}_QMinus"
        self.q_plus_label = f"{self.prefix}{self._safe_corpus}_QPlus"

        self.body_vector_index = f"{self.strategy}_{self._safe_corpus}_vector_idx"
        self.body_text_index = f"{self.strategy}_{self._safe_corpus}_text_idx"
        self.q_minus_vector_index = f"{self.strategy}_{self._safe_corpus}_qminus_vector_idx"
        self.q_plus_vector_index = f"{self.strategy}_{self._safe_corpus}_qplus_vector_idx"
        self.q_minus_text_index = f"{self.strategy}_{self._safe_corpus}_qminus_text_idx"
        self.q_plus_text_index = f"{self.strategy}_{self._safe_corpus}_qplus_text_idx"
        self.vector_index = self.body_vector_index
        self.text_index = self.body_text_index

        self.neo4j = Neo4jService()
        self.llm = VLLMClient()
        self._index_ready = False

        indexing_model_id = indexing_model_id or RAGConfig.DEFAULT_MODEL
        self.indexing_llm = get_llm_client(indexing_model_id)

        self.max_retries = RAGConfig.RETRY_COUNT
        self.vector_dimensions = RAGConfig.EMBEDDING_DIMENSIONS
        self._pending_batch = []
        self._batch_lock = asyncio.Lock()
        self._index_setup_lock = asyncio.Lock()
        raw_run_id = os.environ.get("RAG_RUN_ID") or (
            f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_{os.getpid()}"
        )
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_run_id).strip("._-") or f"run_{os.getpid()}"
        self.debug_output_dir = os.path.join("data", "debug", safe_run_id, self.strategy, self._safe_corpus)
        self.save_intermediate = save_intermediate

    # ---------- helpers ----------
    @classmethod
    def _ensure_answer_prefix(cls, answer: str) -> str:
        text = str(answer or "")
        if _ANSWER_PREFIX not in text:
            return f"{_ANSWER_PREFIX} {text}"
        return text

    @staticmethod
    def _strip_format_instruction(query: str) -> str:
        marker = "[Benchmark Output Format]"
        if marker in query:
            return query.split(marker, 1)[0].strip()
        return query

    @staticmethod
    def _build_unique_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen = set()
        for row in rows:
            doc = row.get("title") or row.get("doc") or "Unknown"
            page = row.get("page", 0)
            sent_id = row.get("sent_id", 0)
            # Corpus filenames are stable identities. Source-less rows use
            # display title, page, and sentence identity.
            source = row.get("source")
            key = ("source", source, page, sent_id) if source else ("legacy", doc, page, sent_id)
            if key in seen:
                continue
            source_record = {
                "doc": doc,
                "page": page,
                "text": row.get("text", ""),
                "sent_id": sent_id,
            }
            # Preserve the filename only when present; unit-test/mocked
            # source shapes retain their established public representation.
            if source:
                source_record["source"] = source
            if row.get("id"):
                source_record["chunk_id"] = row["id"]
            if row.get("retrieval_paths"):
                source_record["retrieval_paths"] = row["retrieval_paths"]
            unique.append(source_record)
            seen.add(key)
        return unique

    @staticmethod
    def _build_answer_prompt(context: str, user_query: str) -> str:
        # Use the same single-pass synthesis prompt as controlled baselines.
        # to the HopRAG / naive baseline prompts so any score gap traces back
        # to retrieval, not synthesis-prompt asymmetry. The role is
        # dataset-neutral. MS GraphRAG owns synthesis inside its
        # official local/global search API and is the documented exception.
        return build_shared_answer_prompt(context, user_query)

    def _fit_ranked_context(self, nodes: list[dict[str, Any]], query: str) -> str:
        """Add ranked chunks until the actual synthesis prompt reaches its budget."""
        accepted: list[dict[str, Any]] = []
        for node in nodes:
            candidate = self._build_context_from_nodes([*accepted, node])
            prompt = self._build_answer_prompt(candidate, query)
            messages = [{"role": "user", "content": prompt}]
            counted = self.llm._count_tokens(messages)
            prompt_tokens = counted if isinstance(counted, int) else max(1, len(prompt) // 4)
            if prompt_tokens + RAGConfig.SYNTHESIS_MAX_OUTPUT_TOKENS > RAGConfig.MAX_CONTEXT_LENGTH:
                break
            accepted.append(node)
        return self._build_context_from_nodes(accepted)

    # ---------- main entry ----------
    async def run_workflow(
        self,
        user_query: str,
        history: list[dict[str, Any]] | None = None,
    ) -> tuple:
        """Run retrieval, traversal, and one synthesis call.

        No agent loop, no reflection, no refinement. The path is:
          1. Parallel role-based retrieve (Q-/body evidence, Q+ dependency seeds).
          2. External-embedding cosine top-k ordering.
          3. Deterministic 1-hop bidirectional-NEXT/outgoing-HOP_ANSWER traversal
             (when RAG_GRAPH_HOP_DEPTH is 1, default).
          4. Single LLM synthesis call.
        """
        _ = history
        retrieval_query = self._strip_format_instruction(user_query)
        graph_depth = RAGConfig.GRAPH_HOP_DEPTH

        if graph_depth > 0:
            context, nodes, timing = await self.graph_search(
                entities=[retrieval_query],
                depth=graph_depth,
                top_k=RAGConfig.DEFAULT_TOP_K,
            )
        else:
            t_retrieve0 = time.perf_counter()
            context, nodes = await self.retrieve(retrieval_query, top_k=RAGConfig.DEFAULT_TOP_K)
            timing = {"retrieve_ms": (time.perf_counter() - t_retrieve0) * 1000, "traversal_ms": 0.0}

        retrieved_nodes = nodes if isinstance(nodes, list) else []
        sources = self._build_unique_sources(retrieved_nodes)
        path_counts: dict[str, int] = {}
        for source in sources:
            for path in source.get("retrieval_paths", []):
                kind = str(path.get("kind") or "unknown")
                channel = str(path.get("channel") or "")
                label = f"{kind}:{channel}" if channel else kind
                path_counts[label] = path_counts.get(label, 0) + 1

        trace: list[dict[str, Any]] = [
            {
                "step": "retrieve",
                "input": {"query": user_query, "top_k": RAGConfig.DEFAULT_TOP_K, "graph_depth": graph_depth},
                "output": {"retrieved_sources": len(sources), "retrieval_path_counts": path_counts},
                "retrieve_ms": timing.get("retrieve_ms", 0.0),
                "traversal_ms": timing.get("traversal_ms", 0.0),
            }
        ]

        if not context:
            answer = self._ensure_answer_prefix("Insufficient evidence.")
            trace.append(
                {
                    "step": "synthesis",
                    "output": {"answer": answer, "reason": "empty_context"},
                    "synthesis_ms": 0.0,
                }
            )
            return answer, sources, trace

        context = self._fit_ranked_context(retrieved_nodes, retrieval_query)
        if not context:
            answer = self._ensure_answer_prefix("Insufficient evidence.")
            trace.append({"step": "synthesis", "output": {"answer": answer, "reason": "context_budget"}, "synthesis_ms": 0.0})
            return answer, sources, trace
        prompt = self._build_answer_prompt(context, retrieval_query)
        messages = [{"role": "user", "content": prompt}]
        t_synthesis0 = time.perf_counter()
        raw = await self.llm.generate_response(
            messages,
            temperature=0.0,
            max_tokens=RAGConfig.SYNTHESIS_MAX_OUTPUT_TOKENS,
        )
        synthesis_ms = (time.perf_counter() - t_synthesis0) * 1000
        if not str(raw or "").strip():
            raise ValueError("Answer synthesis returned an empty response")
        answer = self._ensure_answer_prefix(str(raw))
        trace.append(
            {
                "step": "synthesis",
                "output": {"answer": answer},
                "synthesis_ms": synthesis_ms,
            }
        )
        return answer, sources, trace
