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
from core.index_namespace import index_namespace
from core.neo4j_service import Neo4jService
from core.vllm_client import VLLMClient, get_llm_client
from models.prehop.indexing import IndexingPipeline
from models.prehop.llm_json import generate_json_or_raise
from models.prehop.retrieval import RetrievalPipeline
from utils.prompts.query_rewrite import (
    build_evidence_conditioned_query_prompt,
    build_role_aligned_query_prompt,
)
from utils.prompts.shared import (
    build_answer_prompt as build_shared_answer_prompt,
)
from utils.prompts.shared import (
    mark_answer_boundary,
)

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
        self._safe_corpus = index_namespace(self.corpus_tag)

        self.chunk_label = f"{self.prefix}{self._safe_corpus}_Chunk"
        self.doc_label = f"{self.prefix}{self._safe_corpus}_Document"
        self.q_minus_label = f"{self.prefix}{self._safe_corpus}_QMinus"
        self.q_plus_label = f"{self.prefix}{self._safe_corpus}_QPlus"
        self.sentence_label = f"{self.prefix}{self._safe_corpus}_Sentence"
        self.answer_anchor_label = f"{self.prefix}{self._safe_corpus}_AnswerAnchor"

        self.body_vector_index = f"{self.strategy}_{self._safe_corpus}_vector_idx"
        self.body_text_index = f"{self.strategy}_{self._safe_corpus}_text_idx"
        self.q_minus_vector_index = f"{self.strategy}_{self._safe_corpus}_qminus_vector_idx"
        self.q_plus_vector_index = f"{self.strategy}_{self._safe_corpus}_qplus_vector_idx"
        self.q_minus_text_index = f"{self.strategy}_{self._safe_corpus}_qminus_text_idx"
        self.q_plus_text_index = f"{self.strategy}_{self._safe_corpus}_qplus_text_idx"
        self.sentence_vector_index = f"{self.strategy}_{self._safe_corpus}_sentence_vector_idx"
        self.sentence_text_index = f"{self.strategy}_{self._safe_corpus}_sentence_text_idx"
        self.vector_index = self.body_vector_index
        self.text_index = self.body_text_index

        self.neo4j = Neo4jService()
        self.llm = VLLMClient()
        self._index_ready = False

        indexing_model_id = indexing_model_id or RAGConfig.DEFAULT_MODEL
        self.indexing_model_id = indexing_model_id
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
        return mark_answer_boundary(answer)

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

    @staticmethod
    def _validate_role_queries(payload: dict[str, Any]) -> dict[str, list[str]]:
        validated: dict[str, list[str]] = {}
        for role in ("q_minus", "q_plus"):
            values = payload.get(role)
            if not isinstance(values, list):
                raise TypeError(f"Role-aligned query rewrite field {role!r} must be a list")
            questions: list[str] = []
            seen: set[str] = set()
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Role-aligned query rewrite field {role!r} contains a blank or non-string item")
                question = " ".join(value.split())
                identity = question.casefold()
                if identity not in seen:
                    seen.add(identity)
                    questions.append(question)
            if len(questions) > RAGConfig.QUESTIONS_PER_DIRECTION:
                logger.warning(
                    "Role-aligned query rewrite returned %d unique %s questions; retaining the first %d",
                    len(questions),
                    role,
                    RAGConfig.QUESTIONS_PER_DIRECTION,
                )
                questions = questions[: RAGConfig.QUESTIONS_PER_DIRECTION]
            validated[role] = questions
        return validated

    async def _rewrite_query_roles(self, query: str) -> dict[str, list[str]] | None:
        if RAGConfig.QUERY_REWRITE_VARIANT == "none":
            return None
        max_words = RAGConfig.QUERY_REWRITE_MAX_WORDS
        word_count = len(re.findall(r"\b\w+[\w'-]*\b", query))
        if max_words > 0 and word_count > max_words:
            return None
        prompt = build_role_aligned_query_prompt(query, RAGConfig.QUESTIONS_PER_DIRECTION)
        payload = await generate_json_or_raise(
            self.llm,
            [{"role": "user", "content": prompt}],
            "role-aligned query rewrite",
            f"query={query!r}",
            required_fields={"q_minus": list, "q_plus": list},
            temperature=0.0,
            max_tokens=512,
        )
        return self._validate_role_queries(payload)

    async def _refine_query_roles(
        self,
        query: str,
        evidence: str,
        attempted_questions: list[str],
    ) -> dict[str, list[str]]:
        prompt = build_evidence_conditioned_query_prompt(
            query,
            evidence,
            attempted_questions,
            RAGConfig.QUESTIONS_PER_DIRECTION,
        )
        payload = await generate_json_or_raise(
            self.llm,
            [{"role": "user", "content": prompt}],
            "evidence-conditioned role rewrite",
            f"query={query!r}",
            required_fields={"q_minus": list, "q_plus": list},
            temperature=0.0,
            max_tokens=512,
        )
        return self._validate_role_queries(payload)

    # ---------- main entry ----------
    async def run_workflow(
        self,
        user_query: str,
        history: list[dict[str, Any]] | None = None,
    ) -> tuple:
        """Run retrieval, traversal, and one synthesis call.

        The default path is:
          1. Parallel role-based retrieve (Q-/body evidence, Q+ dependency seeds).
          2. External-embedding cosine top-k ordering.
          3. Deterministic 1-hop bidirectional-NEXT/outgoing-HOP_ANSWER traversal
             (when RAG_GRAPH_HOP_DEPTH is 1, default).
          4. Single LLM synthesis call.
        """
        _ = history
        retrieval_query = self._strip_format_instruction(user_query)
        graph_depth = RAGConfig.GRAPH_HOP_DEPTH
        rewrite_started = time.perf_counter()
        role_queries = await self._rewrite_query_roles(retrieval_query)
        rewrite_ms = (time.perf_counter() - rewrite_started) * 1000 if role_queries is not None else 0.0
        channel_queries = None
        if role_queries is not None:
            additive = RAGConfig.QUERY_REWRITE_VARIANT == "role_aligned_additive"
            channel_queries = {
                "q_minus": list(dict.fromkeys([*([retrieval_query] if additive else []), *role_queries["q_minus"]])),
                "q_plus": list(dict.fromkeys([*([retrieval_query] if additive else []), *role_queries["q_plus"]])),
                "body": [retrieval_query],
            }

        async def execute_retrieval(
            queries: dict[str, list[str]] | None,
            selection_variant: str | None = None,
        ) -> tuple[str, list[dict[str, Any]], dict[str, float]]:
            if graph_depth > 0:
                selection_kwargs = {"selection_variant": selection_variant} if selection_variant is not None else {}
                return await self.graph_search(
                    entities=[retrieval_query],
                    depth=graph_depth,
                    top_k=RAGConfig.DEFAULT_TOP_K,
                    channel_queries=queries,
                    **selection_kwargs,
                )
            t_retrieve0 = time.perf_counter()
            if queries is None:
                result_context, result_nodes = await self.retrieve(retrieval_query, top_k=RAGConfig.DEFAULT_TOP_K)
            else:
                selection_kwargs = {"selection_variant": selection_variant} if selection_variant is not None else {}
                result_context, result_nodes = await self.retrieve_with_views(
                    retrieval_query,
                    top_k=RAGConfig.DEFAULT_TOP_K,
                    channel_queries=queries,
                    **selection_kwargs,
                )
            return (
                result_context,
                result_nodes if isinstance(result_nodes, list) else [],
                {"retrieve_ms": (time.perf_counter() - t_retrieve0) * 1000, "traversal_ms": 0.0},
            )

        retrieval_timing_keys = (
            "retrieve_ms",
            "traversal_ms",
            "graph_expand_ms",
            "deterministic_score_ms",
            "candidate_order_ms",
        )

        def add_timing(target: dict[str, float], source: dict[str, float]) -> None:
            for key in retrieval_timing_keys:
                target[key] = float(target.get(key, 0.0)) + float(source.get(key, 0.0))

        refinement_trace: list[dict[str, Any]] = []
        refinement_stop_reason = "not_applicable"
        retrieval_result: tuple[str, list[dict[str, Any]], dict[str, float]] | None = None
        evidence_variants = {
            "role_aligned_evidence",
            "role_aligned_evidence_iterative",
        }
        if RAGConfig.QUERY_REWRITE_VARIANT in evidence_variants and channel_queries is not None:
            preview_selection_variant = (
                "role_body_rounds" if RAGConfig.SOURCE_SELECTION_VARIANT == "role_body_list_ranking" else None
            )
            current_context, current_nodes, current_timing = await execute_retrieval(
                channel_queries,
                preview_selection_variant,
            )
            total_timing = {key: float(current_timing.get(key, 0.0)) for key in retrieval_timing_keys}
            attempted_questions = list(
                dict.fromkeys(
                    [
                        retrieval_query,
                        *role_queries["q_minus"],
                        *role_queries["q_plus"],
                    ]
                )
            )
            attempted_identities = {" ".join(question.casefold().split()) for question in attempted_questions}
            seen_evidence_ids = {self._node_identity(node) for node in current_nodes}
            refinement_stop_reason = "evidence_or_question_stability"

            while current_context and current_nodes:
                if (
                    RAGConfig.QUERY_REFINEMENT_MAX_ROUNDS > 0
                    and len(refinement_trace) >= RAGConfig.QUERY_REFINEMENT_MAX_ROUNDS
                ):
                    refinement_stop_reason = "configured_round_cap"
                    break
                refinement_evidence = self._fit_ranked_context(current_nodes, retrieval_query)
                if not refinement_evidence:
                    refinement_stop_reason = "context_budget"
                    break
                attempted_snapshot = {role: list(channel_queries[role]) for role in ("q_minus", "q_plus")}
                refine_started = time.perf_counter()
                proposed = await self._refine_query_roles(
                    retrieval_query,
                    refinement_evidence,
                    attempted_questions,
                )
                rewrite_ms += (time.perf_counter() - refine_started) * 1000
                refined_role_queries = {
                    role: [
                        question
                        for question in proposed[role]
                        if " ".join(question.casefold().split()) not in attempted_identities
                    ]
                    for role in ("q_minus", "q_plus")
                }
                refinement_trace.append(
                    {
                        "input": {
                            "query": retrieval_query,
                            "attempted": attempted_snapshot,
                        },
                        "output": refined_role_queries,
                    }
                )
                if not any(refined_role_queries.values()):
                    refinement_stop_reason = "no_new_questions"
                    break

                for role in ("q_minus", "q_plus"):
                    channel_queries[role] = list(dict.fromkeys([*channel_queries[role], *refined_role_queries[role]]))
                    attempted_questions.extend(refined_role_queries[role])
                    attempted_identities.update(
                        " ".join(question.casefold().split()) for question in refined_role_queries[role]
                    )

                next_context, next_nodes, next_timing = await execute_retrieval(
                    channel_queries,
                    preview_selection_variant,
                )
                add_timing(total_timing, next_timing)
                current_context, current_nodes = next_context, next_nodes

                if RAGConfig.QUERY_REWRITE_VARIANT == "role_aligned_evidence":
                    refinement_stop_reason = "single_refinement_variant"
                    break

                # Continue only when the accumulated role queries introduce a
                # previously unseen selected chunk. Exact question and chunk
                # identities provide the stopping rule; no score threshold or
                # dataset-specific round count participates.
                current_evidence_ids = {self._node_identity(node) for node in current_nodes}
                new_evidence_ids = current_evidence_ids - seen_evidence_ids
                seen_evidence_ids.update(current_evidence_ids)
                if not new_evidence_ids:
                    refinement_stop_reason = "no_new_evidence"
                    break

            if RAGConfig.SOURCE_SELECTION_VARIANT == "role_body_list_ranking":
                final_context, final_nodes, final_timing = await execute_retrieval(channel_queries)
                add_timing(total_timing, final_timing)
                retrieval_result = (final_context, final_nodes, total_timing)
            else:
                retrieval_result = (current_context, current_nodes, total_timing)

        if retrieval_result is None:
            retrieval_result = await execute_retrieval(channel_queries)
        context, nodes, timing = retrieval_result

        retrieved_nodes = nodes if isinstance(nodes, list) else []
        sources = self._build_unique_sources(retrieved_nodes)
        path_counts: dict[str, int] = {}
        for source in sources:
            for path in source.get("retrieval_paths", []):
                kind = str(path.get("kind") or "unknown")
                channel = str(path.get("channel") or "")
                label = f"{kind}:{channel}" if channel else kind
                path_counts[label] = path_counts.get(label, 0) + 1
        retrieved_source_ids = {
            str(source.get("source") or source.get("doc") or "").strip()
            for source in sources
            if str(source.get("source") or source.get("doc") or "").strip()
        }

        trace: list[dict[str, Any]] = []
        if role_queries is not None:
            trace.append(
                {
                    "step": "query_rewrite",
                    "input": {"query": retrieval_query, "variant": RAGConfig.QUERY_REWRITE_VARIANT},
                    "output": role_queries,
                    "rewrite_ms": rewrite_ms,
                    "refinement_rounds": len(refinement_trace),
                    "refinement_max_rounds": RAGConfig.QUERY_REFINEMENT_MAX_ROUNDS,
                    "refinement_stop_reason": refinement_stop_reason,
                }
            )
        for refinement in refinement_trace:
            trace.append(
                {
                    "step": "evidence_query_rewrite",
                    "input": refinement["input"],
                    "output": refinement["output"],
                }
            )
        trace.append(
            {
                "step": "retrieve",
                "input": {"query": user_query, "top_k": RAGConfig.DEFAULT_TOP_K, "graph_depth": graph_depth},
                "output": {
                    "retrieved_chunks": len(sources),
                    "retrieved_sources": len(retrieved_source_ids),
                    "retrieval_path_counts": path_counts,
                },
                "retrieve_ms": timing.get("retrieve_ms", 0.0),
                "traversal_ms": timing.get("traversal_ms", 0.0),
                "graph_expand_ms": timing.get("graph_expand_ms", 0.0),
                "deterministic_score_ms": timing.get("deterministic_score_ms", 0.0),
                "candidate_order_ms": timing.get("candidate_order_ms", 0.0),
            }
        )

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
            trace.append(
                {"step": "synthesis", "output": {"answer": answer, "reason": "context_budget"}, "synthesis_ms": 0.0}
            )
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
