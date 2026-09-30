"""Parameter-free fusion of indexed representation and body semantics."""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from core.config import RAGConfig
from core.generation_profiles import request_settings
from core.structured_outputs import ranking_contract
from models.prehop.tracing import traced
from utils.prompts.evidence_ranking import build_evidence_ranking_prompt
from utils.similarity import cosine_similarity

_CANDIDATE_ORDER_TRACE_LOCK = asyncio.Lock()


def _append_jsonl(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


class SimilarityScoringMixin:

    @staticmethod
    def _validated_similarity(query_embedding: list[float], document_embedding: Any) -> float:
        if not isinstance(document_embedding, list) or not document_embedding:
            raise ValueError("Retrieved candidate is missing its indexed embedding")
        if len(document_embedding) != len(query_embedding):
            raise ValueError("Query and indexed candidate embedding dimensions do not match")
        return cosine_similarity(query_embedding, document_embedding)

    @traced
    async def _score_and_select(
        self,
        query_embedding: list[float],
        candidates: list[dict[str, Any]],
        top_k: int,
        query_text: str = "",
        selection_variant: str | None = None,
        timing_sink: dict[str, float] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Fuse representation ranks with body and stored bridge semantics.

        The best matching individual source Q+ represents a traversed
        dependency bridge. The default uses body relevance; the historical
        body_bridge_min option takes the minimum with bridge relevance. Equal reciprocal ranks
        then combine the resulting semantic order with the Q-/body/Q+ retrieval
        order. Neither path compares backend-specific raw scores or introduces
        a fitted interpolation weight or threshold.
        """
        score_started = time.perf_counter()
        if not candidates:
            return [], []
        if not query_embedding:
            raise ValueError("Final scoring received an empty query embedding")

        for candidate in candidates:
            body_score = self._validated_similarity(query_embedding, candidate.get("embedding"))
            bridge_embeddings = candidate.get("bridge_embeddings") or []
            bridge_scores = [
                self._validated_similarity(query_embedding, embedding) for embedding in bridge_embeddings if embedding
            ]
            bridge_score = max(bridge_scores) if bridge_scores else None
            if bridge_score is None or RAGConfig.HOP_SEMANTIC_VARIANT == "body_only":
                final_score = body_score
            else:
                final_score = min(body_score, bridge_score)
            candidate["similarity_score"] = body_score
            if bridge_score is not None:
                candidate["bridge_similarity_score"] = bridge_score
            candidate["final_score"] = final_score

        semantic_order = sorted(
            candidates,
            key=lambda item: (item.get("final_score", 0.0), self._node_identity(item)),
            reverse=True,
        )
        representation_order = sorted(
            candidates,
            key=lambda item: (float(item.get("representation_score", 0.0)), self._node_identity(item)),
            reverse=True,
        )
        semantic_ranks = {self._node_identity(candidate): rank for rank, candidate in enumerate(semantic_order)}
        representation_ranks = {
            self._node_identity(candidate): rank
            for rank, candidate in enumerate(representation_order)
            if float(candidate.get("representation_score", 0.0)) > 0.0
        }
        for candidate in candidates:
            node_id = self._node_identity(candidate)
            score = 1.0 / (semantic_ranks[node_id] + 1)
            if node_id in representation_ranks:
                score += 1.0 / (representation_ranks[node_id] + 1)
            candidate["rank_fusion_score"] = score

        fused_order = sorted(
            candidates,
            key=lambda item: (
                float(item.get("rank_fusion_score", 0.0)),
                float(item.get("final_score", 0.0)),
                self._node_identity(item),
            ),
            reverse=True,
        )
        if RAGConfig.FINAL_RANK_VARIANT == "representation_only":
            ordered = representation_order
        else:
            ordered = fused_order
        if timing_sink is not None:
            timing_sink["deterministic_score_ms"] = (time.perf_counter() - score_started) * 1000
        ordering_started = time.perf_counter()
        active_selection_variant = selection_variant or RAGConfig.SOURCE_SELECTION_VARIANT
        if active_selection_variant == "global":
            selected = ordered[:top_k]
        elif active_selection_variant == "role_body_list_ranking":
            selected = await self._role_body_list_ranking(query_text, ordered, top_k)
        else:
            raise ValueError(f"Unsupported source selection variant: {active_selection_variant!r}")
        if timing_sink is not None:
            timing_sink["candidate_order_ms"] = (time.perf_counter() - ordering_started) * 1000
        return selected, ordered

    @classmethod
    def _prepare_ranking_candidates(
        cls,
        query_text: str,
        ordered: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Share native candidate identity and ordering with recorded-input replay."""
        pool: list[dict[str, Any]] = []
        seen_node_ids: set[str] = set()
        for node in ordered:
            node_id = cls._node_identity(node)
            if node_id and node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                pool.append(node)
        canonical_pool = list(pool)
        return canonical_pool, pool

    @classmethod
    def _complete_ranking(
        cls, selected: list[dict[str, Any]], ordered: list[dict[str, Any]], top_k: int,
    ) -> list[dict[str, Any]]:
        """Fill short rankings in canonical order, preserving returned duplicates."""
        selected_ids = {cls._node_identity(node) for node in selected}
        for node in ordered:
            node_id = cls._node_identity(node)
            if len(selected) >= top_k:
                break
            if node_id and node_id not in selected_ids:
                selected.append(node)
                selected_ids.add(node_id)
        return selected

    @traced
    async def _role_body_list_ranking(
        self,
        query_text: str,
        ordered: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rank the complete established candidate pool by opaque paragraph ID."""
        if not query_text.strip():
            raise ValueError("Body evidence candidate ordering requires the original query text")
        canonical_pool, pool = self._prepare_ranking_candidates(
            query_text, ordered,
        )
        if not pool:
            return ordered[:top_k]

        node_by_candidate_id, prompt = self._ranking_prompt(query_text, pool, top_k)
        contract = ranking_contract(list(node_by_candidate_id), top_k)
        payload = await self.llm.generate_json(
            [{"role": "user", "content": prompt}],
            json_debug_label="evidence ranking",
            structured_contract=contract,
            **request_settings("ranking"),
        )

        model_ranking = payload["ranking"]
        ranking = list(model_ranking)
        selected = [node_by_candidate_id[candidate_id] for candidate_id in ranking]
        trace_path = os.environ.get("RAG_CANDIDATE_ORDER_TRACE_PATH", "").strip()
        if trace_path:
            record = {
                "query": query_text,
                "structured_output": contract.provenance(),
                "top_k": top_k,
                "input_order": "search",
                "shuffle_seed": 0,
                "canonical_node_ids": [self._node_identity(node) for node in canonical_pool],
                "candidates": [
                    {
                        "node_id": self._node_identity(node),
                        "title": str(node.get("title") or node.get("doc") or ""),
                        "source": str(node.get("source") or ""),
                        "paragraph_id": str(node.get("paragraph_id") or ""),
                        "metadata": {
                            key: str(node[key])
                            for key in ("publisher", "published_at", "author", "category")
                            if node.get(key)
                        },
                        "text": str(node.get("text") or ""),
                        "similarity_score": float(node.get("similarity_score", 0.0)),
                        "bridge_similarity_score": (
                            float(node["bridge_similarity_score"]) if node.get("bridge_similarity_score") is not None else None
                        ),
                        "final_score": float(node.get("final_score", 0.0)),
                        "representation_score": float(node.get("representation_score", 0.0)),
                        "representation_scores": {
                            str(key): float(value)
                            for key, value in (node.get("representation_scores") or {}).items()
                        },
                        "rank_fusion_score": float(node.get("rank_fusion_score", 0.0)),
                        "retrieval_paths": node.get("retrieval_paths") or [],
                    }
                    for node in canonical_pool
                ],
                "model_returned_node_ids": [
                    self._node_identity(node_by_candidate_id[candidate_id]) for candidate_id in model_ranking
                ],
                "selected_node_ids": [self._node_identity(node) for node in selected],
            }
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            async with _CANDIDATE_ORDER_TRACE_LOCK:
                await asyncio.to_thread(_append_jsonl, Path(trace_path), line)
        return self._complete_ranking(selected, ordered, top_k)

    @staticmethod
    def _ranking_prompt(query_text, pool, top_k):
        """Share the exact selector rendering with fixed-candidate selector comparisons."""
        candidates = {f"C{index:03d}": node for index, node in enumerate(pool)}
        prompt = build_evidence_ranking_prompt(
            query_text,
            [
                (
                    candidate_id,
                    str(node.get("title") or node.get("doc") or ""),
                    "; ".join(
                        f"{label}: {node[key]}"
                        for key, label in (
                            ("publisher", "Publisher"),
                            ("published_at", "Published"),
                            ("author", "Author"),
                            ("category", "Category"),
                        )
                        if node.get(key)
                    ),
                    str(node.get("text") or ""),
                )
                for candidate_id, node in candidates.items()
            ],
            top_k,
        )
        return candidates, prompt
