"""Embedding-similarity reranking (core-only rewrite — replaces the
cross-encoder reranker, which required a dedicated reranker model that is no
longer part of the served model set).

Score combined with meta boost and boilerplate penalty:
final_score = cosine_sim + W_meta * meta_boost - W_boilerplate * boilerplate_penalty

Threshold tau_r = RAGConfig.RERANKER_THRESHOLD. NOTE: this threshold was
calibrated for cross-encoder classifier scores (roughly a 0-1 probability);
now that it gates raw bi-encoder cosine similarity, it likely needs
re-tuning empirically.
"""
from typing import Any

from core.config import RAGConfig
from models.prehop.llm_json import generate_json_or_raise
from utils.prompts import (
    RERANK_QUERY_SIMPLIFY_FORMAT_INSTRUCTION,
    RERANK_QUERY_SIMPLIFY_PROMPT,
)
from utils.similarity import cosine_similarity


class RerankMixin:
    async def _embedding_rerank_scores(self, query: str, texts: list[str]) -> list[float]:
        """Bi-encoder replacement for the old cross-encoder rerank call:
        embed the query and candidate texts, score by cosine similarity."""
        if not texts:
            return []
        query_embed, doc_embeds = await self._get_query_and_doc_embeddings(query, texts)
        return [cosine_similarity(query_embed, doc_embed) for doc_embed in doc_embeds]

    async def _get_query_and_doc_embeddings(self, query: str, texts: list[str]):
        query_embeds = await self.llm.get_embeddings([query], encoding_type="query")
        doc_embeds = await self.llm.get_embeddings(texts, encoding_type="document")
        return query_embeds[0], doc_embeds

    async def _simplified_rerank_query(self, query: str) -> str:
        """Strip verbose preludes/role-framing/output-format instructions
        from a user query before handing it to the embedding reranker.
        Long, role-played queries silently collapse reranker scores (verified
        empirically: same chunk drops from 0.94 to 0.03 when the query is
        wrapped in "Answer as if you are an equity research analyst...").
        Cached per-query on the GraphRAG instance to avoid repeating the
        LLM call across multi-turn retrievals.
        """
        original = str(query or "").strip()
        if not original:
            return original
        # Skip the LLM call for short queries (no meaningful prelude to strip).
        if len(original) <= 80:
            return original
        cache = getattr(self, "_simplified_rerank_query_cache", None)
        if cache is None:
            cache = {}
            self._simplified_rerank_query_cache = cache
        if original in cache:
            return cache[original]
        response = await generate_json_or_raise(
            self.llm,
            [
                {"role": "user", "content": RERANK_QUERY_SIMPLIFY_PROMPT.format(query=original)},
                {"role": "user", "content": RERANK_QUERY_SIMPLIFY_FORMAT_INSTRUCTION},
            ],
            "Rerank query simplification",
            f"query={original!r}",
        )
        simplified = str(response.get("question", "") or "").strip() or original
        cache[original] = simplified
        return simplified

    async def _rerank_and_select(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
        query_meta: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not candidates:
            return [], []

        self._apply_retrieval_calibration(candidates, query_meta)
        doc_texts = [node.get("text", "") for node in candidates]
        rerank_query = await self._simplified_rerank_query(query)
        scores = await self._embedding_rerank_scores(rerank_query, doc_texts)

        for index, score in enumerate(scores):
            candidates[index]["rerank_score"] = score
            candidates[index]["final_score"] = (
                score
                + (RAGConfig.META_BOOST_WEIGHT * candidates[index].get("meta_boost", 0.0))
                - (RAGConfig.BOILERPLATE_PENALTY_WEIGHT * candidates[index].get("boilerplate_penalty", 0.0))
            )

        reranked_nodes = sorted(candidates, key=lambda item: item.get("final_score", 0.0), reverse=True)
        company_keys = set(query_meta.get("company_keys") or [])
        if company_keys:
            # Strict filter: when the query is anchored to a company, drop
            # cross-company chunks entirely instead of merely demoting them
            # (a cross-company chunk surviving past top_k previously got
            # cited by the synthesis stage, e.g. AMD content under an AMEX
            # query).
            reranked_nodes = [node for node in reranked_nodes if self._node_matches_company(node, query_meta)]

        final_nodes = [
            node for node in reranked_nodes
            if node.get("rerank_score", 0.0) >= RAGConfig.RERANKER_THRESHOLD
        ][:top_k]
        return final_nodes, reranked_nodes
