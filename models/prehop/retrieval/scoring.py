"""External bi-encoder cosine ordering for retrieved candidates.

This is not a separately served reranker and has no score gate. The configured
external embedding endpoint encodes the query and candidate bodies; cosine
similarity orders all candidates and the first ``top_k`` are returned.
"""

from typing import Any

from utils.similarity import cosine_similarity


class SimilarityScoringMixin:
    async def _embedding_similarity_scores(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        query_embed, doc_embeds = await self._get_query_and_doc_embeddings(query, texts)
        return [cosine_similarity(query_embed, doc_embed) for doc_embed in doc_embeds]

    async def _get_query_and_doc_embeddings(self, query: str, texts: list[str]):
        query_embeds = await self.llm.get_embeddings([query], encoding_type="query")
        doc_embeds = await self.llm.get_embeddings(texts, encoding_type="document")
        if len(query_embeds) != 1 or not query_embeds[0]:
            raise ValueError("Similarity query embedding is missing")
        if len(doc_embeds) != len(texts):
            raise ValueError(
                f"Similarity document embedding count mismatch: expected {len(texts)}, got {len(doc_embeds)}"
            )
        if any(not embedding for embedding in doc_embeds):
            raise ValueError("Similarity document embeddings contain an empty vector")
        query_dim = len(query_embeds[0])
        if any(len(embedding) != query_dim for embedding in doc_embeds):
            raise ValueError("Similarity query/document embedding dimensions do not match")
        return query_embeds[0], doc_embeds

    async def _score_and_select(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not candidates:
            return [], []

        body_scores = await self._embedding_similarity_scores(
            query,
            [
                f"Document title: {node.get('title', '')}\n{node.get('text', '')}".strip()
                for node in candidates
            ],
        )
        bridge_indices = [
            index for index, node in enumerate(candidates) if str(node.get("bridge_text") or "").strip()
        ]
        bridge_scores: list[float] = []
        if bridge_indices:
            bridge_scores = await self._embedding_similarity_scores(
                query,
                [str(candidates[index]["bridge_text"]).strip() for index in bridge_indices],
            )
        bridge_by_index = dict(zip(bridge_indices, bridge_scores))

        for index, body_score in enumerate(body_scores):
            # A traversed target should agree on both sides of the stored
            # evidence path: the source Q+ must match the user need and the
            # target body must itself remain relevant. A single concatenated
            # embedding let a strong bridge phrase mask an unrelated target.
            bridge_score = bridge_by_index.get(index)
            final_score = (
                (body_score + bridge_score) / 2.0 if bridge_score is not None else body_score
            )
            candidates[index]["similarity_score"] = body_score
            if bridge_score is not None:
                candidates[index]["bridge_similarity_score"] = bridge_score
            candidates[index]["final_score"] = final_score

        ordered = sorted(candidates, key=lambda item: item.get("final_score", 0.0), reverse=True)
        return ordered[:top_k], ordered
