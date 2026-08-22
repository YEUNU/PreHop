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

        scores = await self._embedding_similarity_scores(
            query,
            [str(node.get("text", "")) for node in candidates],
        )
        for index, score in enumerate(scores):
            candidates[index]["similarity_score"] = score
            candidates[index]["final_score"] = score

        ordered = sorted(candidates, key=lambda item: item.get("final_score", 0.0), reverse=True)
        return ordered[:top_k], ordered
