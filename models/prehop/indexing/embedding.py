"""Strict sparse-text embedding helper for offline indexing."""


class SparseEmbeddingMixin:
    async def _embed_sparse_texts(
        self,
        texts: list[str],
        *,
        encoding_type: str = "document",
    ) -> list[list[float]]:
        if not texts:
            return []
        positions: list[int] = []
        payload: list[str] = []
        for index, text in enumerate(texts):
            normalized = str(text or "").strip()
            if not normalized:
                continue
            positions.append(index)
            payload.append(normalized)

        result: list[list[float]] = [[] for _ in texts]
        if not payload:
            return result

        embeddings = await self.llm.get_embeddings(payload, encoding_type=encoding_type)
        if len(embeddings) != len(payload) or any(not embedding for embedding in embeddings):
            raise ValueError(
                f"Sparse-text embedding failure: expected {len(payload)} non-empty vectors, got {len(embeddings)}"
            )
        dimensions = {len(embedding) for embedding in embeddings}
        if len(dimensions) != 1:
            raise ValueError("Sparse-text embeddings have inconsistent dimensions")
        for embedding, src_index in zip(embeddings, positions):
            result[src_index] = embedding
        return result
