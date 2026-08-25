"""Parameter-free fusion of indexed representation and body semantics."""

from collections import defaultdict, deque
from typing import Any

from core.config import RAGConfig
from utils.similarity import cosine_similarity


class SimilarityScoringMixin:
    @staticmethod
    def _validated_similarity(query_embedding: list[float], document_embedding: Any) -> float:
        if not isinstance(document_embedding, list) or not document_embedding:
            raise ValueError("Retrieved candidate is missing its indexed embedding")
        if len(document_embedding) != len(query_embedding):
            raise ValueError("Query and indexed candidate embedding dimensions do not match")
        return cosine_similarity(query_embedding, document_embedding)

    async def _score_and_select(
        self,
        query_embedding: list[float],
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Fuse representation ranks with body and stored-Q+ semantics.

        A traversed HOP target must satisfy both sides of its offline path.
        The best matching individual source Q+ represents the bridge, and the
        conservative minimum with body relevance requires neither side to be
        rescued by a tuned interpolation weight. Equal reciprocal ranks then
        combine that semantic order with the Q-/body/Q+ retrieval order. This
        avoids comparing backend-specific raw scores and introduces no fitted
        weight or threshold.
        """
        if not candidates:
            return [], []
        if not query_embedding:
            raise ValueError("Final scoring received an empty query embedding")

        for candidate in candidates:
            body_score = self._validated_similarity(query_embedding, candidate.get("embedding"))
            bridge_embeddings = candidate.get("bridge_embeddings") or []
            bridge_scores = [
                self._validated_similarity(query_embedding, embedding)
                for embedding in bridge_embeddings
                if embedding
            ]
            bridge_score = max(bridge_scores) if bridge_scores else None
            final_score = min(body_score, bridge_score) if bridge_score is not None else body_score
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
        semantic_ranks = {
            self._node_identity(candidate): rank for rank, candidate in enumerate(semantic_order)
        }
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

        ordered = sorted(
            candidates,
            key=lambda item: (
                float(item.get("rank_fusion_score", 0.0)),
                float(item.get("final_score", 0.0)),
                self._node_identity(item),
            ),
            reverse=True,
        )
        selected = (
            ordered[:top_k]
            if RAGConfig.SOURCE_SELECTION_VARIANT == "global"
            else self._source_round_robin(ordered, top_k)
        )
        return selected, ordered

    @staticmethod
    def _source_round_robin(ordered: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Take one ranked chunk per source per round, then repeat."""
        groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        source_order: list[str] = []
        for node in ordered:
            source = str(node.get("source") or node.get("title") or "")
            if source not in groups:
                source_order.append(source)
            groups[source].append(node)

        selected: list[dict[str, Any]] = []
        while len(selected) < top_k:
            progressed = False
            for source in source_order:
                if groups[source] and len(selected) < top_k:
                    selected.append(groups[source].popleft())
                    progressed = True
            if not progressed:
                break
        return selected
