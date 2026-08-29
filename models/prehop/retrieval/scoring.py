"""Parameter-free fusion of indexed representation and body semantics."""

from collections import defaultdict, deque
from typing import Any

from core.config import RAGConfig
from utils.similarity import cosine_similarity


class SimilarityScoringMixin:
    @staticmethod
    def _document_identity(node: dict[str, Any]) -> str:
        """Return the logical document identity used for evidence diversity.

        Some prepared corpora store one paragraph per file, so ``source`` is
        a chunk container rather than a document boundary. The indexed title
        is the shared, dataset-neutral document identity in that case. Keep
        the filename as a fallback for sources without a title.
        """
        return str(node.get("title") or node.get("doc") or node.get("source") or "")

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

        The best matching individual source Q+ represents a traversed HOP
        bridge. The conservative default takes its minimum with direct body
        relevance. A query-time ablation uses the bridge alone because the
        offline Q+->Q- edge already selected the target body. Equal reciprocal
        ranks then combine the resulting semantic order with the Q-/body/Q+
        retrieval order. Neither path compares backend-specific raw scores or
        introduces a fitted interpolation weight or threshold.
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
            if bridge_score is None:
                final_score = body_score
            elif RAGConfig.HOP_SEMANTIC_VARIANT == "bridge_only":
                final_score = bridge_score
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
        if RAGConfig.SOURCE_SELECTION_VARIANT == "global":
            selected = ordered[:top_k]
        elif RAGConfig.SOURCE_SELECTION_VARIANT == "round_robin":
            selected = self._source_round_robin(ordered, top_k)
        elif RAGConfig.SOURCE_SELECTION_VARIANT == "source_balanced":
            selected = self._source_balanced(ordered, top_k)
        elif RAGConfig.SOURCE_SELECTION_VARIANT == "graph_pairs":
            selected = self._graph_pairs(ordered, top_k)
        elif RAGConfig.SOURCE_SELECTION_VARIANT == "source_balanced_graph_pairs":
            selected = self._graph_pairs(self._source_balanced_order(ordered), top_k)
        else:
            raise ValueError(
                f"Unsupported source selection variant: {RAGConfig.SOURCE_SELECTION_VARIANT!r}"
            )
        return selected, ordered

    @staticmethod
    def _source_round_robin(ordered: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Take one ranked chunk per source per round, then repeat."""
        groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        source_order: list[str] = []
        for node in ordered:
            source = SimilarityScoringMixin._document_identity(node)
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

    @staticmethod
    def _source_balanced(ordered: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Discount repeated-source candidates by their selected occurrence."""
        return SimilarityScoringMixin._source_balanced_order(ordered)[:top_k]

    @staticmethod
    def _source_balanced_order(ordered: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return every candidate in soft document-diversified order."""
        remaining = list(ordered)
        source_counts: dict[str, int] = defaultdict(int)
        selected: list[dict[str, Any]] = []
        while remaining:
            best_index = max(
                range(len(remaining)),
                key=lambda index: (
                    float(remaining[index].get("rank_fusion_score", 0.0))
                    / (source_counts[SimilarityScoringMixin._document_identity(remaining[index])] + 1),
                    float(remaining[index].get("final_score", 0.0)),
                    -index,
                ),
            )
            node = remaining.pop(best_index)
            source = SimilarityScoringMixin._document_identity(node)
            selected.append(node)
            source_counts[source] += 1
        return selected

    @classmethod
    def _graph_pairs(cls, ordered: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Keep a high-ranked HOP target with its best-ranked source owner."""
        by_id = {cls._node_identity(node): node for node in ordered}
        rank = {cls._node_identity(node): index for index, node in enumerate(ordered)}
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()

        def append(node: dict[str, Any]) -> None:
            node_id = cls._node_identity(node)
            if node_id and node_id not in selected_ids and len(selected) < top_k:
                selected.append(node)
                selected_ids.add(node_id)

        for node in ordered:
            if len(selected) >= top_k:
                break
            hop_sources = {
                str(path.get("source_chunk_id") or "")
                for path in (node.get("retrieval_paths") or [])
                if path.get("kind") == "hop" and str(path.get("source_chunk_id") or "") in by_id
            }
            if not hop_sources or len(selected) == top_k - 1:
                append(node)
                continue
            best_source_id = min(hop_sources, key=lambda source_id: rank[source_id])
            pair = sorted((node, by_id[best_source_id]), key=lambda item: rank[cls._node_identity(item)])
            for pair_node in pair:
                append(pair_node)
        return selected
