"""Retrieve direct evidence and dependency seeds by representation role.

Each enabled representation is searched exactly once with the original
benchmark query. Q- and body hits are direct-evidence candidates; Q+ hits are
dependency seeds whose owner chunks can expose offline ``HOP_ANSWER`` edges.
The representation results form an unweighted set union. Direction is
expressed only by graph role.
"""

import asyncio
from typing import Any

from core.config import RAGConfig


class RetrieveMixin:
    async def retrieve(self, query: str, top_k: int = RAGConfig.DEFAULT_TOP_K) -> tuple:
        selected_nodes, _ = await self._retrieve_with_candidate_pool(query, top_k)
        output_nodes = [self._without_transient_retrieval_scores(node) for node in selected_nodes]
        return self._build_context_from_nodes(output_nodes), output_nodes

    async def _retrieve_with_candidate_pool(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return semantic top-k nodes and the representation-union seed pool."""
        top_k = max(1, int(top_k))
        query_embedding = query_embedding or await self.llm.get_embedding(query)
        if not query_embedding:
            raise ValueError(f"Retrieval received an empty query embedding for query={query!r}")

        variant = RAGConfig.HYPO_CHANNEL_VARIANT
        if variant == "qminus_only":
            channels = ["q_minus"]
        elif variant == "qplus_only":
            channels = ["q_plus"]
        elif variant == "single_combined":
            channels = ["q_minus", "q_plus"]
        else:
            channels = ["body"]
            if RAGConfig.ABLATION_Q_MINUS:
                channels.insert(0, "q_minus")
            if RAGConfig.ABLATION_Q_PLUS:
                channels.append("q_plus")

        searches = await asyncio.gather(
            *[
                self._hybrid_rrf_candidates(
                    query,
                    query_embedding=query_embedding,
                    limit=top_k,
                    channel=channel,
                )
                for channel in channels
            ]
        )

        merged: dict[str, dict[str, Any]] = {}
        for channel, nodes in zip(channels, searches):
            for node in nodes:
                node_id = self._node_identity(node)
                merged.setdefault(node_id, dict(node))
                if channel == "q_plus":
                    merged[node_id]["dependency_seed"] = True
        for node in merged.values():
            node.setdefault("dependency_seed", False)

        # Each representation returns at most top_k owner chunks, so the set
        # union is already structurally bounded.
        base_candidates = list(merged.values())

        final_nodes, _ = await self._score_and_select(query_embedding, base_candidates, top_k)
        return final_nodes, base_candidates

    @staticmethod
    def _without_transient_retrieval_scores(node: dict[str, Any]) -> dict[str, Any]:
        output = dict(node)
        for key in (
            "rrf_score",
            "dependency_seed",
            "embedding",
            "bridge_embeddings",
        ):
            output.pop(key, None)
        return output
