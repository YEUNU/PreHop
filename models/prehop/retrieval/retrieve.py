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
    async def retrieve(self, query: str, top_k: int) -> tuple:
        selected_nodes, _ = await self._retrieve_with_candidate_pool(query, top_k)
        output_nodes = [self._without_transient_retrieval_scores(node) for node in selected_nodes]
        return self._build_context_from_nodes(output_nodes), output_nodes

    async def retrieve_with_views(
        self,
        query: str,
        top_k: int,
        channel_queries: dict[str, list[str]],
    ) -> tuple:
        selected_nodes, _ = await self._retrieve_with_candidate_pool(
            query,
            top_k,
            channel_queries=channel_queries,
        )
        output_nodes = [self._without_transient_retrieval_scores(node) for node in selected_nodes]
        return self._build_context_from_nodes(output_nodes), output_nodes

    async def _retrieve_with_candidate_pool(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
        channel_queries: dict[str, list[str]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return semantic top-k nodes and the representation-union seed pool."""
        top_k = max(1, int(top_k))

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

        search_specs: list[tuple[str, str]] = []
        for channel in channels:
            views = (channel_queries or {}).get(channel) or [query]
            search_specs.extend((channel, view) for view in views)
        if channel_queries:
            texts_to_embed = list(
                dict.fromkeys(
                    [*([] if query_embedding else [query]), *(view for _channel, view in search_specs)]
                )
            )
            embeddings = await self.llm.get_embeddings(texts_to_embed, encoding_type="query")
            if len(embeddings) != len(texts_to_embed) or any(not embedding for embedding in embeddings):
                raise ValueError("Role-aligned retrieval received an invalid query-embedding batch")
            embedding_by_text = dict(zip(texts_to_embed, embeddings))
            query_embedding = query_embedding or embedding_by_text[query]
        else:
            query_embedding = query_embedding or await self.llm.get_embedding(query)
            embedding_by_text = {query: query_embedding}
        if not query_embedding:
            raise ValueError(f"Retrieval received an empty query embedding for query={query!r}")

        searches = await asyncio.gather(
            *[
                self._hybrid_rrf_candidates(
                    view,
                    query_embedding=embedding_by_text[view],
                    limit=top_k,
                    channel=channel,
                )
                for channel, view in search_specs
            ]
        )

        merged: dict[str, dict[str, Any]] = {}
        for (channel, view), nodes in zip(search_specs, searches):
            for rank, node in enumerate(nodes):
                node_id = self._node_identity(node)
                candidate = merged.setdefault(node_id, dict(node))
                # The rank returned by a representation already fuses its
                # vector and lexical modalities. Preserve that evidence when
                # representation lists are merged instead of discarding it
                # during the later body-only ordering. Reciprocal rank has no
                # fitted scale or dataset-dependent threshold.
                representation_scores = candidate.setdefault("representation_scores", {})
                representation_scores[channel] = float(representation_scores.get(channel, 0.0)) + 1.0 / (
                    rank + 1
                )
                candidate["representation_score"] = sum(representation_scores.values())
                paths = candidate.setdefault("retrieval_paths", [])
                direct_path = {"kind": "direct", "channel": channel, "query_view": view, "depth": 0}
                if direct_path not in paths:
                    paths.append(direct_path)
                if channel == "q_plus":
                    candidate["dependency_seed"] = True
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
            "representation_score",
            "representation_scores",
            "rank_fusion_score",
            "dependency_seed",
            "embedding",
            "bridge_embeddings",
        ):
            output.pop(key, None)
        return output
