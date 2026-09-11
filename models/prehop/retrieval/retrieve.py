"""Retrieve direct evidence and dependency seeds by representation role.

Each enabled representation is searched exactly once with the original
benchmark query. All hits are direct-evidence candidates and default HOP starts;
Q+ hits additionally mark dependency seeds for the historical qplus policy.
Starting chunks expose the configured ``HOP_ANSWER`` provenance (owner-wide unfiltered provenance by default; reciprocal filtering
and exact matched-Q+ activation remain ablations).
The representation results form an unweighted set union. Direction is
expressed only by graph role.
"""

import asyncio
from typing import Any

from core.config import RAGConfig
from models.prehop.tracing import traced


class RetrieveMixin:
    async def retrieve(self, query: str, top_k: int) -> tuple:
        selected_nodes, _ = await self._retrieve_with_candidate_pool(query, top_k)
        output_nodes = [self._without_transient_retrieval_scores(node) for node in selected_nodes]
        return self._build_context_from_nodes(output_nodes), output_nodes


    @traced
    async def _retrieve_with_candidate_pool(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
        select_final: bool = True,
        selection_variant: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return selected nodes and the representation-union seed pool.

        Graph traversal consumes the complete pool and performs selection only
        after expansion, so it skips the redundant pre-expansion scoring pass.
        """
        top_k = max(1, int(top_k))
        candidate_k = top_k * RAGConfig.CANDIDATE_POOL_MULTIPLIER

        variant = RAGConfig.HYPO_CHANNEL_VARIANT
        active_selection_variant = selection_variant or RAGConfig.SOURCE_SELECTION_VARIANT
        if variant == "body_only":
            channels = ["body"]
        elif variant == "qminus_only":
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

        # Sentence retrieval is a second resolution of the body role, not a
        # fourth evidence role. Fuse it with chunk-body retrieval before the
        # Q-/body/Q+ union so a chunk cannot gain an extra representation vote
        # merely because the same text matched at two granularities.
        search_specs: list[tuple[str, str, str]] = []
        for channel in channels:
            search_specs.append((channel, channel, query))
            if channel == "body" and RAGConfig.SENTENCE_CHANNEL_ENABLED:
                search_specs.append(("body", "sentence", query))
        query_embedding = query_embedding or await self.llm.get_embedding(query)
        if not query_embedding:
            raise ValueError(f"Retrieval received an empty query embedding for query={query!r}")

        searches = await asyncio.gather(
            *[
                self._hybrid_rrf_candidates(
                    view,
                    query_embedding=query_embedding,
                    limit=candidate_k,
                    channel=channel,
                )
                for _role, channel, view in search_specs
            ]
        )

        # Fuse multiple query views inside their role before roles are fused
        # together. Without this boundary, a channel with three rewritten
        # views receives three times the reciprocal-rank mass of the unchanged
        # body channel before the Q-/body/Q+ union. Each role searches the
        # same original query.
        per_channel: dict[str, dict[str, dict[str, Any]]] = {}
        for (role, index_channel, view), nodes in zip(search_specs, searches, strict=True):
            for rank, node in enumerate(nodes):
                node_id = self._node_identity(node)
                candidate = per_channel.setdefault(role, {}).setdefault(node_id, dict(node))
                candidate["view_rank_score"] = float(candidate.get("view_rank_score", 0.0)) + 1.0 / (rank + 1)
                if role == "q_plus":
                    matched_qplus_ids = {
                        str(question_id).strip()
                        for question_id in (candidate.get("matched_qplus_ids") or [])
                        if str(question_id).strip()
                    }
                    matched_qplus_ids.update(
                        str(question_id).strip()
                        for question_id in (node.get("matched_qplus_ids") or [])
                        if str(question_id).strip()
                    )
                    candidate["matched_qplus_ids"] = sorted(matched_qplus_ids)
                elif role == "q_minus":
                    matched_qminus_ids = {
                        str(question_id).strip()
                        for question_id in (candidate.get("matched_qminus_ids") or [])
                        if str(question_id).strip()
                    }
                    matched_qminus_ids.update(
                        str(question_id).strip()
                        for question_id in (node.get("matched_qminus_ids") or [])
                        if str(question_id).strip()
                    )
                    candidate["matched_qminus_ids"] = sorted(matched_qminus_ids)
                paths = candidate.setdefault("retrieval_paths", [])
                direct_path = {
                    "kind": "direct",
                    "channel": index_channel,
                    "query_view": view,
                    "depth": 0,
                }
                if direct_path not in paths:
                    paths.append(direct_path)

        channel_ranked: dict[str, list[dict[str, Any]]] = {}
        for channel, candidates_by_id in per_channel.items():
            ordered = sorted(
                candidates_by_id.values(),
                key=lambda item: (-float(item.get("view_rank_score", 0.0)), self._node_identity(item)),
            )[:candidate_k]
            for rank, candidate in enumerate(ordered):
                candidate["channel_rank_score"] = 1.0 / (rank + 1)
            channel_ranked[channel] = ordered

        merged: dict[str, dict[str, Any]] = {}
        for channel in channels:
            for node in channel_ranked.get(channel, []):
                node_id = self._node_identity(node)
                candidate = merged.setdefault(node_id, dict(node))
                representation_scores = candidate.setdefault("representation_scores", {})
                representation_scores[channel] = float(node["channel_rank_score"])
                candidate["representation_score"] = sum(representation_scores.values())
                if channel == "q_plus":
                    matched_qplus_ids = {
                        str(question_id).strip()
                        for question_id in (candidate.get("matched_qplus_ids") or [])
                        if str(question_id).strip()
                    }
                    matched_qplus_ids.update(
                        str(question_id).strip()
                        for question_id in (node.get("matched_qplus_ids") or [])
                        if str(question_id).strip()
                    )
                    candidate["matched_qplus_ids"] = sorted(matched_qplus_ids)
                    candidate["dependency_seed"] = bool(candidate.get("matched_qplus_ids"))
                elif channel == "q_minus":
                    matched_qminus_ids = {
                        str(question_id).strip()
                        for question_id in (candidate.get("matched_qminus_ids") or [])
                        if str(question_id).strip()
                    }
                    matched_qminus_ids.update(
                        str(question_id).strip()
                        for question_id in (node.get("matched_qminus_ids") or [])
                        if str(question_id).strip()
                    )
                    candidate["matched_qminus_ids"] = sorted(matched_qminus_ids)
                    candidate["continuation_seed"] = bool(candidate.get("matched_qminus_ids"))
                paths = candidate.setdefault("retrieval_paths", [])
                for path in node.get("retrieval_paths") or []:
                    if path not in paths:
                        paths.append(path)
        for node in merged.values():
            node.setdefault("dependency_seed", False)
            node.setdefault("continuation_seed", False)

        # Each representation returns a bounded owner pool. The default
        # multiplier is one; a wider pool remains a query-time ablation and
        # never changes the final evidence count.
        base_candidates = list(merged.values())

        if not select_final:
            return [], base_candidates
        score_kwargs = {"query_text": query} if active_selection_variant == "role_body_list_ranking" else {}
        if selection_variant is not None:
            score_kwargs["selection_variant"] = selection_variant
        final_nodes, _ = await self._score_and_select(
            query_embedding,
            base_candidates,
            top_k,
            **score_kwargs,
        )
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
            "continuation_seed",
            "matched_qplus_ids",
            "matched_qminus_ids",
            "embedding",
            "bridge_embeddings",
            "view_rank_score",
            "channel_rank_score",
            "role_body_owner_orders",
            "role_body_round_entries",
            "role_body_owner_only",
        ):
            output.pop(key, None)
        return output
