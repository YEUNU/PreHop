"""Two-stage Q-/Q+ retrieval entry point (paper §3.2.3).

Stage 1 — grounded retrieval: RRF over Q- (weight 0.7) and body (weight 0.3).
Stage 2 — Q+ expansion: always runs in the full method. It adds Q+ (weight
   0.6) plus a Q- support pool (weight 0.4), then applies cosine ordering.

The original benchmark query is used directly; retrieval has no query-time
generation or heuristic query transformation.
"""

from typing import Any

from core.config import RAGConfig


class RetrieveMixin:
    async def retrieve(self, query: str, top_k: int = RAGConfig.DEFAULT_TOP_K) -> tuple:
        stage1_merged: dict[str, dict[str, Any]] = {}
        candidate_limit_per_query = max(20, top_k * 8)
        rrf_score_keys = ("stage1_rrf_score", "stage2_rrf_score", "stage2_support_score")

        def _accumulate(
            merged: dict[str, dict[str, Any]],
            nodes: list[dict[str, Any]],
            score_key: str,
            weight: float,
        ) -> None:
            self._rrf_accumulate(merged, nodes, score_key, weight, default_keys=rrf_score_keys)

        variant = RAGConfig.HYPO_CHANNEL_VARIANT
        # --- Stage 1: grounded retrieval (Q- 0.7 + body 0.3) per paper §3.2.3 ---
        if variant == "qminus_only":
            q_minus_nodes = await self._hybrid_rrf_candidates(query, limit=candidate_limit_per_query, channel="q_minus")
            _accumulate(stage1_merged, q_minus_nodes, "stage1_rrf_score", 1.0)
        elif variant == "qplus_only":
            q_plus_nodes = await self._hybrid_rrf_candidates(query, limit=candidate_limit_per_query, channel="q_plus")
            _accumulate(stage1_merged, q_plus_nodes, "stage1_rrf_score", 1.0)
        elif variant == "single_combined":
            # HopRAG-style single hypothetical channel: equal-weight Q-/Q+
            # candidates fused through the same RRF accumulator, no direction
            # distinction. Body is excluded to isolate the direction split.
            q_minus_nodes = await self._hybrid_rrf_candidates(query, limit=candidate_limit_per_query, channel="q_minus")
            q_plus_nodes = await self._hybrid_rrf_candidates(query, limit=candidate_limit_per_query, channel="q_plus")
            _accumulate(stage1_merged, q_minus_nodes, "stage1_rrf_score", 0.5)
            _accumulate(stage1_merged, q_plus_nodes, "stage1_rrf_score", 0.5)
        elif RAGConfig.ABLATION_Q_MINUS:
            q_minus_nodes = await self._hybrid_rrf_candidates(query, limit=candidate_limit_per_query, channel="q_minus")
            body_nodes = await self._hybrid_rrf_candidates(query, limit=max(10, top_k * 4), channel="body")
            _accumulate(stage1_merged, q_minus_nodes, "stage1_rrf_score", 0.7)
            _accumulate(stage1_merged, body_nodes, "stage1_rrf_score", 0.3)
        else:
            body_nodes = await self._hybrid_rrf_candidates(query, limit=candidate_limit_per_query, channel="body")
            _accumulate(stage1_merged, body_nodes, "stage1_rrf_score", 1.0)

        stage1_candidates = sorted(
            stage1_merged.values(),
            key=lambda item: item.get("stage1_rrf_score", 0.0),
            reverse=True,
        )[: max(20, top_k * 6)]

        stage1_nodes, _ = await self._score_and_select(query, stage1_candidates, top_k)
        use_q_plus_stage = RAGConfig.ABLATION_Q_PLUS and variant == "full"

        if not use_q_plus_stage:
            for node in stage1_nodes:
                node.pop("stage1_rrf_score", None)
                node.pop("stage2_rrf_score", None)
                node.pop("stage2_support_score", None)
            return self._build_context_from_nodes(stage1_nodes), stage1_nodes

        # --- Stage 2: Q+ expansion (Q+ 0.6 + Q- support 0.4) per paper §3.2.3 ---
        expanded: dict[str, dict[str, Any]] = {self._node_identity(node): dict(node) for node in stage1_candidates}
        q_plus_weight = 0.6
        q_minus_support_weight = 0.4
        q_plus_nodes = await self._hybrid_rrf_candidates(query, limit=candidate_limit_per_query, channel="q_plus")
        q_minus_support_nodes = await self._hybrid_rrf_candidates(query, limit=max(10, top_k * 4), channel="q_minus")
        _accumulate(expanded, q_plus_nodes, "stage2_rrf_score", q_plus_weight)
        _accumulate(expanded, q_minus_support_nodes, "stage2_support_score", q_minus_support_weight)

        if not expanded:
            return "", []

        for node in expanded.values():
            node["hybrid_rrf_score"] = (
                node.get("stage1_rrf_score", 0.0)
                + node.get("stage2_rrf_score", 0.0)
                + node.get("stage2_support_score", 0.0)
            )

        expanded_candidates = sorted(
            expanded.values(),
            key=lambda item: item.get("hybrid_rrf_score", 0.0),
            reverse=True,
        )[: max(24, top_k * 8)]

        final_nodes, _ = await self._score_and_select(query, expanded_candidates, top_k)
        if not final_nodes:
            return "", []

        for node in final_nodes:
            node.pop("stage1_rrf_score", None)
            node.pop("stage2_rrf_score", None)
            node.pop("stage2_support_score", None)
            node.pop("hybrid_rrf_score", None)
        return self._build_context_from_nodes(final_nodes), final_nodes
