"""Graph traversal over pre-built NEXT/HOP_ANSWER edges (paper §3.2.3).

Starting from similarity-ordered seed nodes, the system expands deterministically along
NEXT (sequential) and HOP_ANSWER (semantic, pre-built §3.1.4) edges. NEXT is
read in both directions so a seed can recover preceding or following local
context. HOP_ANSWER is read only in its Q+ -> answer-evidence direction.

The query path traverses only the NEXT/HOP_ANSWER edges built during indexing.
"""

import time
from typing import Any

from core.config import RAGConfig


class TraversalMixin:
    async def graph_search(
        self,
        entities: list[str],
        depth: int = 2,
        top_k: int = 5,
        excluded_chunk_ids: set[str] | None = None,
    ) -> tuple:
        """Graph-traversal retrieval (paper §3.2.3).

        Every depth hop runs deterministically. Retrieval contains no
        query-time generation call; ordering uses embedding cosine similarity.

        Returns (context, nodes, timing) where timing = {"retrieve_ms":
        <initial hybrid retrieve>, "traversal_ms": <everything after that —
        seed selection + NEXT/HOP_ANSWER frontier expansion>}. Split out so the paper's
        headline latency claim (deterministic traversal, no per-hop LLM
        reasoning) can be reported as a stage breakdown, not just one total.
        """
        t0 = time.perf_counter()
        normalized_entities: list[str] = []
        for entity in entities:
            normalized = self._normalize_entity_term(entity)
            if normalized:
                normalized_entities.append(normalized)
        seed_query = " ".join(normalized_entities).strip() or " ".join(entities).strip()
        if not seed_query:
            return "", [], {"retrieve_ms": 0.0, "traversal_ms": 0.0}

        depth = max(1, min(int(depth), 4))

        excluded_ids: set[str] = {str(eid).strip() for eid in (excluded_chunk_ids or set()) if str(eid).strip()}

        # Start from the same top-k budget as depth=0. The previous top_k-1
        # plus GRAPH_SEARCH_LIMIT=10 silently dropped two flat-retrieval
        # candidates before traversal when the paper budget was 12, making
        # depth=1 incomparable and able to hurt even when no useful edge was
        # found. Neighbors now compete with, rather than pre-empt, the same
        # complete seed set under the final shared similarity score.
        seed_top_k = max(1, top_k)
        t_retrieve0 = time.perf_counter()
        _, seed_nodes = await self.retrieve(seed_query, top_k=seed_top_k)
        retrieve_ms = (time.perf_counter() - t_retrieve0) * 1000

        def _timing() -> dict[str, float]:
            total_ms = (time.perf_counter() - t0) * 1000
            return {"retrieve_ms": retrieve_ms, "traversal_ms": max(0.0, total_ms - retrieve_ms)}

        # Filter out chunks already retrieved on prior turns of this query so
        # NEXT/HOP_ANSWER graph traversal explores fresh territory rather than
        # re-surfacing the same hub chunks via different seed paths.
        if excluded_ids:
            seed_nodes = [n for n in seed_nodes if str(n.get("id", "")).strip() not in excluded_ids]
        seed_ids = [
            str(node.get("id")).strip()
            for node in seed_nodes
            if node.get("id") is not None and str(node.get("id")).strip()
        ]

        if not seed_ids:
            return "", [], _timing()
        search_query = " ".join(entities).strip() or " ".join(normalized_entities)
        # A chunk has at most two sequential neighbors and HOP_LINK_LIMIT
        # outgoing answer edges. This bound retains the complete first-hop
        # neighborhood of every selected seed instead of applying an
        # arbitrary, non-deterministic LIMIT before semantic scoring.
        step_limit = max(RAGConfig.GRAPH_SEARCH_LIMIT, top_k * (RAGConfig.HOP_LINK_LIMIT + 2))
        frontier_ids = [seed_id for seed_id in seed_ids if seed_id]
        # Track within-call traversal history AND prior-turn exclusions so
        # neither this call's expansion nor the final result returns chunks
        # already surfaced by previous calls.
        visited_ids = set(frontier_ids) | excluded_ids
        collected: dict[str, dict[str, Any]] = {}

        seed_selected, _ = await self._score_and_select(search_query, seed_nodes, top_k)
        frontier_ids = []
        for node in seed_selected:
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            frontier_ids.append(node_id)
            previous = collected.get(node_id)
            if (previous is None) or (
                node.get("final_score", node.get("similarity_score", 0.0))
                > previous.get("final_score", previous.get("similarity_score", 0.0))
            ):
                collected[node_id] = node

        for _ in range(depth):
            if not frontier_ids:
                break

            async with self.neo4j.driver.session() as session:
                query = f"""
                    UNWIND $frontier_ids AS src_id
                    MATCH (src:{self.chunk_label} {{id: src_id}})
                    CALL (src) {{
                        MATCH (src)-[:NEXT]-(related:{self.chunk_label})
                        RETURN related, [] AS bridge_questions
                        UNION ALL
                        MATCH (src)-[hop:HOP_ANSWER]->(related:{self.chunk_label})
                        RETURN related, coalesce(hop.source_question_texts, []) AS bridge_questions
                    }}
                    WITH related, collect(DISTINCT bridge_questions) AS bridge_question_groups
                    WHERE NOT related.id IN $visited_ids
                    RETURN DISTINCT related.id as id, related.title as title, related.sent_id as sent_id,
                                    related.page as page, related.text as text, related.source as source,
                                    bridge_question_groups
                    ORDER BY id
                    LIMIT $limit
                """
                result = await session.run(
                    query,
                    {  # type: ignore
                        "frontier_ids": frontier_ids,
                        "visited_ids": list(visited_ids),
                        "limit": step_limit,
                    },
                )
                candidates = [dict(record) async for record in result]

            for candidate in candidates:
                bridge_questions = list(
                    dict.fromkeys(
                        str(question).strip()
                        for group in candidate.pop("bridge_question_groups", [])
                        for question in group
                        if str(question).strip()
                    )
                )
                if bridge_questions:
                    candidate["bridge_text"] = "Bridge questions: " + " ".join(bridge_questions)

            if not candidates:
                break

            selected_nodes, _ = await self._score_and_select(search_query, candidates, top_k)
            if not selected_nodes:
                break

            next_frontier: list[str] = []
            for node in selected_nodes:
                node_id = str(node.get("id", "")).strip()
                if not node_id:
                    continue
                visited_ids.add(node_id)
                next_frontier.append(node_id)
                previous = collected.get(node_id)
                if (previous is None) or (
                    node.get("final_score", node.get("similarity_score", 0.0))
                    > previous.get("final_score", previous.get("similarity_score", 0.0))
                ):
                    collected[node_id] = node

            frontier_ids = next_frontier[:step_limit]
        nodes = sorted(
            collected.values(),
            key=lambda item: item.get("final_score", item.get("similarity_score", 0.0)),
            reverse=True,
        )[:top_k]
        if not nodes:
            return "", [], _timing()
        return self._build_context_from_nodes(nodes), nodes, _timing()
