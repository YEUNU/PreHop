"""Graph traversal over pre-built NEXT/HOP edges (paper §3.2.3 "Graph Traversal").

Starting from reranked seed nodes, the system expands along NEXT (sequential)
and HOP (semantic, pre-built §3.1.4) edges. At each hop a continuation
decision asks the LLM whether the accumulated context is sufficient; if yes
the traversal stops early.

Two HOP modes:
- "offline" (paper canonical): traverse the pre-built [:NEXT|HOP] edges
- "runtime" (HopRAG-style fallback): follow only [:NEXT] in the graph but
  expand the frontier at query time via Q+ ANN + embedding-similarity rerank.
"""
import logging
import time
from typing import Any, Optional

from core.config import RAGConfig
from models.prehop.llm_json import generate_json_or_raise
from utils.prompts import SEARCH_CONTINUATION_FORMAT_INSTRUCTION


logger = logging.getLogger(__name__)


class TraversalMixin:
    async def graph_search(self, entities: list[str], depth: int = 2, top_k: int = 5, user_query: str = "", excluded_chunk_ids: Optional[set[str]] = None, force_expand: bool = False) -> tuple:
        """Graph-traversal retrieval (paper §3.2.3).

        When `force_expand=True`, every depth hop runs deterministically:
        the LLM `_need_more_for_next_depth` continuation check is skipped.
        This is used by the agentic-OFF baseline path so retrieval contains
        no agentic decision-making (the only LLM call left is the query
        simplification grader; reranking itself is now embedding
        cosine-similarity, not an LLM call).

        Returns (context, nodes, timing) where timing = {"retrieve_ms":
        <initial hybrid retrieve>, "traversal_ms": <everything after that —
        seed gating + NEXT/HOP frontier expansion>}. Split out so the paper's
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

        seed_top_k = max(1, min(max(1, top_k - 1), RAGConfig.GRAPH_SEARCH_LIMIT))
        t_retrieve0 = time.perf_counter()
        _, seed_nodes = await self.retrieve(seed_query, top_k=seed_top_k, user_query=user_query)
        retrieve_ms = (time.perf_counter() - t_retrieve0) * 1000

        def _timing() -> dict[str, float]:
            total_ms = (time.perf_counter() - t0) * 1000
            return {"retrieve_ms": retrieve_ms, "traversal_ms": max(0.0, total_ms - retrieve_ms)}
        # Filter out chunks already retrieved on prior turns of this query so
        # NEXT/HOP graph traversal explores fresh territory rather than
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
        # Company-key extraction must come from the human-written query when
        # available — joining LLM-generated `entities` produces synthetic
        # phrases like "amd fy21 income statement of operations consolidated
        # statements of operations" whose normalized form falsely registers
        # as a company key and empties the strict company filter pool. The
        # rerank/finalize stage still uses the synthetic search_query for
        # reranker scoring (where the broader phrasing is useful); only the
        # company-anchor metadata is sourced from user_query.
        meta_query = user_query.strip() if user_query and user_query.strip() else search_query
        search_query_meta = self._extract_query_metadata(meta_query)
        step_limit = max(RAGConfig.GRAPH_SEARCH_LIMIT, top_k * 6)
        frontier_ids = [seed_id for seed_id in seed_ids if seed_id]
        # Track within-call traversal history AND prior-turn exclusions so
        # neither this call's expansion nor the final result returns chunks
        # already surfaced by previous calls.
        visited_ids = set(frontier_ids) | excluded_ids
        collected: dict[str, dict[str, Any]] = {}

        async def _need_more_for_next_depth(nodes_for_judge: list[dict[str, Any]]) -> bool:
            if not nodes_for_judge:
                return True
            ranked = sorted(
                nodes_for_judge,
                key=lambda item: item.get("rerank_score", 0.0),
                reverse=True,
            )[: max(top_k, 6)]
            context_preview = "\n\n".join([
                f"[[{node.get('title', 'Unknown')}, Page {node.get('page', 0)}, Chunk {node.get('sent_id', -1)}]]\n"
                f"{str(node.get('text', '') or '')[:450]}"
                for node in ranked
            ])
            messages = [
                {"role": "user", "content": self._search_continuation_prompt().format(query=search_query, context=context_preview)},
                {"role": "user", "content": SEARCH_CONTINUATION_FORMAT_INSTRUCTION},
            ]
            decision_data = await generate_json_or_raise(
                self.llm, messages, "Search-continuation decision", f"query={search_query!r}",
                apply_default_sampling=False,
            )
            decision = str(decision_data.get("decision", "INSUFFICIENT")).strip().upper()
            need_more = decision != "SUFFICIENT"
            logger.info("Graph depth continuation decision=%s (need_more=%s)", decision, need_more)
            return need_more

        seed_gated, _ = await self._rerank_and_select(search_query, seed_nodes, top_k, search_query_meta)
        frontier_ids = []
        for node in seed_gated:
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            frontier_ids.append(node_id)
            previous = collected.get(node_id)
            if (previous is None) or (
                node.get("final_score", node.get("rerank_score", 0.0))
                > previous.get("final_score", previous.get("rerank_score", 0.0))
            ):
                collected[node_id] = node

        if collected and not force_expand:
            need_more = await _need_more_for_next_depth(list(collected.values()))
            if not need_more:
                nodes = sorted(
                    collected.values(),
                    key=lambda item: item.get("final_score", item.get("rerank_score", 0.0)),
                    reverse=True,
                )[:top_k]
                return self._build_context_from_nodes(nodes), nodes, _timing()

        for hop_index in range(depth):
            if not frontier_ids:
                break

            edge_pattern = "[:NEXT]" if RAGConfig.HOP_MODE == "runtime" else "[:NEXT|HOP]"
            async with self.neo4j.driver.session() as session:
                query = f"""
                    UNWIND $frontier_ids AS src_id
                    MATCH (src:{self.chunk_label} {{id: src_id}})-{edge_pattern}->(related:{self.chunk_label})
                    WHERE NOT related.id IN $visited_ids
                    RETURN DISTINCT related.id as id, related.title as title, related.sent_id as sent_id,
                                    related.page as page, related.text as text, related.source as source
                    LIMIT $limit
                """
                result = await session.run(query, {  # type: ignore
                    "frontier_ids": frontier_ids,
                    "visited_ids": list(visited_ids),
                    "limit": step_limit,
                })
                candidates = [dict(record) async for record in result]

            # Runtime HOP supplement: enabled only when HOP_MODE == "runtime".
            # In offline mode (default), the Cypher pattern above already walks
            # the pre-built [:HOP] edges from paper §3.1.4 — adding live Q+
            # ANN on top would double-source HOP candidates and re-introduce
            # chunks the offline rerank filter (tau_r=0.5, L_hop=5) was
            # designed to exclude. Prior unconditional invocation here caused
            # ~3-4 seed chunks per query to be displaced from the final
            # top_k slice by runtime ANN noise.
            if RAGConfig.HOP_MODE == "runtime":
                runtime_cands = await self._runtime_hop_candidates(
                    frontier_ids=frontier_ids,
                    visited_ids=visited_ids,
                    step_limit=step_limit,
                )
                seen_ids = {str(node.get("id", "")) for node in candidates}
                for cand in runtime_cands:
                    cand_id = str(cand.get("id", ""))
                    if cand_id and cand_id not in seen_ids:
                        candidates.append(cand)
                        seen_ids.add(cand_id)

            if not candidates:
                break

            gated_nodes, _ = await self._rerank_and_select(search_query, candidates, top_k, search_query_meta)
            if not gated_nodes:
                break

            next_frontier: list[str] = []
            for node in gated_nodes:
                node_id = str(node.get("id", "")).strip()
                if not node_id:
                    continue
                visited_ids.add(node_id)
                next_frontier.append(node_id)
                previous = collected.get(node_id)
                if (previous is None) or (
                    node.get("final_score", node.get("rerank_score", 0.0))
                    > previous.get("final_score", previous.get("rerank_score", 0.0))
                ):
                    collected[node_id] = node

            frontier_ids = next_frontier[:step_limit]
            if hop_index < depth - 1 and collected and not force_expand:
                need_more = await _need_more_for_next_depth(list(collected.values()))
                if not need_more:
                    break

        nodes = sorted(
            collected.values(),
            key=lambda item: item.get("final_score", item.get("rerank_score", 0.0)),
            reverse=True,
        )[:top_k]
        if not nodes:
            return "", [], _timing()
        return self._build_context_from_nodes(nodes), nodes, _timing()

    async def _runtime_hop_candidates(
        self,
        frontier_ids: list[str],
        visited_ids: set,
        step_limit: int,
    ) -> list[dict[str, Any]]:
        """Runtime mirror of offline HOP construction: for each frontier node,
        ANN-search the q_plus index with the node's own q_plus embedding.

        This is the dynamic (HopRAG-style) counterpart to pre-built HOP edges.
        Used only when RAGConfig.HOP_MODE == "runtime".
        """

        if not frontier_ids:
            return []

        async with self.neo4j.driver.session() as session:
            fetch_query = f"""
                UNWIND $frontier_ids AS fid
                MATCH (n:{self.chunk_label} {{id: fid}})
                WHERE n.q_plus_embedding IS NOT NULL
                RETURN n.id as src_id, n.q_plus_embedding as embed, n.source as src_source
            """
            fetch_res = await session.run(fetch_query, {"frontier_ids": frontier_ids})  # type: ignore
            sources = [dict(record) async for record in fetch_res]

        if not sources:
            return []

        per_source_k = max(1, min(10, step_limit))
        candidates_by_id: dict[str, dict[str, Any]] = {}
        for source in sources:
            ann_query = f"""
                CALL db.index.vector.queryNodes('{self.q_plus_vector_index}', $k, $embed)
                YIELD node, score
                WHERE node.id <> $src_id
                  AND node.source <> $src_source
                  AND NOT node.id IN $visited
                RETURN node.id as id, node.title as title, node.sent_id as sent_id,
                       node.page as page, node.text as text, node.source as source, score
            """
            rows = await self.retry_query(
                ann_query,
                {
                    "k": per_source_k,
                    "embed": source["embed"],
                    "src_id": source["src_id"],
                    "src_source": source["src_source"],
                    "visited": list(visited_ids),
                },
            )
            for row in rows or []:
                cid = str(row.get("id") or "")
                if not cid or cid in candidates_by_id:
                    continue
                candidates_by_id[cid] = dict(row)

        ranked = sorted(
            candidates_by_id.values(),
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )
        return ranked[:step_limit]
