"""One-step expansion of every direct start over stored HOP and NEXT edges.

All structurally bounded candidates reach the shared scoring and selection path.
"""

import os
import time
from collections import defaultdict
from typing import Any

from core.config import RAGConfig
from models.prehop.tracing import traced


class TraversalMixin:
    def _trace_link_usefulness(self, direct, expanded, selected, starts=()):
        if not RAGConfig.PREHOP_ABLATION_PROFILE or getattr(self, "trace_recorder", None) is None:
            return
        from models.prehop.tracing import _IDENTITY
        def evidence(node):
            return {k: node.get(k) for k in ("id", "source", "title", "text", "paragraph_id", "path_type", "source_id")}
        self.trace_recorder.emit("ablation_link_usefulness", {
            "direct": [evidence(n) for n in direct],
            "expanded": [evidence(n) for n in expanded],
            "selected": [evidence(n) for n in selected],
            "starts": sorted(set(starts)),
        }, identity=dict(_IDENTITY.get() or {}))

    @traced
    async def graph_search(
        self,
        entities: list[str],
        depth: int,
        top_k: int,
        excluded_chunk_ids: set[str] | None = None,
        selection_variant: str | None = None,
    ) -> tuple:
        """Retrieve evidence through level-batched, duplicate-free graph expansion."""
        t0 = time.perf_counter()
        normalized_entities = [normalized for entity in entities if (normalized := self._normalize_entity_term(entity))]
        seed_query = " ".join(normalized_entities).strip() or " ".join(entities).strip()
        if not seed_query:
            self._trace_link_usefulness([], [], [])
            return "", [], {"retrieve_ms": 0.0, "traversal_ms": 0.0}

        if int(depth) != 1:
            raise ValueError("Graph traversal depth must be exactly one")
        excluded_ids = {str(chunk_id).strip() for chunk_id in (excluded_chunk_ids or set()) if str(chunk_id).strip()}

        t_retrieve0 = time.perf_counter()
        frozen_inputs = os.environ.get("RAG_ABLATION_DIRECT_INPUTS", "") if RAGConfig.PREHOP_ABLATION_PROFILE else ""
        if frozen_inputs:
            import asyncio

            from models.prehop.ablation_inputs import read_input
            frozen = await asyncio.to_thread(read_input, frozen_inputs, seed_query)
            query_embedding, base_candidates = frozen["query_embedding"], frozen["base_candidates"]
        else:
            query_embedding = await self.llm.get_embedding(seed_query)
            if not query_embedding:
                raise ValueError(f"Graph retrieval received an empty query embedding for query={seed_query!r}")
            _unused_selected, base_candidates = await self._retrieve_with_candidate_pool(
                seed_query,
                top_k=top_k,
                query_embedding=query_embedding,
                select_final=False,
                selection_variant=selection_variant,
            )
        retrieve_ms = (time.perf_counter() - t_retrieve0) * 1000
        graph_expand_ms = 0.0
        score_timing: dict[str, float] = {}

        def timing() -> dict[str, float]:
            total_ms = (time.perf_counter() - t0) * 1000
            return {
                "retrieve_ms": retrieve_ms,
                "traversal_ms": max(0.0, total_ms - retrieve_ms),
                "graph_expand_ms": graph_expand_ms,
                "deterministic_score_ms": float(score_timing.get("deterministic_score_ms", 0.0)),
                "candidate_order_ms": float(score_timing.get("candidate_order_ms", 0.0)),
            }

        base_candidates = [node for node in base_candidates if self._node_identity(node) not in excluded_ids]
        if not base_candidates:
            self._trace_link_usefulness([], [], [])
            return "", [], timing()
        collected = {
            self._node_identity(node): dict(node)
            for node in base_candidates
            if self._node_identity(node) and self._node_identity(node) not in excluded_ids
        }
        base_candidate_ids = set(collected)
        frontier_ids = [self._node_identity(node) for node in base_candidates]
        hop_source_question_ids = {node_id: set() for node_id in frontier_ids if node_id}
        graph_expand_started = time.perf_counter()
        rows = await self._expand_frontier(
            frontier_ids,
            excluded_ids,
            hop_source_question_ids,
        )
        graph_expand_ms = (time.perf_counter() - graph_expand_started) * 1000
        for row, edge_rank in self._rank_frontier_rows(rows):
            target_id = str(row.get("id") or "").strip()
            path_type = str(row.get("path_type") or "").strip().lower()
            if not target_id or target_id in excluded_ids or path_type not in {"next", "hop"}:
                continue
            already_direct = target_id in base_candidate_ids
            candidate = collected.setdefault(
                target_id,
                {
                    key: row.get(key)
                    for key in (
                        "id",
                        "title",
                        "sent_id",
                        "page",
                        "text",
                        "source",
                        "author",
                        "publisher",
                        "published_at",
                        "category",
                        "url",
                        "embedding",
                    )
                },
            )
            source_candidate = collected.get(str(row.get("source_id") or ""), {})
            if not already_direct:
                inherited_score = float(source_candidate.get("representation_score", 0.0))
                # A graph-only target is supported indirectly through one
                # edge. Its inherited rank evidence therefore receives the
                # default reciprocal one-edge factor 1 / (depth + 1) = 0.5,
                # rather than being treated as strongly as a directly
                # retrieved owner. The env-driven value exists only for the
                # declared sensitivity experiment. Direct candidates retain
                # their original score; only their path provenance is enriched.
                inherited_score *= RAGConfig.GRAPH_PATH_DECAY
                if inherited_score > float(candidate.get("representation_score", 0.0)):
                    candidate["representation_score"] = inherited_score
            bridge_embeddings = [embedding for embedding in (row.get("bridge_embeddings") or []) if embedding]
            if not already_direct and path_type == "hop" and bridge_embeddings:
                existing = candidate.setdefault("bridge_embeddings", [])
                for embedding in bridge_embeddings:
                    if embedding not in existing:
                        existing.append(embedding)
            candidate.setdefault("retrieval_paths", []).append(
                {
                    "kind": path_type,
                    "source_chunk_id": row.get("source_id"),
                    "source_question_ids": row.get("activated_question_ids") or [],
                    "depth": 1,
                    "edge_rank": edge_rank,
                }
            )

        ranked_candidates = list(collected.values())
        active_selection_variant = selection_variant or RAGConfig.SOURCE_SELECTION_VARIANT
        score_kwargs = {"query_text": seed_query} if active_selection_variant == "role_body_list_ranking" else {}
        if selection_variant is not None:
            score_kwargs["selection_variant"] = selection_variant
        nodes, _ = await self._score_and_select(
            query_embedding,
            ranked_candidates,
            top_k,
            timing_sink=score_timing,
            **score_kwargs,
        )
        output_nodes = [self._without_transient_retrieval_scores(node) for node in nodes]
        self._trace_link_usefulness(base_candidates, rows, output_nodes, frontier_ids)
        if not output_nodes:
            return "", [], timing()
        return self._build_context_from_nodes(output_nodes), output_nodes, timing()

    async def _expand_frontier(
        self,
        frontier_ids: list[str],
        excluded_ids: set[str],
        hop_source_question_ids: dict[str, set[str]],
    ) -> list[dict[str, Any]]:
        if RAGConfig.GRAPH_EDGE_VARIANT == "none":
            return []
        branches: list[str] = []
        if RAGConfig.GRAPH_EDGE_VARIANT in {"full", "next_only"}:
            branches.append(
                f"""
                    MATCH (src)-[:NEXT]-(related:{self.chunk_label})
                    RETURN related, 'next' AS path_type,
                           null AS bridge_embeddings,
                           null AS activated_question_ids
                """
            )
        if RAGConfig.GRAPH_EDGE_VARIANT in {"full", "hop_only"}:
            branches.append(
                f"""
                    MATCH (src)-[hop:HOP_ANSWER]->(related:{self.chunk_label})
                    WHERE src.id IN $hop_source_ids
                    RETURN related, 'hop' AS path_type,
                           [(src)-[:HAS_Q_PLUS]->(q:{self.q_plus_label})
                            WHERE q.id IN coalesce(hop.source_question_ids, []) | q.embedding]
                           AS bridge_embeddings,
                           coalesce(hop.source_question_ids, []) AS activated_question_ids
                """
            )
        expansion_query = "\nUNION ALL\n".join(branches)
        async with self.neo4j.driver.session() as session:
            query = f"""
                UNWIND $frontier_ids AS src_id
                MATCH (src:{self.chunk_label} {{id: src_id}})
                CALL (src) {{
                    {expansion_query}
                }}
                WITH src, related, path_type, bridge_embeddings, activated_question_ids
                WHERE NOT related.id IN $excluded_ids
                RETURN src.id AS source_id, related.id AS id,
                       related.title AS title, related.sent_id AS sent_id,
                       related.page AS page, related.text AS text,
                       related.source AS source,
                       related[$author_property] AS author, related[$publisher_property] AS publisher,
                       related[$published_at_property] AS published_at,
                       related[$category_property] AS category, related[$url_property] AS url,
                       related.embedding AS embedding,
                       path_type, bridge_embeddings, activated_question_ids
                ORDER BY source_id, path_type, id
            """
            result = await session.run(
                query,
                {
                    "frontier_ids": frontier_ids,
                    "excluded_ids": list(excluded_ids),
                    "hop_source_ids": sorted(hop_source_question_ids),
                    "author_property": "author",
                    "publisher_property": "publisher",
                    "published_at_property": "published_at",
                    "category_property": "category",
                    "url_property": "url",
                },
            )
            rows = [dict(record) async for record in result]
        return sorted(rows, key=lambda r: (str(r.get("source_id")), str(r.get("path_type")), str(r.get("id"))))

    @staticmethod
    def _rank_frontier_rows(
        rows: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], int]]:
        """Assign a local rank within each source/relation channel."""
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row.get("source_id") or ""), str(row.get("path_type") or ""))].append(row)

        ranked: list[tuple[dict[str, Any], int]] = []
        for key in sorted(grouped):
            ordered = sorted(
                grouped[key],
                key=lambda row: str(row.get("id") or ""),
            )
            ranked.extend((row, rank) for rank, row in enumerate(ordered))
        return ranked
