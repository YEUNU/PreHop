"""Level-batched traversal over pre-built NEXT/HOP_ANSWER edges.

The complete representation-union seed pool is expanded in one Neo4j request.
Only query-matched Q+ owners expose HOP at level zero; later levels expose
NEXT only. All structurally bounded results are retained until final indexed-
embedding selection, so traversal has no reservoir multiplier.
"""

import time
from collections import defaultdict
from typing import Any

from core.config import RAGConfig


class TraversalMixin:
    async def graph_search(
        self,
        entities: list[str],
        depth: int = 1,
        top_k: int = 5,
        excluded_chunk_ids: set[str] | None = None,
    ) -> tuple:
        """Retrieve evidence through level-batched, duplicate-free graph expansion."""
        t0 = time.perf_counter()
        normalized_entities = [normalized for entity in entities if (normalized := self._normalize_entity_term(entity))]
        seed_query = " ".join(normalized_entities).strip() or " ".join(entities).strip()
        if not seed_query:
            return "", [], {"retrieve_ms": 0.0, "traversal_ms": 0.0}

        if int(depth) != 1:
            raise ValueError("Graph traversal depth must be exactly one")
        excluded_ids = {str(chunk_id).strip() for chunk_id in (excluded_chunk_ids or set()) if str(chunk_id).strip()}

        t_retrieve0 = time.perf_counter()
        query_embedding = await self.llm.get_embedding(seed_query)
        if not query_embedding:
            raise ValueError(f"Graph retrieval received an empty query embedding for query={seed_query!r}")
        _semantic_seeds, base_candidates = await self._retrieve_with_candidate_pool(
            seed_query,
            top_k=top_k,
            query_embedding=query_embedding,
        )
        retrieve_ms = (time.perf_counter() - t_retrieve0) * 1000

        def timing() -> dict[str, float]:
            total_ms = (time.perf_counter() - t0) * 1000
            return {
                "retrieve_ms": retrieve_ms,
                "traversal_ms": max(0.0, total_ms - retrieve_ms),
            }

        base_candidates = [node for node in base_candidates if self._node_identity(node) not in excluded_ids]
        if not base_candidates:
            return "", [], timing()
        collected = {
            self._node_identity(node): dict(node)
            for node in base_candidates
            if self._node_identity(node) and self._node_identity(node) not in excluded_ids
        }
        discovered_ids = set(excluded_ids) | set(collected)
        frontier_ids = list(collected)
        hop_source_ids = {
            self._node_identity(node) for node in base_candidates if bool(node.get("dependency_seed"))
        }
        rows = await self._expand_frontier(frontier_ids, discovered_ids, hop_source_ids)
        for row, edge_rank in self._rank_frontier_rows(rows):
            target_id = str(row.get("id") or "").strip()
            path_type = str(row.get("path_type") or "").strip().lower()
            if not target_id or target_id in discovered_ids or path_type not in {"next", "hop"}:
                continue
            candidate = collected.setdefault(
                target_id,
                {
                    key: row.get(key)
                    for key in ("id", "title", "sent_id", "page", "text", "source", "embedding")
                },
            )
            bridge_embeddings = [embedding for embedding in (row.get("bridge_embeddings") or []) if embedding]
            if path_type == "hop" and bridge_embeddings:
                existing = candidate.setdefault("bridge_embeddings", [])
                for embedding in bridge_embeddings:
                    if embedding not in existing:
                        existing.append(embedding)
            candidate.setdefault("retrieval_paths", []).append(
                {
                    "kind": path_type,
                    "source_chunk_id": row.get("source_id"),
                    "depth": 1,
                    "edge_rank": edge_rank,
                }
            )

        ranked_candidates = list(collected.values())
        nodes, _ = await self._score_and_select(query_embedding, ranked_candidates, top_k)
        output_nodes = [self._without_transient_retrieval_scores(node) for node in nodes]
        if not output_nodes:
            return "", [], timing()
        return self._build_context_from_nodes(output_nodes), output_nodes, timing()

    async def _expand_frontier(
        self,
        frontier_ids: list[str],
        discovered_ids: set[str],
        hop_source_ids: set[str],
    ) -> list[dict[str, Any]]:
        step_limit = len(frontier_ids) * (RAGConfig.QUESTIONS_PER_DIRECTION + 2)
        async with self.neo4j.driver.session() as session:
            query = f"""
                UNWIND $frontier_ids AS src_id
                MATCH (src:{self.chunk_label} {{id: src_id}})
                CALL (src) {{
                    MATCH (src)-[:NEXT]-(related:{self.chunk_label})
                    RETURN related, 'next' AS path_type, null AS edge_score,
                           null AS bridge_embeddings
                    UNION ALL
                    MATCH (src)-[hop:HOP_ANSWER]->(related:{self.chunk_label})
                    WHERE src.id IN $hop_source_ids
                    RETURN related, 'hop' AS path_type, hop.score AS edge_score,
                           [(src)-[:HAS_Q_PLUS]->(q:{self.q_plus_label})
                            WHERE q.id IN coalesce(hop.source_question_ids, []) | q.embedding]
                           AS bridge_embeddings
                }}
                WITH src, related, path_type, edge_score, bridge_embeddings
                WHERE NOT related.id IN $discovered_ids
                RETURN src.id AS source_id, related.id AS id,
                       related.title AS title, related.sent_id AS sent_id,
                       related.page AS page, related.text AS text,
                       related.source AS source, related.embedding AS embedding,
                       path_type, edge_score, bridge_embeddings
                ORDER BY source_id, path_type, edge_score DESC, id
                LIMIT $limit
            """
            result = await session.run(
                query,
                {
                    "frontier_ids": frontier_ids,
                    "discovered_ids": list(discovered_ids),
                    "hop_source_ids": list(hop_source_ids),
                    "limit": step_limit,
                },
            )
            return [dict(record) async for record in result]

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
            path_type = key[1]
            ordered = sorted(
                grouped[key],
                key=lambda row: (
                    -(float(row.get("edge_score")) if row.get("edge_score") is not None else 0.0)
                    if path_type == "hop"
                    else 0.0,
                    str(row.get("id") or ""),
                ),
            )
            ranked.extend((row, rank) for rank, row in enumerate(ordered))
        return ranked
