"""Incremental best-first traversal over pre-built NEXT/HOP_ANSWER edges.

The wider Stage-2 RRF pool is a seed queue. One unseen seed is expanded at a
time; discovered chunks are not selected again as independent seeds. NEXT and
HOP paths are ranked separately, fused at chunk level, and pruned to the
existing retrieval reservoir before external-embedding evidence selection.
"""

import time
from collections import defaultdict
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
        """Retrieve evidence through incremental, duplicate-free graph expansion."""
        t0 = time.perf_counter()
        normalized_entities = [
            normalized
            for entity in entities
            if (normalized := self._normalize_entity_term(entity))
        ]
        seed_query = " ".join(normalized_entities).strip() or " ".join(entities).strip()
        if not seed_query:
            return "", [], {"retrieve_ms": 0.0, "traversal_ms": 0.0}

        depth = max(1, min(int(depth), 4))
        excluded_ids = {
            str(chunk_id).strip()
            for chunk_id in (excluded_chunk_ids or set())
            if str(chunk_id).strip()
        }
        candidate_budget = max(24, top_k * RAGConfig.WIDE_POOL_MULTIPLIER)

        t_retrieve0 = time.perf_counter()
        semantic_seeds, base_candidates = await self._retrieve_with_candidate_pool(
            seed_query,
            top_k=top_k,
        )
        retrieve_ms = (time.perf_counter() - t_retrieve0) * 1000

        def timing() -> dict[str, float]:
            total_ms = (time.perf_counter() - t0) * 1000
            return {
                "retrieve_ms": retrieve_ms,
                "traversal_ms": max(0.0, total_ms - retrieve_ms),
            }

        base_candidates = [
            node
            for node in base_candidates
            if self._node_identity(node) not in excluded_ids
        ]
        if not base_candidates:
            return "", [], timing()

        base_rank = {
            self._node_identity(node): rank for rank, node in enumerate(base_candidates)
        }
        collected: dict[str, dict[str, Any]] = {}
        for node in semantic_seeds:
            node_id = self._node_identity(node)
            if node_id and node_id not in excluded_ids:
                collected[node_id] = dict(node)

        discovered_ids = set(excluded_ids)
        expanded_ids: set[str] = set()
        best_path_strength: dict[str, dict[str, float]] = {
            "next": {},
            "hop": {},
        }

        for seed_rank, seed in enumerate(base_candidates):
            if len(collected) >= candidate_budget:
                break
            seed_id = self._node_identity(seed)
            if not seed_id or seed_id in expanded_ids:
                continue

            collected.setdefault(seed_id, dict(seed))
            discovered_ids.add(seed_id)
            expanded_ids.add(seed_id)
            frontier_ids = [seed_id]

            for level in range(depth):
                if not frontier_ids or len(collected) >= candidate_budget:
                    break
                rows = await self._expand_frontier(frontier_ids, discovered_ids)
                if not rows:
                    break

                ranked_rows = self._rank_frontier_rows(rows)
                next_frontier: list[str] = []
                for row, edge_rank in ranked_rows:
                    target_id = str(row.get("id") or "").strip()
                    if not target_id or target_id in discovered_ids:
                        continue
                    path_type = str(row.get("path_type") or "").strip().lower()
                    if path_type not in best_path_strength:
                        continue
                    path_strength = (
                        1.0 / (RAGConfig.RRF_K_CONSTANT + seed_rank)
                        + 1.0 / (RAGConfig.RRF_K_CONSTANT + edge_rank)
                        + 1.0 / (RAGConfig.RRF_K_CONSTANT + level)
                    )
                    previous_strength = best_path_strength[path_type].get(target_id)
                    if previous_strength is None or path_strength > previous_strength:
                        best_path_strength[path_type][target_id] = path_strength

                    candidate = {
                        key: row.get(key)
                        for key in ("id", "title", "sent_id", "page", "text", "source")
                    }
                    candidate["retrieval_paths"] = [
                        {
                            "kind": path_type,
                            "source_chunk_id": row.get("source_id"),
                            "depth": level + 1,
                            "edge_rank": edge_rank,
                        }
                    ]
                    collected[target_id] = candidate
                    discovered_ids.add(target_id)
                    next_frontier.append(target_id)

                collected = dict(
                    self._incremental_rrf_order(
                        collected,
                        base_rank,
                        best_path_strength,
                    )[:candidate_budget]
                )
                frontier_ids = [
                    node_id
                    for node_id in next_frontier
                    if node_id in collected and node_id not in expanded_ids
                ]
                expanded_ids.update(frontier_ids)

        ranked_candidates = [
            node
            for _, node in self._incremental_rrf_order(
                collected,
                base_rank,
                best_path_strength,
            )[:candidate_budget]
        ]
        search_query = " ".join(entities).strip() or " ".join(normalized_entities)
        nodes, _ = await self._score_and_select(search_query, ranked_candidates, top_k)
        output_nodes = [self._without_transient_retrieval_scores(node) for node in nodes]
        if not output_nodes:
            return "", [], timing()
        return self._build_context_from_nodes(output_nodes), output_nodes, timing()

    async def _expand_frontier(
        self,
        frontier_ids: list[str],
        discovered_ids: set[str],
    ) -> list[dict[str, Any]]:
        step_limit = max(
            RAGConfig.GRAPH_SEARCH_LIMIT,
            len(frontier_ids) * (RAGConfig.HOP_LINK_LIMIT + 2),
        )
        async with self.neo4j.driver.session() as session:
            query = f"""
                UNWIND $frontier_ids AS src_id
                MATCH (src:{self.chunk_label} {{id: src_id}})
                CALL (src) {{
                    MATCH (src)-[:NEXT]-(related:{self.chunk_label})
                    RETURN related, 'next' AS path_type, null AS edge_score
                    UNION ALL
                    MATCH (src)-[hop:HOP_ANSWER]->(related:{self.chunk_label})
                    RETURN related, 'hop' AS path_type, hop.score AS edge_score
                }}
                WITH src, related, path_type, edge_score
                WHERE NOT related.id IN $discovered_ids
                RETURN src.id AS source_id, related.id AS id,
                       related.title AS title, related.sent_id AS sent_id,
                       related.page AS page, related.text AS text,
                       related.source AS source, path_type, edge_score
                ORDER BY source_id, path_type, edge_score DESC, id
                LIMIT $limit
            """
            result = await session.run(
                query,
                {
                    "frontier_ids": frontier_ids,
                    "discovered_ids": list(discovered_ids),
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

    def _incremental_rrf_order(
        self,
        collected: dict[str, dict[str, Any]],
        base_rank: dict[str, int],
        best_path_strength: dict[str, dict[str, float]],
    ) -> list[tuple[str, dict[str, Any]]]:
        channel_ranks: dict[str, dict[str, int]] = {}
        present_ids = set(collected)
        base_order = sorted(
            (node_id for node_id in present_ids if node_id in base_rank),
            key=lambda node_id: (base_rank[node_id], node_id),
        )
        channel_ranks["base"] = {
            node_id: rank for rank, node_id in enumerate(base_order)
        }
        for path_type in ("next", "hop"):
            path_order = sorted(
                (
                    node_id
                    for node_id in present_ids
                    if node_id in best_path_strength[path_type]
                ),
                key=lambda node_id: (-best_path_strength[path_type][node_id], node_id),
            )
            channel_ranks[path_type] = {
                node_id: rank for rank, node_id in enumerate(path_order)
            }

        ordered: list[tuple[str, dict[str, Any]]] = []
        for node_id, node in collected.items():
            score = sum(
                1.0 / (RAGConfig.RRF_K_CONSTANT + ranks[node_id])
                for ranks in channel_ranks.values()
                if node_id in ranks
            )
            node["graph_rrf_score"] = score
            ordered.append((node_id, node))
        ordered.sort(
            key=lambda item: (item[1].get("graph_rrf_score", 0.0), item[0]),
            reverse=True,
        )
        return ordered
