"""Level-batched traversal over pre-built NEXT/HOP_ANSWER edges.

The complete representation-union seed pool is expanded in one Neo4j request.
Only query-matched Q+ owners expose HOP at level zero. The default P2 path
activates that owner's reciprocal HOP provenance; the exact matched-ID
intersection remains selectable for ablation. Later levels expose NEXT only. All structurally bounded results are
retained until final indexed-embedding selection, so traversal has no
reservoir multiplier.
"""

import time
from collections import defaultdict
from typing import Any

from core.config import RAGConfig


class TraversalMixin:
    async def graph_search(
        self,
        entities: list[str],
        depth: int,
        top_k: int,
        excluded_chunk_ids: set[str] | None = None,
        channel_queries: dict[str, list[str]] | None = None,
        selection_variant: str | None = None,
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
        _unused_selected, base_candidates = await self._retrieve_with_candidate_pool(
            seed_query,
            top_k=top_k,
            query_embedding=query_embedding,
            channel_queries=channel_queries,
            select_final=False,
            selection_variant=selection_variant,
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
        base_candidate_ids = set(collected)
        traversal_seeds = [node for node in base_candidates if not bool(node.get("role_body_owner_only"))]
        frontier_ids = [self._node_identity(node) for node in traversal_seeds]
        hop_source_question_ids = {
            self._node_identity(node): {
                str(question_id).strip()
                for question_id in (node.get("matched_qplus_ids") or [])
                if str(question_id).strip()
            }
            for node in traversal_seeds
            if bool(node.get("dependency_seed"))
        }
        hop_source_question_ids = {
            source_id: question_ids
            for source_id, question_ids in hop_source_question_ids.items()
            if source_id and question_ids
        }
        continuation_source_question_ids = {
            self._node_identity(node): {
                str(question_id).strip()
                for question_id in (node.get("matched_qminus_ids") or [])
                if str(question_id).strip()
            }
            for node in traversal_seeds
            if bool(node.get("continuation_seed"))
        }
        continuation_source_question_ids = {
            source_id: question_ids
            for source_id, question_ids in continuation_source_question_ids.items()
            if source_id and question_ids
        }
        rows = await self._expand_frontier(
            frontier_ids,
            excluded_ids,
            hop_source_question_ids,
            continuation_source_question_ids,
            query_embedding,
            top_k,
        )
        for row, edge_rank in self._rank_frontier_rows(rows):
            target_id = str(row.get("id") or "").strip()
            path_type = str(row.get("path_type") or "").strip().lower()
            if not target_id or target_id in excluded_ids or path_type not in {"next", "hop", "continuation"}:
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
            source_channel_scores = source_candidate.get("representation_scores") or {}
            if not already_direct:
                if path_type == "hop":
                    inherited_score = float(source_channel_scores.get("q_plus", 0.0))
                elif path_type == "continuation":
                    inherited_score = float(source_channel_scores.get("q_minus", 0.0))
                else:
                    inherited_score = float(source_candidate.get("representation_score", 0.0))
                # A graph-only target is supported indirectly through one
                # edge. Its inherited rank evidence therefore receives the
                # reciprocal path length 1 / (depth + 1), rather than being
                # treated as strongly as a directly retrieved owner. Direct
                # candidates retain their original score; only their path
                # provenance is enriched below.
                inherited_score *= 1.0 / (int(depth) + 1)
                if inherited_score > float(candidate.get("representation_score", 0.0)):
                    candidate["representation_score"] = inherited_score
            bridge_embeddings = [embedding for embedding in (row.get("bridge_embeddings") or []) if embedding]
            if not already_direct and path_type in {"hop", "continuation"} and bridge_embeddings:
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
            **score_kwargs,
        )
        output_nodes = [self._without_transient_retrieval_scores(node) for node in nodes]
        if not output_nodes:
            return "", [], timing()
        return self._build_context_from_nodes(output_nodes), output_nodes, timing()

    async def _expand_frontier(
        self,
        frontier_ids: list[str],
        excluded_ids: set[str],
        hop_source_question_ids: dict[str, set[str]],
        continuation_source_question_ids: dict[str, set[str]] | None = None,
        query_embedding: list[float] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        continuation_source_question_ids = continuation_source_question_ids or {}
        query_embedding = query_embedding or []
        top_k = RAGConfig.DEFAULT_TOP_K if top_k is None else max(1, int(top_k))
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
            if RAGConfig.QUESTION_SCHEMA == "linked_v2" and RAGConfig.CONTINUATION_EDGES_ENABLED:
                continuation_branch = f"""
                    MATCH (src)-[:HAS_Q_MINUS]->(matched_q:{self.q_minus_label})
                          -[:ANSWER_ANCHOR]->(:{self.answer_anchor_label})
                          -[:MENTIONED_IN]->(related:{self.chunk_label})
                    WHERE src.id IN $continuation_source_ids
                      AND matched_q.id IN coalesce($continuation_source_question_ids[src.id], [])
                      AND related.source <> src.source
                    WITH src, related,
                         collect(DISTINCT matched_q.id) AS activated_question_ids,
                         collect(DISTINCT matched_q.embedding) AS bridge_embeddings,
                         vector.similarity.cosine(related.embedding, $query_embedding)
                         AS expansion_score
                    ORDER BY expansion_score DESC, related.id
                    LIMIT $top_k
                    RETURN related, 'continuation' AS path_type,
                           bridge_embeddings,
                           activated_question_ids
                """
                branches.append(continuation_branch)
            if RAGConfig.HOP_EDGE_FILTER == "none":
                branches.append(
                    f"""
                        MATCH (src)-[hop:HOP_ANSWER]->(related:{self.chunk_label})
                        WHERE src.id IN $hop_source_ids
                              AND ($qplus_hop_activation = 'owner'
                                   OR any(question_id IN coalesce(hop.source_question_ids, [])
                                          WHERE question_id IN coalesce($hop_source_question_ids[src.id], [])))
                        RETURN related, 'hop' AS path_type,
                               [(src)-[:HAS_Q_PLUS]->(q:{self.q_plus_label})
                                WHERE q.id IN coalesce(hop.source_question_ids, [])
                                  AND ($qplus_hop_activation = 'owner'
                                       OR q.id IN coalesce($hop_source_question_ids[src.id], [])) | q.embedding]
                               AS bridge_embeddings,
                               [question_id IN coalesce(hop.source_question_ids, [])
                                WHERE $qplus_hop_activation = 'owner'
                                   OR question_id IN coalesce($hop_source_question_ids[src.id], [])]
                               AS activated_question_ids
                    """
                )
            elif RAGConfig.HOP_EDGE_FILTER == "reciprocal":
                branches.append(
                    f"""
                        MATCH (src)-[hop:HOP_ANSWER]->(related:{self.chunk_label})
                        WHERE src.id IN $hop_source_ids
                        MATCH (src)-[:HAS_Q_PLUS]->(edge_qplus:{self.q_plus_label})
                              -[:ANSWERED_BY]->(edge_qminus:{self.q_minus_label})
                        WHERE edge_qplus.id IN coalesce(hop.source_question_ids, [])
                              AND ($qplus_hop_activation = 'owner'
                                   OR edge_qplus.id IN coalesce($hop_source_question_ids[src.id], []))
                              AND (related)-[:HAS_Q_MINUS]->(edge_qminus)
                        CALL (related, edge_qplus, edge_qminus) {{
                            OPTIONAL MATCH (same_source_owner:{self.chunk_label})
                                           -[:HAS_Q_PLUS]->(same_source_qplus:{self.q_plus_label})
                            WHERE same_source_owner.source = related.source
                            WITH related, edge_qplus, edge_qminus,
                                 count(same_source_qplus) + 1 AS reverse_pool
                            CALL db.index.vector.queryNodes(
                                $qplus_vector_index, reverse_pool, edge_qminus.embedding
                            ) YIELD node, score
                            MATCH (reverse_owner:{self.chunk_label})-[:HAS_Q_PLUS]->(node)
                            WHERE reverse_owner.source <> related.source
                            WITH edge_qplus, node, score
                            ORDER BY score DESC, node.id
                            WITH edge_qplus, collect(node)[0] AS reverse_best
                            WHERE reverse_best.id = edge_qplus.id
                            RETURN edge_qplus.id AS reciprocal_question_id,
                                   edge_qplus.embedding AS reciprocal_embedding
                        }}
                        WITH related,
                             collect(DISTINCT reciprocal_embedding) AS bridge_embeddings,
                             collect(DISTINCT reciprocal_question_id) AS activated_question_ids
                        RETURN related, 'hop' AS path_type,
                               bridge_embeddings, activated_question_ids
                    """
                )
            else:
                branches.append(
                    f"""
                        MATCH (src)-[hop:HOP_ANSWER]->(related:{self.chunk_label})
                        WHERE src.id IN $hop_source_ids
                        WITH src, related, hop,
                             [question_id IN coalesce(hop.reciprocal_source_question_ids, [])
                              WHERE $qplus_hop_activation = 'owner'
                                 OR question_id IN coalesce($hop_source_question_ids[src.id], [])]
                             AS activated_question_ids
                        WHERE size(activated_question_ids) > 0
                        RETURN related, 'hop' AS path_type,
                               [(src)-[:HAS_Q_PLUS]->(q:{self.q_plus_label})
                                WHERE q.id IN activated_question_ids | q.embedding] AS bridge_embeddings,
                               activated_question_ids
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
                       related.author AS author, related.publisher AS publisher,
                       related.published_at AS published_at,
                       related.category AS category, related.url AS url,
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
                    "hop_source_question_ids": {
                        source_id: sorted(question_ids) for source_id, question_ids in hop_source_question_ids.items()
                    },
                    "continuation_source_ids": sorted(continuation_source_question_ids),
                    "continuation_source_question_ids": {
                        source_id: sorted(question_ids)
                        for source_id, question_ids in continuation_source_question_ids.items()
                    },
                    "query_embedding": query_embedding,
                    "top_k": top_k,
                    "qplus_hop_activation": RAGConfig.QPLUS_HOP_ACTIVATION,
                    "qplus_vector_index": self.q_plus_vector_index,
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
            ordered = sorted(
                grouped[key],
                key=lambda row: str(row.get("id") or ""),
            )
            ranked.extend((row, rank) for rank, row in enumerate(ordered))
        return ranked
