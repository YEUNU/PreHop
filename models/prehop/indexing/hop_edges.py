"""Build cross-document evidence edges from dependency questions."""

import asyncio
import logging
import os
from typing import Any

from core.config import RAGConfig

logger = logging.getLogger(__name__)


class HopEdgeMixin:
    async def _find_hop_candidates_batch(
        self,
        wave: list[dict[str, Any]],
        channel: str,
    ) -> list[dict[str, Any]]:
        source_questions = [
            {
                "chunk_id": source_item["id"],
                "source": source_item["source"],
                "id": question["id"],
                "embed": question["query_embedding"],
                "ann_pool": max(1, int(source_item.get("ann_pools", {}).get(channel, 0) or 0)),
            }
            for source_item in wave
            for question in source_item["questions"]
            if question.get("query_embedding")
        ]
        if not source_questions:
            return []

        if channel == "body":
            query = """
                UNWIND $source_questions AS source_question
                CALL db.index.vector.queryNodes($index, source_question.ann_pool, source_question.embed)
                YIELD node, score
                WHERE node.source <> source_question.source
                WITH source_question, node, score
                ORDER BY source_question.id, score DESC, node.id
                WITH source_question, collect({
                    target_id: node.id,
                    target_question_id: null,
                    score: score
                })[0..1] AS candidates
                UNWIND candidates AS candidate
                RETURN source_question.chunk_id AS source_chunk_id,
                       source_question.id AS source_question_id,
                       candidate.target_id AS target_id,
                       candidate.target_question_id AS target_question_id,
                       candidate.score AS score
                ORDER BY source_question_id, score DESC
            """
            index_name = self.body_vector_index
        elif channel == "q_minus":
            relationship = "HAS_Q_MINUS"
            index_name = self.q_minus_vector_index
            query = f"""
                UNWIND $source_questions AS source_question
                CALL db.index.vector.queryNodes($index, source_question.ann_pool, source_question.embed)
                YIELD node, score
                MATCH (target:{self.chunk_label})-[:{relationship}]->(node)
                WHERE target.source <> source_question.source
                WITH source_question, target, node, score
                ORDER BY source_question.id, target.id, score DESC
                WITH source_question, target,
                     collect({{target_question_id: node.id,
                               score: score}})[0] AS best
                ORDER BY source_question.id, best.score DESC, target.id
                WITH source_question, collect({{
                    target_id: target.id,
                    target_question_id: best.target_question_id,
                    score: best.score
                }})[0..1] AS candidates
                UNWIND candidates AS candidate
                RETURN source_question.chunk_id AS source_chunk_id,
                       source_question.id AS source_question_id,
                       candidate.target_id AS target_id,
                       candidate.target_question_id AS target_question_id,
                       candidate.score AS score
                ORDER BY source_question_id, score DESC
            """
        else:
            raise ValueError(f"Unsupported HOP candidate channel: {channel!r}")

        return await self.retry_query(
            query,
            {
                "index": index_name,
                "source_questions": source_questions,
            },
        )

    async def _collect_hop_wave_candidates(
        self,
        wave: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Resolve each Q+ through one answer-bearing representation."""
        channel = "q_minus" if RAGConfig.ABLATION_Q_MINUS else "body"
        candidates = await self._find_hop_candidates_batch(wave, channel)
        source_states: dict[str, dict[str, dict[str, dict[str, Any]]]] = {item["id"]: {} for item in wave}
        source_questions = {question["id"]: question for item in wave for question in item["questions"]}
        for candidate in candidates:
            source_chunk_id = str(candidate.get("source_chunk_id") or "").strip()
            source_question_id = str(candidate.get("source_question_id") or "").strip()
            if source_chunk_id not in source_states or source_question_id not in source_questions:
                continue
            fused = source_states[source_chunk_id].setdefault(source_question_id, {})
            target_id = str(candidate.get("target_id") or "").strip()
            if not target_id or target_id == source_chunk_id:
                continue
            state = fused.setdefault(
                target_id,
                {
                    "src_id": source_chunk_id,
                    "tgt_id": target_id,
                    "direct_channels": set(),
                    "best": {},
                    "source_question_id": source_question_id,
                    "source_question_text": source_questions[source_question_id].get("text", ""),
                },
            )
            state["direct_channels"].add(channel)

            raw_score = float(candidate.get("score") or 0.0)
            previous = state["best"].get(channel)
            if previous is None or raw_score > previous["raw_score"]:
                state["best"][channel] = {
                    "raw_score": raw_score,
                    "source_question_id": source_question_id,
                    "source_question_text": source_questions[source_question_id].get("text", ""),
                    "target_question_id": candidate.get("target_question_id"),
                }

        def state_rank(state: dict[str, Any]) -> tuple[float, str]:
            return (
                -max((item["raw_score"] for item in state["best"].values()), default=0.0),
                state["tgt_id"],
            )

        required_channels = {channel}
        tentative_by_source: list[list[dict[str, Any]]] = []
        for source_item in wave:
            fused_by_question = source_states[source_item["id"]]
            eligible_by_question: dict[str, list[dict[str, Any]]] = {}
            for question in source_item["questions"]:
                question_id = question["id"]
                eligible = [
                    state
                    for state in fused_by_question.get(question_id, {}).values()
                    if state["direct_channels"] == required_channels
                ]
                eligible.sort(key=state_rank)
                eligible_by_question[question_id] = eligible

            chosen_pairs: list[dict[str, Any]] = []
            for question in source_item["questions"]:
                eligible = eligible_by_question.get(question["id"], [])
                if not eligible:
                    continue
                chosen_pairs.append(eligible[0])
            tentative_by_source.append(chosen_pairs)

        return tentative_by_source

    async def _merge_hop_candidates(
        self,
        tentative_by_source: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Merge question-level resolutions into source-target edges."""

        selected_edges: list[dict[str, Any]] = []
        for chosen_pairs in tentative_by_source:
            merged: dict[str, dict[str, Any]] = {}
            for state in chosen_pairs:
                best = state["best"]
                edge = merged.setdefault(
                    state["tgt_id"],
                    {
                        "src_id": state["src_id"],
                        "tgt_id": state["tgt_id"],
                        "direct_channels": set(),
                        "q_minus_matches": [],
                        "body_matches": [],
                    },
                )
                edge["direct_channels"].update(state["direct_channels"])
                for channel, key in (
                    ("q_minus", "q_minus_matches"),
                    ("body", "body_matches"),
                ):
                    if best.get(channel):
                        edge[key].append(best[channel])

            for edge in merged.values():
                edge["direct_channels"] = sorted(edge["direct_channels"])
                for plural, singular in (
                    ("q_minus_matches", "q_minus_match"),
                    ("body_matches", "body_match"),
                ):
                    matches = edge[plural]
                    edge[singular] = max(matches, key=lambda item: item["raw_score"], default=None)
                all_matches = edge["q_minus_matches"] + edge["body_matches"]
                edge["source_question_ids"] = list(dict.fromkeys(match["source_question_id"] for match in all_matches))
                edge["source_question_texts"] = list(
                    dict.fromkeys(
                        match["source_question_text"]
                        for match in all_matches
                        if str(match.get("source_question_text") or "").strip()
                    )
                )
                edge["construction_mode"] = (
                    "qplus_to_qminus_owner" if RAGConfig.ABLATION_Q_MINUS else "qplus_to_body_ablation"
                )
                selected_edges.append(edge)

        return selected_edges

    async def _process_hop_wave(
        self,
        wave: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build answer-owner edges for one bounded collection wave."""
        tentative_by_source = await self._collect_hop_wave_candidates(wave)
        return await self._merge_hop_candidates(tentative_by_source)

    async def _collect_hop_page_candidates(
        self,
        page_items: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Resolve bounded ANN waves concurrently while preserving source order."""
        wave_size = RAGConfig.HOP_GATHER_WAVE
        concurrent_span = wave_size * RAGConfig.HOP_BUILD_CONCURRENCY
        tentative_by_source: list[list[dict[str, Any]]] = []
        for group_start in range(0, len(page_items), concurrent_span):
            group = page_items[group_start : group_start + concurrent_span]
            resolved_waves = await asyncio.gather(
                *(
                    self._collect_hop_wave_candidates(group[offset : offset + wave_size])
                    for offset in range(0, len(group), wave_size)
                )
            )
            for resolved in resolved_waves:
                tentative_by_source.extend(resolved)
        return tentative_by_source

    async def _flush_hop_edges(self, edges: list[dict[str, Any]]) -> None:
        if not edges:
            return

        await self.retry_query(
            f"""
            WITH $edges AS edges
            CALL (edges) {{
                UNWIND edges AS edge
                MATCH (src:{self.chunk_label} {{id: edge.src_id}})
                MATCH (tgt:{self.chunk_label} {{id: edge.tgt_id}})
                MERGE (src)-[r:HOP_ANSWER]->(tgt)
                SET r.direct_channels = edge.direct_channels,
                    r.q_minus_score = edge.q_minus_match.raw_score,
                    r.body_score = edge.body_match.raw_score,
                    r.source_question_ids = edge.source_question_ids,
                    r.source_question_texts = edge.source_question_texts,
                    r.type = edge.construction_mode
                RETURN count(r) AS hop_edges_written
            }}
            CALL (edges) {{
                UNWIND edges AS edge
                UNWIND edge.q_minus_matches AS match
                WITH match WHERE match.target_question_id IS NOT NULL
                MATCH (src:{self.q_plus_label} {{id: match.source_question_id}})
                MATCH (tgt:{self.q_minus_label} {{id: match.target_question_id}})
                MERGE (src)-[r:ANSWERED_BY]->(tgt)
                SET r.score = match.raw_score
                RETURN count(r) AS answered_by_written
            }}
            CALL (edges) {{
                UNWIND edges AS edge
                UNWIND edge.body_matches AS match
                MATCH (src:{self.q_plus_label} {{id: match.source_question_id}})
                MATCH (tgt:{self.chunk_label} {{id: edge.tgt_id}})
                MERGE (src)-[r:SUPPORTED_BY]->(tgt)
                SET r.score = match.raw_score
                RETURN count(r) AS supported_by_written
            }}
            RETURN hop_edges_written, answered_by_written, supported_by_written
            """,
            {"edges": edges},
        )

    async def clear_hop_edges(self) -> None:
        """Delete HOP and question-level provenance edges before rebuilding."""
        await self.retry_query(f"MATCH (:{self.chunk_label})-[r:HOP_ANSWER]->(:{self.chunk_label}) DELETE r")
        await self.retry_query(f"MATCH (anchor:{self.answer_anchor_label}) DETACH DELETE anchor")
        await self.retry_query(f"MATCH (:{self.q_plus_label})-[r]->() DELETE r")

    async def _precompute_reciprocal_hop_provenance(self) -> int:
        """Materialize the reciprocal Q+ rule once for query-time reuse.

        This uses the same reverse nearest-neighbour definition as the legacy
        query-time filter. Bounded ANN pages are resolved concurrently, then
        grouped by HOP edge and written once so concurrent updates cannot lose
        provenance IDs.
        """
        await self.retry_query(
            f"MATCH (:{self.chunk_label})-[h:HOP_ANSWER]->(:{self.chunk_label}) "
            "SET h.reciprocal_source_question_ids = []"
        )
        page_size = max(32, min(512, RAGConfig.HOP_GATHER_WAVE * 2))
        last_id = ""
        question_pages: list[list[str]] = []
        while True:
            id_rows = await self.retry_query(
                f"""
                MATCH (q:{self.q_plus_label})-[:ANSWERED_BY]->(:{self.q_minus_label})
                WHERE q.id > $last_id
                RETURN q.id AS id
                ORDER BY q.id
                LIMIT $limit
                """,
                {"last_id": last_id, "limit": page_size},
            )
            if not id_rows:
                break
            question_ids = [str(row["id"]) for row in id_rows]
            last_id = question_ids[-1]
            question_pages.append(question_ids)
            if len(id_rows) < page_size:
                break

        async def resolve_page(question_ids: list[str]) -> list[dict[str, Any]]:
            return await self.retry_query(
                f"""
                UNWIND $question_ids AS question_id
                MATCH (source_owner:{self.chunk_label})-[:HAS_Q_PLUS]->
                      (edge_qplus:{self.q_plus_label} {{id: question_id}})
                      -[:ANSWERED_BY]->(edge_qminus:{self.q_minus_label})
                MATCH (related:{self.chunk_label})-[:HAS_Q_MINUS]->(edge_qminus)
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
                    RETURN reverse_best.id = edge_qplus.id AS reciprocal
                }}
                WITH source_owner, related, edge_qplus, reciprocal
                WHERE reciprocal
                MATCH (source_owner)-[hop:HOP_ANSWER]->(related)
                WHERE edge_qplus.id IN coalesce(hop.source_question_ids, [])
                RETURN source_owner.id AS source_id, related.id AS target_id,
                       edge_qplus.id AS question_id
                """,
                {
                    "question_ids": question_ids,
                    "qplus_vector_index": self.q_plus_vector_index,
                },
            )

        resolved_rows: list[dict[str, Any]] = []
        concurrency = RAGConfig.HOP_BUILD_CONCURRENCY
        for start in range(0, len(question_pages), concurrency):
            resolved_pages = await asyncio.gather(
                *(resolve_page(page) for page in question_pages[start : start + concurrency])
            )
            for rows in resolved_pages:
                resolved_rows.extend(rows)

        reciprocal_by_edge: dict[tuple[str, str], set[str]] = {}
        for row in resolved_rows:
            edge = (str(row["source_id"]), str(row["target_id"]))
            reciprocal_by_edge.setdefault(edge, set()).add(str(row["question_id"]))
        write_rows = [
            {
                "source_id": source_id,
                "target_id": target_id,
                "question_ids": sorted(question_ids),
            }
            for (source_id, target_id), question_ids in sorted(reciprocal_by_edge.items())
        ]
        for start in range(0, len(write_rows), page_size):
            await self.retry_query(
                f"""
                UNWIND $rows AS row
                MATCH (source:{self.chunk_label} {{id: row.source_id}})
                      -[hop:HOP_ANSWER]->
                      (target:{self.chunk_label} {{id: row.target_id}})
                SET hop.reciprocal_source_question_ids = row.question_ids
                """,
                {"rows": write_rows[start : start + page_size]},
            )

        reciprocal_pairs = sum(len(row["question_ids"]) for row in write_rows)
        logger.info(
            "Precomputed %d reciprocal Q+ provenance pair(s) for query-time traversal.",
            reciprocal_pairs,
        )
        return reciprocal_pairs

    async def build_all_hop_edges(self) -> None:
        """Build Q+-to-answer-owner HOP edges after the complete corpus is visible."""
        if not RAGConfig.ABLATION_Q_PLUS:
            logger.info("Skipping HOP edge construction because Q+ is disabled.")
            return

        await self.retry_query("CALL db.awaitIndexes($timeout_seconds)", {"timeout_seconds": 300})

        if RAGConfig.ABLATION_Q_MINUS:
            pool_rows = await self.retry_query(
                f"""
                MATCH (c:{self.chunk_label})-[:HAS_Q_MINUS]->(q:{self.q_minus_label})
                RETURN c.source AS source, count(q) AS count_per_source
                """
            )
            pool_channel = "q_minus"
        else:
            pool_rows = await self.retry_query(
                f"""
                MATCH (c:{self.chunk_label})
                RETURN c.source AS source, count(c) AS count_per_source
                """
            )
            pool_channel = "body"

        def cross_document_pools(rows: list[dict[str, Any]]) -> dict[str, int]:
            # Neo4j filters documents after ANN, so each source pool includes
            # its own representations plus one foreign candidate.
            return {str(row.get("source") or ""): int(row.get("count_per_source", 0) or 0) + 1 for row in rows}

        source_ann_pools = {pool_channel: cross_document_pools(pool_rows)}

        pool_maxima = {channel: max(pools.values(), default=1) for channel, pools in source_ann_pools.items()}

        page_size = max(100, int(os.environ.get("RAG_HOP_PAGE_SIZE", "5000")))
        wave_size = RAGConfig.HOP_GATHER_WAVE
        wave_concurrency = RAGConfig.HOP_BUILD_CONCURRENCY
        total_sources = 0
        total_edges = 0
        last_id = ""
        while True:
            # Keyset pagination limits each transaction before joining Q+
            # questions and high-dimensional embeddings.
            rows = await self.retry_query(
                f"""
                MATCH (src:{self.chunk_label})
                WHERE src.id > $last_id
                WITH src
                ORDER BY src.id
                LIMIT $limit
                OPTIONAL MATCH (src)-[:HAS_Q_PLUS]->(q:{self.q_plus_label})
                WHERE q.query_embedding IS NOT NULL
                WITH src, q
                ORDER BY src.id, q.ordinal, q.id
                WITH src, collect(CASE WHEN q IS NULL THEN null ELSE {{
                    id: q.id,
                    text: q.text,
                    query_embedding: q.query_embedding
                }} END) AS raw_questions
                RETURN src.id AS id, src.source AS source,
                       [x IN raw_questions WHERE x IS NOT NULL] AS questions
                ORDER BY src.id
                """,
                {"last_id": last_id, "limit": page_size},
            )
            if not rows:
                break

            last_id = rows[-1]["id"]
            page_items = [
                {
                    "id": row["id"],
                    "source": row["source"],
                    "questions": row["questions"],
                    "ann_pools": {
                        channel: pools.get(str(row["source"]), 1) for channel, pools in source_ann_pools.items()
                    },
                }
                for row in rows
                if row["questions"]
            ]
            page_count = len(page_items)
            if total_sources == 0:
                logger.info(
                    "build_all_hop_edges: Q+ answer-owner resolution "
                    "(page=%d wave=%d concurrent_waves=%d per-source-ann-pool-maxima=%s).",
                    page_size,
                    wave_size,
                    wave_concurrency,
                    pool_maxima,
                )

            tentative_by_source = await self._collect_hop_page_candidates(page_items)
            edges = await self._merge_hop_candidates(tentative_by_source)
            for edge_start in range(0, len(edges), wave_size):
                await self._flush_hop_edges(edges[edge_start : edge_start + wave_size])
            total_edges += len(edges)

            total_sources += page_count
            logger.info(
                "build_all_hop_edges: progress %d Q+-bearing chunks / %d HOP_ANSWER edges.",
                total_sources,
                total_edges,
            )
            # Candidate scan size, not Q+-bearing count, signals the end --
            # a partial page can still be full of Q+-less chunks with more
            # Q+-bearing chunks further along the id range.
            if len(rows) < page_size:
                break

        logger.info(
            "build_all_hop_edges: wrote %d HOP_ANSWER edges over %d Q+-bearing chunks.",
            total_edges,
            total_sources,
        )
        if RAGConfig.QUESTION_SCHEMA == "linked_v2":
            await self.build_answer_links()
        if RAGConfig.PRECOMPUTE_RECIPROCAL_HOPS and RAGConfig.ABLATION_Q_MINUS:
            await self._precompute_reciprocal_hop_provenance()
