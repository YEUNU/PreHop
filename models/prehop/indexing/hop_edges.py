"""Build cross-document evidence edges from generated dependency questions.

Every generated Q+ is represented independently.  For each source Q+ the
builder retrieves two cross-document candidate channels:

* Q+ -> Q-: a target question that its owner chunk can answer;
* Q+ -> body: direct passage-level answer evidence;
Each Q+ retains one target supported by Q- or body retrieval. Direct-channel
agreement ranks candidates without a fusion constant, channel weight, or score
threshold.
"""

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
                ORDER BY source_question.id, score DESC
                WITH source_question, collect({
                    target_id: node.id,
                    target_source: node.source,
                    target_question_id: null,
                    score: score
                })[0..1] AS candidates
                UNWIND candidates AS candidate
                RETURN source_question.chunk_id AS source_chunk_id,
                       source_question.id AS source_question_id,
                       candidate.target_id AS target_id,
                       candidate.target_source AS target_source,
                       candidate.target_question_id AS target_question_id,
                       candidate.score AS score
                ORDER BY source_question_id, score DESC
            """
            index_name = self.body_vector_index
        elif channel in {"q_minus", "q_plus"}:
            relationship = "HAS_Q_MINUS" if channel == "q_minus" else "HAS_Q_PLUS"
            index_name = self.q_minus_vector_index if channel == "q_minus" else self.q_plus_vector_index
            query = f"""
                UNWIND $source_questions AS source_question
                CALL db.index.vector.queryNodes($index, source_question.ann_pool, source_question.embed)
                YIELD node, score
                MATCH (target:{self.chunk_label})-[:{relationship}]->(node)
                WHERE target.source <> source_question.source
                WITH source_question, target, node, score
                ORDER BY source_question.id, target.id, score DESC
                WITH source_question, target,
                     collect({{target_question_id: node.id, score: score}})[0] AS best
                ORDER BY source_question.id, best.score DESC
                WITH source_question, collect({{
                    target_id: target.id,
                    target_source: target.source,
                    target_question_id: best.target_question_id,
                    score: best.score
                }})[0..1] AS candidates
                UNWIND candidates AS candidate
                RETURN source_question.chunk_id AS source_chunk_id,
                       source_question.id AS source_question_id,
                       candidate.target_id AS target_id,
                       candidate.target_source AS target_source,
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

    async def _process_hop_wave(
        self,
        wave: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Fuse all individual Q+ directions for each source chunk."""
        channels = ["body"]
        if RAGConfig.ABLATION_Q_MINUS:
            channels.insert(0, "q_minus")
        channel_results = await asyncio.gather(
            *(self._find_hop_candidates_batch(wave, channel) for channel in channels)
        )
        source_states: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
            item["id"]: {} for item in wave
        }
        source_questions = {
            question["id"]: question
            for item in wave
            for question in item["questions"]
        }
        for channel, candidates in zip(channels, channel_results):
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
                        "score": 0.0,
                        "direct_channels": set(),
                        "best": {},
                    },
                )
                state["score"] += 1.0
                if channel in {"q_minus", "body"}:
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

        def state_rank(state: dict[str, Any]) -> tuple[float, float, str]:
            return (
                -state["score"],
                -max((item["raw_score"] for item in state["best"].values()), default=0.0),
                state["tgt_id"],
            )

        selected_edges: list[dict[str, Any]] = []
        for source_item in wave:
            fused_by_question = source_states[source_item["id"]]
            eligible_by_question: dict[str, list[dict[str, Any]]] = {}
            for question in source_item["questions"]:
                question_id = question["id"]
                eligible = [
                    state
                    for state in fused_by_question.get(question_id, {}).values()
                    if state["direct_channels"]
                ]
                eligible.sort(key=state_rank)
                eligible_by_question[question_id] = eligible

            chosen_pairs: list[dict[str, Any]] = []
            selected_targets: set[str] = set()
            for question in source_item["questions"]:
                eligible = eligible_by_question.get(question["id"], [])
                if not eligible:
                    continue
                if eligible[0]["tgt_id"] not in selected_targets:
                    chosen_pairs.append(eligible[0])
                    selected_targets.add(eligible[0]["tgt_id"])

            merged: dict[str, dict[str, Any]] = {}
            for state in chosen_pairs:
                best = state["best"]
                edge = merged.setdefault(
                    state["tgt_id"],
                    {
                        "src_id": state["src_id"],
                        "tgt_id": state["tgt_id"],
                        "score": 0.0,
                        "direct_channels": set(),
                        "q_minus_matches": [],
                        "body_matches": [],
                    },
                )
                edge["score"] = max(edge["score"], state["score"])
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
                all_matches = (
                    edge["q_minus_matches"]
                    + edge["body_matches"]
                )
                edge["source_question_ids"] = list(
                    dict.fromkeys(match["source_question_id"] for match in all_matches)
                )
                edge["source_question_texts"] = list(
                    dict.fromkeys(
                        match["source_question_text"]
                        for match in all_matches
                        if str(match.get("source_question_text") or "").strip()
                    )
                )
                selected_edges.append(edge)

        return selected_edges

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
                SET r.score = edge.score,
                    r.direct_channels = edge.direct_channels,
                    r.q_minus_score = edge.q_minus_match.raw_score,
                    r.body_score = edge.body_match.raw_score,
                    r.source_question_ids = edge.source_question_ids,
                    r.source_question_texts = edge.source_question_texts,
                    r.type = 'direct_evidence_support'
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
        """Delete existing HOP/provenance edges before a rebuild.

        build_all_hop_edges only ever adds/updates edges via MERGE keyed on
        (src_id, tgt_id) or (question_id, question_id) pairs; it never
        removes one. A one-shot build assumes it runs exactly once against a
        fresh graph, but a retried target (a prior attempt already wrote a
        full edge set before failing on an unrelated document) or an
        explicit rebuild would otherwise leave edges from the previous
        candidate selection stranded once regenerated Q-/Q+ text picks a
        slightly different candidate set.
        """
        await self.retry_query(
            f"MATCH (:{self.chunk_label})-[r:HOP_ANSWER]->(:{self.chunk_label}) DELETE r"
        )
        await self.retry_query(
            f"MATCH (:{self.q_plus_label})-[r]->() DELETE r"
        )

    async def build_all_hop_edges(self) -> None:
        """Build rank-fused HOP edges after the complete corpus is visible."""
        if not RAGConfig.ABLATION_Q_PLUS:
            logger.info("Skipping HOP edge construction because Q+ is disabled.")
            return

        await self.retry_query("CALL db.awaitIndexes($timeout_seconds)", {"timeout_seconds": 300})

        body_rows = await self.retry_query(
            f"""
            MATCH (c:{self.chunk_label})
            RETURN c.source AS source, count(c) AS count_per_source
            """
        )
        q_minus_rows = await self.retry_query(
            f"""
            MATCH (c:{self.chunk_label})-[:HAS_Q_MINUS]->(q:{self.q_minus_label})
            RETURN c.source AS source, count(q) AS count_per_source
            """
        )
        def cross_document_pools(rows: list[dict[str, Any]]) -> dict[str, int]:
            # Neo4j applies the cross-document WHERE after ANN retrieval.  A
            # source only needs to over-fetch its own representations plus L
            # retained foreign candidates.  Using the corpus-wide maximum for
            # every question made one unusually long document inflate all ANN
            # requests, which is unnecessary and expensive on large corpora.
            return {
                str(row.get("source") or ""): int(row.get("count_per_source", 0) or 0)
                + 1
                for row in rows
            }

        source_ann_pools = {
            "body": cross_document_pools(body_rows),
            "q_minus": cross_document_pools(q_minus_rows),
        }

        pool_maxima = {
            channel: max(pools.values(), default=1)
            for channel, pools in source_ann_pools.items()
        }

        page_size = max(100, int(os.environ.get("RAG_HOP_PAGE_SIZE", "5000")))
        wave_size = RAGConfig.HOP_GATHER_WAVE
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
                        channel: pools.get(str(row["source"]), 1)
                        for channel, pools in source_ann_pools.items()
                    },
                }
                for row in rows
                if row["questions"]
            ]
            page_count = len(page_items)
            if total_sources == 0:
                logger.info(
                    "build_all_hop_edges: direct-evidence rank fusion "
                    "(page=%d wave=%d per-source-ann-pool-maxima=%s).",
                    page_size,
                    wave_size,
                    pool_maxima,
                )

            for wave_start in range(0, page_count, wave_size):
                edges = await self._process_hop_wave(
                    page_items[wave_start : wave_start + wave_size],
                )
                await self._flush_hop_edges(edges)
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
