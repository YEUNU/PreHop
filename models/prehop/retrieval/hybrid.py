"""Two-stage hybrid retrieval with RRF (paper §3.2.3).

Each retrieval round combines vector and fulltext search via reciprocal rank
fusion: RRF(d) = w_v / (k + r_v(d)) + w_t / (k + r_t(d))
with k = RAGConfig.RRF_K_CONSTANT (60), w_v = RAGConfig.RRF_VECTOR_WEIGHT,
w_t = RAGConfig.RRF_TEXT_WEIGHT.

Q-/Q+ indices contain individual question nodes. Search results are mapped
back to their owner chunks before RRF, so several matching questions from one
chunk cannot consume several result slots.
"""

import asyncio
from typing import Any

from core.config import RAGConfig


class HybridSearchMixin:
    async def _run_channel_query(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        async with self.neo4j.driver.session() as session:
            result = await session.run(query, params)  # type: ignore
            return [dict(record) async for record in result]

    def _channel_index_names(self, channel: str) -> tuple[str, str]:
        if channel == "q_minus":
            return self.q_minus_vector_index, self.q_minus_text_index
        if channel == "q_plus":
            return self.q_plus_vector_index, self.q_plus_text_index
        return self.body_vector_index, self.body_text_index

    async def _hybrid_rrf_candidates(self, query: str, limit: int, channel: str = "body") -> list[dict[str, Any]]:
        embed = await self.llm.get_embedding(query)
        if not embed:
            raise ValueError(f"Hybrid candidate collection: empty query embedding for query={query!r}")

        vector_index, text_index = self._channel_index_names(channel)
        question_relationship = {
            "q_minus": "HAS_Q_MINUS",
            "q_plus": "HAS_Q_PLUS",
        }.get(channel)

        safe_query = self._sanitize_fulltext_query(query)
        fulltext_query = safe_query or self._normalize_entity_term(query) or str(query or "")

        if question_relationship:
            query_vec = f"""
                CALL db.index.vector.queryNodes('{vector_index}', $limit, $embedding)
                YIELD node, score
                MATCH (owner:{self.chunk_label})-[:{question_relationship}]->(node)
                WITH owner, max(score) AS score
                RETURN owner.id AS id, owner.title AS title,
                       owner.sent_id AS sent_id, owner.page AS page,
                       owner.text AS text, owner.source AS source,
                       score AS score, 'vector' AS type, $channel AS channel
            """
            query_ft = f"""
                CALL db.index.fulltext.queryNodes('{text_index}', $query, {{limit: $limit}})
                YIELD node, score
                MATCH (owner:{self.chunk_label})-[:{question_relationship}]->(node)
                WITH owner, max(score) AS score
                RETURN owner.id AS id, owner.title AS title,
                       owner.sent_id AS sent_id, owner.page AS page,
                       owner.text AS text, owner.source AS source,
                       score AS score, 'text' AS type, $channel AS channel
            """
        else:
            query_vec = f"""
                CALL db.index.vector.queryNodes('{vector_index}', $limit, $embedding)
                YIELD node, score
                RETURN node.id AS id, node.title AS title,
                       node.sent_id AS sent_id, node.page AS page,
                       node.text AS text, node.source AS source,
                       score AS score, 'vector' AS type, $channel AS channel
            """
            query_ft = f"""
                CALL db.index.fulltext.queryNodes('{text_index}', $query, {{limit: $limit}})
                YIELD node, score
                RETURN node.id AS id, node.title AS title,
                       node.sent_id AS sent_id, node.page AS page,
                       node.text AS text, node.source AS source,
                       score AS score, 'text' AS type, $channel AS channel
            """

        # Independent read queries against different indexes; run each on its
        # own session so they execute concurrently instead of sharing one
        # session's serialized request/response cycle.
        vector_nodes, text_nodes = await asyncio.gather(
            self._run_channel_query(
                query_vec, {"limit": RAGConfig.VECTOR_SEARCH_LIMIT, "embedding": embed, "channel": channel}
            ),
            self._run_channel_query(
                query_ft, {"query": fulltext_query, "limit": RAGConfig.TEXT_SEARCH_LIMIT, "channel": channel}
            ),
        )

        all_nodes: dict[str, dict[str, Any]] = {}
        self._rrf_accumulate(all_nodes, vector_nodes, "rrf_score", RAGConfig.RRF_VECTOR_WEIGHT)
        self._rrf_accumulate(all_nodes, text_nodes, "rrf_score", RAGConfig.RRF_TEXT_WEIGHT)

        nodes = sorted(
            all_nodes.values(),
            key=lambda item: item.get("rrf_score", 0.0),
            reverse=True,
        )
        return nodes[:limit]
