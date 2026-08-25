"""Hybrid retrieval with unweighted reciprocal-rank fusion.

Each representation combines vector and full-text ranks as
``1 / (rank + 1)``. Both modalities contribute equally and there is no
query-time fusion parameter.

Q-/Q+ indices contain individual question nodes. Search results are mapped
back to their owner chunks before RRF, so several matching questions from one
chunk cannot consume several result slots.
"""

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

    async def _hybrid_rrf_candidates(
        self,
        query: str,
        query_embedding: list[float],
        limit: int,
        channel: str = "body",
    ) -> list[dict[str, Any]]:
        if not query_embedding:
            raise ValueError(f"Hybrid candidate collection: empty query embedding for query={query!r}")
        retained_limit = max(1, int(limit))

        vector_index, text_index = self._channel_index_names(channel)
        question_relationship = {
            "q_minus": "HAS_Q_MINUS",
            "q_plus": "HAS_Q_PLUS",
        }.get(channel)
        # Q-/Q+ ANN/full-text results are question nodes and are collapsed to
        # owner chunks after search. Indexing admits at most three questions
        # per direction, so use that schema bound—not a tuned search knob—to
        # make the requested owner-candidate budget attainable.
        representation_limit = retained_limit * (
            RAGConfig.QUESTIONS_PER_DIRECTION if question_relationship else 1
        )

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
                       owner.embedding AS embedding,
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
                       owner.embedding AS embedding,
                       score AS score, 'text' AS type, $channel AS channel
            """
        else:
            query_vec = f"""
                CALL db.index.vector.queryNodes('{vector_index}', $limit, $embedding)
                YIELD node, score
                RETURN node.id AS id, node.title AS title,
                       node.sent_id AS sent_id, node.page AS page,
                       node.text AS text, node.source AS source,
                       node.embedding AS embedding,
                       score AS score, 'vector' AS type, $channel AS channel
            """
            query_ft = f"""
                CALL db.index.fulltext.queryNodes('{text_index}', $query, {{limit: $limit}})
                YIELD node, score
                RETURN node.id AS id, node.title AS title,
                       node.sent_id AS sent_id, node.page AS page,
                       node.text AS text, node.source AS source,
                       node.embedding AS embedding,
                       score AS score, 'text' AS type, $channel AS channel
            """

        # One database round trip per representation. The enclosing caller
        # still searches all enabled representations concurrently.
        combined_query = f"""
            CALL () {{
                {query_vec}
                UNION ALL
                {query_ft}
            }}
            RETURN *
        """
        rows = await self._run_channel_query(
            combined_query,
            {
                "limit": representation_limit,
                "embedding": query_embedding,
                "query": fulltext_query,
                "channel": channel,
            },
        )
        vector_nodes = [node for node in rows if node.get("type") == "vector"]
        text_nodes = [node for node in rows if node.get("type") == "text"]

        all_nodes: dict[str, dict[str, Any]] = {}
        self._rrf_accumulate(all_nodes, vector_nodes, "rrf_score")
        self._rrf_accumulate(all_nodes, text_nodes, "rrf_score")

        nodes = sorted(
            all_nodes.values(),
            key=lambda item: (-float(item.get("rrf_score", 0.0)), self._node_identity(item)),
        )
        return nodes[:retained_limit]
