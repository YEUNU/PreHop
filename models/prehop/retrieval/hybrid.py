"""Hybrid retrieval with unweighted reciprocal-rank fusion.

Each representation combines vector and full-text ranks as
``1 / (rank + 1)``. Both modalities contribute equally and there is no
query-time fusion parameter.

Q-/Q+ and optional sentence indices contain child representation nodes. Search
results are mapped back to their owner chunks before RRF, so several matching
children from one chunk cannot consume several result slots. For Q+, the exact
matched question IDs survive this owner collapse for provenance-scoped HOP
activation.
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
        if channel == "sentence":
            return self.sentence_vector_index, self.sentence_text_index
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
        owner_relationship = {
            "q_minus": "HAS_Q_MINUS",
            "q_plus": "HAS_Q_PLUS",
            "sentence": "HAS_SENTENCE",
        }.get(channel)
        # Child-representation results are collapsed to owner chunks after
        # search. Use each indexing schema's exact maximum children per chunk,
        # not a fitted search width, so the owner budget remains attainable.
        child_bound = {
            "q_minus": RAGConfig.QUESTIONS_PER_DIRECTION,
            "q_plus": RAGConfig.QUESTIONS_PER_DIRECTION,
            "sentence": RAGConfig.CHUNK_SENTENCES,
        }.get(channel, 1)
        representation_limit = retained_limit * child_bound

        safe_query = self._sanitize_fulltext_query(query)
        fulltext_query = safe_query or self._normalize_entity_term(query) or str(query or "")

        if owner_relationship:
            query_vec = f"""
                CALL db.index.vector.queryNodes('{vector_index}', $limit, $embedding)
                YIELD node, score
                MATCH (owner:{self.chunk_label})-[:{owner_relationship}]->(node)
                WITH owner, max(score) AS score,
                     collect(DISTINCT node.id) AS matched_question_ids
                RETURN owner.id AS id, owner.title AS title,
                       owner.sent_id AS sent_id, owner.page AS page,
                       owner.text AS text, owner.source AS source,
                       owner[$author_property] AS author, owner[$publisher_property] AS publisher,
                       owner[$published_at_property] AS published_at,
                       owner[$category_property] AS category, owner[$url_property] AS url,
                       owner.embedding AS embedding,
                       matched_question_ids,
                       score AS score, 'vector' AS type, $channel AS channel
            """
            query_ft = f"""
                CALL db.index.fulltext.queryNodes('{text_index}', $query, {{limit: $limit}})
                YIELD node, score
                MATCH (owner:{self.chunk_label})-[:{owner_relationship}]->(node)
                WITH owner, max(score) AS score,
                     collect(DISTINCT node.id) AS matched_question_ids
                RETURN owner.id AS id, owner.title AS title,
                       owner.sent_id AS sent_id, owner.page AS page,
                       owner.text AS text, owner.source AS source,
                       owner[$author_property] AS author, owner[$publisher_property] AS publisher,
                       owner[$published_at_property] AS published_at,
                       owner[$category_property] AS category, owner[$url_property] AS url,
                       owner.embedding AS embedding,
                       matched_question_ids,
                       score AS score, 'text' AS type, $channel AS channel
            """
        else:
            query_vec = f"""
                CALL db.index.vector.queryNodes('{vector_index}', $limit, $embedding)
                YIELD node, score
                RETURN node.id AS id, node.title AS title,
                       node.sent_id AS sent_id, node.page AS page,
                       node.text AS text, node.source AS source,
                       node[$author_property] AS author, node[$publisher_property] AS publisher,
                       node[$published_at_property] AS published_at,
                       node[$category_property] AS category, node[$url_property] AS url,
                       node.embedding AS embedding,
                       score AS score, 'vector' AS type, $channel AS channel
            """
            query_ft = f"""
                CALL db.index.fulltext.queryNodes('{text_index}', $query, {{limit: $limit}})
                YIELD node, score
                RETURN node.id AS id, node.title AS title,
                       node.sent_id AS sent_id, node.page AS page,
                       node.text AS text, node.source AS source,
                       node[$author_property] AS author, node[$publisher_property] AS publisher,
                       node[$published_at_property] AS published_at,
                       node[$category_property] AS category, node[$url_property] AS url,
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
                "author_property": "author",
                "publisher_property": "publisher",
                "published_at_property": "published_at",
                "category_property": "category",
                "url_property": "url",
            },
        )

        # Cypher only guarantees row order when the final result is explicitly
        # ordered.  In particular, aggregation and UNION ALL may discard the
        # order produced by the vector/full-text procedures.  RRF consumes
        # ranks rather than raw scores, so restore each modality's rank here
        # before accumulating reciprocal ranks.  Raw backend scores are used
        # only within their own modality and are never interpolated.
        def modality_order(node: dict[str, Any]) -> tuple[float, str]:
            return (-float(node.get("score", 0.0)), self._node_identity(node))

        vector_nodes = sorted(
            (node for node in rows if node.get("type") == "vector"),
            key=modality_order,
        )
        text_nodes = sorted(
            (node for node in rows if node.get("type") == "text"),
            key=modality_order,
        )

        all_nodes: dict[str, dict[str, Any]] = {}
        self._rrf_accumulate(all_nodes, vector_nodes, "rrf_score")
        self._rrf_accumulate(all_nodes, text_nodes, "rrf_score")

        # Question searches rank owner chunks, but exact graph activation needs
        # the identities of the question nodes that caused each owner to match.
        # Preserve the union across vector and lexical modalities instead of
        # silently inheriting only the first modality copied by RRF.
        if channel in {"q_minus", "q_plus"}:
            matched_by_owner: dict[str, set[str]] = {}
            for node in (*vector_nodes, *text_nodes):
                node_id = self._node_identity(node)
                matched = matched_by_owner.setdefault(node_id, set())
                matched.update(
                    str(question_id).strip()
                    for question_id in (node.get("matched_question_ids") or [])
                    if str(question_id).strip()
                )
            for node_id, node in all_nodes.items():
                field = "matched_qplus_ids" if channel == "q_plus" else "matched_qminus_ids"
                node[field] = sorted(matched_by_owner.get(node_id, set()))
                node.pop("matched_question_ids", None)
        else:
            for node in all_nodes.values():
                node.pop("matched_question_ids", None)

        nodes = sorted(
            all_nodes.values(),
            key=lambda item: (-float(item.get("rrf_score", 0.0)), self._node_identity(item)),
        )
        return nodes[:retained_limit]
