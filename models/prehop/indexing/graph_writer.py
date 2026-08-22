"""Neo4j storage for the indexing pipeline.

Owns the graph-write side: index lifecycle (vector + fulltext), document and
chunk MERGE, NEXT edge creation, and batched writes. HOP edges are delegated
to HopEdgeMixin (paper §3.1.4).

Q- and Q+ are stored as individual question nodes rather than concatenated
chunk properties.  This preserves multiple independent directions emitted by
one chunk and makes every question-level HOP decision inspectable.
"""

import asyncio
import hashlib
import logging
import random
import re
from typing import Any

from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

from core.config import RAGConfig

from .chunking import _make_semantic_chunk_id

logger = logging.getLogger(__name__)


def _make_question_id(chunk_id: str, channel: str, ordinal: int, text: str) -> str:
    payload = f"{chunk_id}|{channel}|{ordinal}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scoped_document_text(title: str, text: str) -> str:
    """Keep document/version scope in every document-side embedding."""
    return f"Document title: {title}\n{text}".strip()


class GraphWriterMixin:
    async def setup_index(self):
        analyzer = re.sub(r"[^a-zA-Z0-9_\-]", "", RAGConfig.FULLTEXT_ANALYZER) or "english"
        vector_specs = [
            (self.body_vector_index, self.chunk_label, "embedding"),
            (self.q_minus_vector_index, self.q_minus_label, "embedding"),
            (self.q_plus_vector_index, self.q_plus_label, "embedding"),
        ]
        for index_name, label, property_name in vector_specs:
            await self.neo4j.execute_query(
                f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:{label}) ON (n.{property_name})
                OPTIONS {{indexConfig: {{`vector.dimensions`: $dimensions, `vector.similarity_function`: 'cosine'}}}} """,
                {"dimensions": self.vector_dimensions},
            )

        await self.neo4j.execute_query(f"""
            CREATE FULLTEXT INDEX {self.body_text_index} IF NOT EXISTS
            FOR (n:{self.chunk_label}) ON EACH [n.title, n.text, n.chunk_summary]
            OPTIONS {{indexConfig: {{`fulltext.analyzer`: '{analyzer}'}}}} """)
        await self.neo4j.execute_query(f"""
            CREATE FULLTEXT INDEX {self.q_minus_text_index} IF NOT EXISTS
            FOR (n:{self.q_minus_label}) ON EACH [n.title, n.text]
            OPTIONS {{indexConfig: {{`fulltext.analyzer`: '{analyzer}'}}}} """)
        await self.neo4j.execute_query(f"""
            CREATE FULLTEXT INDEX {self.q_plus_text_index} IF NOT EXISTS
            FOR (n:{self.q_plus_label}) ON EACH [n.title, n.text]
            OPTIONS {{indexConfig: {{`fulltext.analyzer`: '{analyzer}'}}}} """)

        await self.neo4j.execute_query(
            f"CREATE INDEX {self.chunk_label}_id_idx IF NOT EXISTS FOR (n:{self.chunk_label}) ON (n.id)"
        )
        await self.neo4j.execute_query(
            f"CREATE INDEX {self.doc_label}_fn_idx IF NOT EXISTS FOR (n:{self.doc_label}) ON (n.filename)"
        )
        await self.neo4j.execute_query(
            f"CREATE INDEX {self.q_minus_label}_id_idx IF NOT EXISTS FOR (n:{self.q_minus_label}) ON (n.id)"
        )
        await self.neo4j.execute_query(
            f"CREATE INDEX {self.q_plus_label}_id_idx IF NOT EXISTS FOR (n:{self.q_plus_label}) ON (n.id)"
        )

    async def _ensure_index_ready(self):
        if self._index_ready:
            return
        async with self._index_setup_lock:
            if self._index_ready:
                return
            await self.setup_index()
            self._index_ready = True

    @staticmethod
    def _is_retryable_neo4j_error(error: Exception) -> bool:
        if isinstance(error, (TransientError, ServiceUnavailable, SessionExpired)):
            return True
        code = str(getattr(error, "code", "") or "")
        text = str(error)
        markers = [
            "DeadlockDetected",
            "Neo.TransientError",
            "TransientError",
            "ServiceUnavailable",
        ]
        return any(marker in code or marker in text for marker in markers)

    async def retry_query(self, query: str, parameters: dict[str, Any] | None = None):
        for attempt in range(self.max_retries):
            try:
                return await self.neo4j.execute_query(query, parameters)
            except Exception as error:
                if not self._is_retryable_neo4j_error(error):
                    raise
                if attempt == self.max_retries - 1:
                    raise
                delay = (RAGConfig.RETRY_DELAY * (2**attempt)) + random.uniform(0, RAGConfig.RETRY_DELAY)
                logger.warning(
                    "Neo4j transient error (attempt %d/%d), retrying in %.2fs: %s",
                    attempt + 1,
                    self.max_retries,
                    delay,
                    error,
                )
                await asyncio.sleep(delay)

    async def reconcile_dataset_files(self, filenames: list[str]) -> int:
        """Remove documents no longer present in the current input snapshot."""
        removed = 0
        while True:
            rows = await self.retry_query(
                f"""
                MATCH (d:{self.doc_label})
                WHERE NOT d.filename IN $filenames
                WITH d LIMIT 100
                OPTIONAL MATCH (d)-[:CONTAINS]->(old:{self.chunk_label})
                OPTIONAL MATCH (old)-[:HAS_Q_MINUS|HAS_Q_PLUS]->(old_q)
                WITH collect(DISTINCT d) AS old_docs,
                     collect(DISTINCT old) AS old_chunks,
                     collect(DISTINCT old_q) AS old_questions
                FOREACH (q IN old_questions | DETACH DELETE q)
                FOREACH (c IN old_chunks | DETACH DELETE c)
                FOREACH (d IN old_docs | DETACH DELETE d)
                RETURN size(old_docs) AS deleted
                """,
                {"filenames": filenames},
            )
            deleted = int(rows[0].get("deleted", 0) or 0) if rows else 0
            removed += deleted
            if deleted == 0:
                break
        if removed:
            logger.info("Removed %d stale Prehop document(s) before indexing.", removed)
        return removed

    async def build_graph(self, knowledge: dict[str, Any], source: str, document_filename: str):
        chunks = knowledge.get("chunks", [])
        if not chunks:
            raise ValueError(f"No chunks generated for source={source!r}")

        body_texts = [
            _scoped_document_text(str(chunk.get("title", "") or ""), str(chunk.get("text", "") or ""))
            for chunk in chunks
        ]
        q_minus_items: list[tuple[int, int, str]] = []
        q_plus_items: list[tuple[int, int, str]] = []
        for chunk_index, chunk in enumerate(chunks):
            for channel in ("q_minus", "q_plus"):
                values = chunk.get(channel, [])
                if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                    raise TypeError(
                        f"Cached/generated {channel} must be a list of strings: "
                        f"source={source!r} sent_id={chunk.get('sent_id', -1)}"
                    )
            q_minus = self._dedupe_preserve_order(chunk.get("q_minus", []))
            q_plus = self._dedupe_preserve_order(chunk.get("q_plus", []))
            if RAGConfig.ABLATION_Q_MINUS:
                q_minus_items.extend((chunk_index, ordinal, text) for ordinal, text in enumerate(q_minus))
            if RAGConfig.ABLATION_Q_PLUS:
                q_plus_items.extend((chunk_index, ordinal, text) for ordinal, text in enumerate(q_plus))

        q_minus_document_texts = [
            _scoped_document_text(str(chunks[chunk_index].get("title", "") or ""), text)
            for chunk_index, _ordinal, text in q_minus_items
        ]
        q_plus_document_texts = [
            _scoped_document_text(str(chunks[chunk_index].get("title", "") or ""), text)
            for chunk_index, _ordinal, text in q_plus_items
        ]
        q_plus_query_texts = [text for _chunk_index, _ordinal, text in q_plus_items]

        body_embeds, q_minus_embeds, q_plus_embeds, q_plus_query_embeds = await asyncio.gather(
            self._embed_sparse_texts(body_texts),
            self._embed_sparse_texts(q_minus_document_texts),
            self._embed_sparse_texts(q_plus_document_texts),
            self._embed_sparse_texts(q_plus_query_texts, encoding_type="query"),
        )

        for stage, embeddings, expected in (
            ("body", body_embeds, len(body_texts)),
            ("Q-", q_minus_embeds, len(q_minus_items)),
            ("Q+ document", q_plus_embeds, len(q_plus_items)),
            ("Q+ query", q_plus_query_embeds, len(q_plus_items)),
        ):
            if len(embeddings) != expected or any(
                len(embedding) != self.vector_dimensions for embedding in embeddings
            ):
                raise ValueError(
                    f"{stage} embedding validation failed for source={source!r}: "
                    f"expected {expected} vectors of dimension {self.vector_dimensions}"
                )

        # Create the fixed-dimension Neo4j schema only after the external
        # endpoint has returned vectors of the configured dimension. A bad
        # environment must fail without leaving an incompatible vector index
        # behind for the next run.
        await self._ensure_index_ready()

        q_minus_by_chunk: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(chunks))}
        for flat_index, (chunk_index, ordinal, text) in enumerate(q_minus_items):
            chunk = chunks[chunk_index]
            chunk_id = _make_semantic_chunk_id(source, chunk["title"], chunk["sent_id"])
            q_minus_by_chunk[chunk_index].append(
                {
                    "id": _make_question_id(chunk_id, "q_minus", ordinal, text),
                    "text": text,
                    "ordinal": ordinal,
                    "source": source,
                    "title": chunk["title"],
                    "embedding": q_minus_embeds[flat_index],
                }
            )

        q_plus_by_chunk: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(chunks))}
        for flat_index, (chunk_index, ordinal, text) in enumerate(q_plus_items):
            chunk = chunks[chunk_index]
            chunk_id = _make_semantic_chunk_id(source, chunk["title"], chunk["sent_id"])
            q_plus_by_chunk[chunk_index].append(
                {
                    "id": _make_question_id(chunk_id, "q_plus", ordinal, text),
                    "text": text,
                    "ordinal": ordinal,
                    "source": source,
                    "title": chunk["title"],
                    "embedding": q_plus_embeds[flat_index],
                    "query_embedding": q_plus_query_embeds[flat_index],
                }
            )

        batch_data = []
        for index, chunk in enumerate(chunks):
            body_embedding = body_embeds[index] if index < len(body_embeds) else []
            if not body_embedding:
                raise ValueError(
                    f"Missing embedding for chunk: source={source} "
                    f"title={chunk.get('title', '')} sent_id={chunk.get('sent_id', -1)}"
                )
            chunk_id = _make_semantic_chunk_id(source, chunk["title"], chunk["sent_id"])
            batch_data.append(
                {
                    "id": chunk_id,
                    "text": chunk["text"],
                    "source": source,
                    "title": chunk["title"],
                    "sent_id": chunk["sent_id"],
                    "page": chunk.get("page", 0),
                    "embedding": body_embedding,
                    "q_minus": q_minus_by_chunk[index],
                    "q_plus": q_plus_by_chunk[index],
                    "chunk_summary": chunk["summary"],
                }
            )

        async with self._batch_lock:
            self._pending_batch.append(
                {
                    "data": batch_data,
                    "doc_id": document_filename,
                    "doc_title": str(knowledge.get("title") or document_filename),
                }
            )
            if len(self._pending_batch) >= RAGConfig.NEO4J_BATCH_SIZE:
                await self._flush_graph_batch_unlocked()

    async def flush_graph_batch(self):
        async with self._batch_lock:
            await self._flush_graph_batch_unlocked()

    async def _flush_graph_batch_unlocked(self):
        if not self._pending_batch:
            return

        current_batch = self._pending_batch
        self._pending_batch = []

        try:
            for item in current_batch:
                await self.retry_query(
                    f"""
                    MERGE (d:{self.doc_label} {{filename: $doc_id}})
                    SET d.title = $doc_title, d.updated_at = timestamp()
                    WITH d
                    OPTIONAL MATCH (d)-[:CONTAINS]->(old:{self.chunk_label})
                    OPTIONAL MATCH (old)-[:HAS_Q_MINUS|HAS_Q_PLUS]->(old_q)
                    WITH d, $batch AS new_batch,
                         collect(DISTINCT old_q) AS old_questions,
                         collect(DISTINCT old) AS old_chunks
                    FOREACH (q IN old_questions | DETACH DELETE q)
                    FOREACH (c IN old_chunks | DETACH DELETE c)
                    WITH d, new_batch
                    UNWIND new_batch AS item
                    MERGE (c:{self.chunk_label} {{id: item.id}})
                    SET c.text = item.text, c.source = item.source,
                        c.title = item.title,
                        c.sent_id = item.sent_id, c.page = item.page,
                        c.embedding = item.embedding,
                        c.chunk_summary = item.chunk_summary
                    MERGE (d)-[:CONTAINS]->(c)
                    FOREACH (question IN item.q_minus |
                        MERGE (q:{self.q_minus_label} {{id: question.id}})
                        SET q.text = question.text, q.ordinal = question.ordinal,
                            q.source = question.source, q.title = question.title,
                            q.embedding = question.embedding
                        MERGE (c)-[:HAS_Q_MINUS]->(q)
                    )
                    FOREACH (question IN item.q_plus |
                        MERGE (q:{self.q_plus_label} {{id: question.id}})
                        SET q.text = question.text, q.ordinal = question.ordinal,
                            q.source = question.source, q.title = question.title,
                            q.embedding = question.embedding,
                            q.query_embedding = question.query_embedding
                        MERGE (c)-[:HAS_Q_PLUS]->(q)
                    )
                """,
                    {
                        "batch": item["data"],
                        "doc_id": item["doc_id"],
                        "doc_title": item["doc_title"],
                    },
                )

                await self.retry_query(
                    f"""
                    UNWIND range(0, size($batch)-2) AS i
                    MATCH (c1:{self.chunk_label} {{id: $batch[i].id}})
                    MATCH (c2:{self.chunk_label} {{id: $batch[i+1].id}})
                    MERGE (c1)-[:NEXT]->(c2)
                """,
                    {"batch": item["data"]},
                )
        except Exception:
            # MERGE makes replay safe. Restore the whole wave, including items
            # already written before the failure, so a later flush cannot
            # silently lose documents when one Neo4j write exhausts retries.
            self._pending_batch = current_batch + self._pending_batch
            raise

        # HOP edge construction is now a single one-shot pass at the end of
        # indexing (`build_all_hop_edges`), invoked from cli/index.py after
        # all files have been flushed. The previous per-batch call here
        # produced an asymmetric graph (early batches had only 24 other
        # docs as candidates, late batches saw the whole corpus). See
        # paper §3.1.4: "Multi-hop discovery happens once, at indexing time".
