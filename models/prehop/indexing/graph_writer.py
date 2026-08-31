"""Neo4j storage for the indexing pipeline.

Owns the graph-write side: index lifecycle (vector + fulltext), document and
chunk MERGE, NEXT edge creation, and batched writes. HOP edges are delegated
to the evidence-edge builder.

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

from .chunking import _make_semantic_chunk_id, split_fixed_sentence_windows

logger = logging.getLogger(__name__)


def _make_question_id(chunk_id: str, channel: str, ordinal: int, text: str) -> str:
    payload = f"{chunk_id}|{channel}|{ordinal}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scoped_document_text(title: str, text: str) -> str:
    """Keep document/version scope in every document-side embedding."""
    return f"Document title: {title}\n{text}".strip()


def _question_record(value: Any, channel: str, source: str, sent_id: int) -> dict[str, Any]:
    if isinstance(value, str):
        text = value.strip()
        record: dict[str, Any] = {"text": text, "question_schema": "legacy"}
    elif isinstance(value, dict):
        text = str(value.get("text") or "").strip()
        record = dict(value)
        record["text"] = text
    else:
        raise TypeError(
            f"Cached/generated {channel} item must be a string or grounded object: source={source!r} sent_id={sent_id}"
        )
    if not text:
        raise ValueError(f"Cached/generated {channel} item has blank text: source={source!r} sent_id={sent_id}")
    return record


def _dedupe_question_records(values: list[Any], channel: str, source: str, sent_id: int) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, dict) and not str(value.get("text") or "").strip():
            continue
        record = _question_record(value, channel, source, sent_id)
        identity = " ".join(record["text"].casefold().split())
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(record)
    return unique


class GraphWriterMixin:
    async def setup_index(self):
        analyzer = re.sub(r"[^a-zA-Z0-9_\-]", "", RAGConfig.FULLTEXT_ANALYZER) or "english"
        vector_specs = [
            (self.body_vector_index, self.chunk_label, "embedding"),
            (self.q_minus_vector_index, self.q_minus_label, "embedding"),
            (self.q_plus_vector_index, self.q_plus_label, "embedding"),
        ]
        if RAGConfig.SENTENCE_CHANNEL_ENABLED:
            vector_specs.append((self.sentence_vector_index, self.sentence_label, "embedding"))
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
            FOR (n:{self.chunk_label}) ON EACH [n.title, n.text]
            OPTIONS {{indexConfig: {{`fulltext.analyzer`: '{analyzer}'}}}} """)
        await self.neo4j.execute_query(f"""
            CREATE FULLTEXT INDEX {self.q_minus_text_index} IF NOT EXISTS
            FOR (n:{self.q_minus_label}) ON EACH [n.title, n.text]
            OPTIONS {{indexConfig: {{`fulltext.analyzer`: '{analyzer}'}}}} """)
        await self.neo4j.execute_query(f"""
            CREATE FULLTEXT INDEX {self.q_plus_text_index} IF NOT EXISTS
            FOR (n:{self.q_plus_label}) ON EACH [n.title, n.text]
            OPTIONS {{indexConfig: {{`fulltext.analyzer`: '{analyzer}'}}}} """)
        if RAGConfig.SENTENCE_CHANNEL_ENABLED:
            await self.neo4j.execute_query(f"""
                CREATE FULLTEXT INDEX {self.sentence_text_index} IF NOT EXISTS
                FOR (n:{self.sentence_label}) ON EACH [n.title, n.text]
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
        if RAGConfig.QUESTION_SCHEMA == "linked_v2":
            await self.neo4j.execute_query(
                f"CREATE INDEX {self.answer_anchor_label}_id_idx IF NOT EXISTS "
                f"FOR (n:{self.answer_anchor_label}) ON (n.id)"
            )
        if RAGConfig.SENTENCE_CHANNEL_ENABLED:
            await self.neo4j.execute_query(
                f"CREATE INDEX {self.sentence_label}_id_idx IF NOT EXISTS FOR (n:{self.sentence_label}) ON (n.id)"
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
                OPTIONAL MATCH (old)-[:HAS_Q_MINUS|HAS_Q_PLUS|HAS_SENTENCE]->(old_q)
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
        q_minus_items: list[tuple[int, int, dict[str, Any]]] = []
        q_plus_items: list[tuple[int, int, dict[str, Any]]] = []
        sentence_items: list[tuple[int, int, str]] = []
        for chunk_index, chunk in enumerate(chunks):
            for channel in ("q_minus", "q_plus"):
                values = chunk.get(channel, [])
                if not isinstance(values, list):
                    raise TypeError(
                        f"Cached/generated {channel} must be a list: "
                        f"source={source!r} sent_id={chunk.get('sent_id', -1)}"
                    )
            sent_id = int(chunk.get("sent_id", -1))
            q_minus = _dedupe_question_records(chunk.get("q_minus", []), "q_minus", source, sent_id)
            q_plus = _dedupe_question_records(chunk.get("q_plus", []), "q_plus", source, sent_id)
            if RAGConfig.ABLATION_Q_MINUS:
                q_minus_items.extend((chunk_index, ordinal, record) for ordinal, record in enumerate(q_minus))
            if RAGConfig.ABLATION_Q_PLUS:
                q_plus_items.extend((chunk_index, ordinal, record) for ordinal, record in enumerate(q_plus))
            if RAGConfig.SENTENCE_CHANNEL_ENABLED:
                sentence_items.extend(
                    (chunk_index, ordinal, sentence)
                    for ordinal, sentence in enumerate(
                        split_fixed_sentence_windows(str(chunk.get("text", "") or ""), chunk_sentences=1)
                    )
                )

        q_minus_document_texts = [
            _scoped_document_text(str(chunks[chunk_index].get("title", "") or ""), record["text"])
            for chunk_index, _ordinal, record in q_minus_items
        ]
        q_plus_document_texts = [
            _scoped_document_text(str(chunks[chunk_index].get("title", "") or ""), record["text"])
            for chunk_index, _ordinal, record in q_plus_items
        ]
        q_plus_query_texts = [record["text"] for _chunk_index, _ordinal, record in q_plus_items]
        sentence_document_texts = [
            _scoped_document_text(str(chunks[chunk_index].get("title", "") or ""), sentence)
            for chunk_index, _ordinal, sentence in sentence_items
        ]

        body_embeds, q_minus_embeds, q_plus_embeds, q_plus_query_embeds, sentence_embeds = await asyncio.gather(
            self._embed_sparse_texts(body_texts),
            self._embed_sparse_texts(q_minus_document_texts),
            self._embed_sparse_texts(q_plus_document_texts),
            self._embed_sparse_texts(q_plus_query_texts, encoding_type="query"),
            self._embed_sparse_texts(sentence_document_texts),
        )

        for stage, embeddings, expected in (
            ("body", body_embeds, len(body_texts)),
            ("Q-", q_minus_embeds, len(q_minus_items)),
            ("Q+ document", q_plus_embeds, len(q_plus_items)),
            ("Q+ query", q_plus_query_embeds, len(q_plus_items)),
            ("sentence", sentence_embeds, len(sentence_items)),
        ):
            if len(embeddings) != expected or any(len(embedding) != self.vector_dimensions for embedding in embeddings):
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
        for flat_index, (chunk_index, ordinal, record) in enumerate(q_minus_items):
            chunk = chunks[chunk_index]
            chunk_id = _make_semantic_chunk_id(source, chunk["title"], chunk["sent_id"])
            q_minus_by_chunk[chunk_index].append(
                {
                    "id": _make_question_id(chunk_id, "q_minus", ordinal, record["text"]),
                    "text": record["text"],
                    "ordinal": ordinal,
                    "source": source,
                    "title": chunk["title"],
                    "embedding": q_minus_embeds[flat_index],
                    "question_schema": record.get("question_schema", "legacy"),
                    "grounding_quote": record.get("grounding_quote"),
                    "anchor_entities": record.get("anchor_entities", []),
                    "answer": record.get("answer"),
                    "continuation_anchor": record.get("continuation_anchor"),
                }
            )

        q_plus_by_chunk: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(chunks))}
        for flat_index, (chunk_index, ordinal, record) in enumerate(q_plus_items):
            chunk = chunks[chunk_index]
            chunk_id = _make_semantic_chunk_id(source, chunk["title"], chunk["sent_id"])
            q_plus_by_chunk[chunk_index].append(
                {
                    "id": _make_question_id(chunk_id, "q_plus", ordinal, record["text"]),
                    "text": record["text"],
                    "ordinal": ordinal,
                    "source": source,
                    "title": chunk["title"],
                    "embedding": q_plus_embeds[flat_index],
                    "query_embedding": q_plus_query_embeds[flat_index],
                    "question_schema": record.get("question_schema", "legacy"),
                    "grounding_quote": record.get("grounding_quote"),
                    "anchor_entities": record.get("anchor_entities", []),
                    "missing_information": record.get("missing_information"),
                }
            )

        sentences_by_chunk: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(chunks))}
        for flat_index, (chunk_index, ordinal, sentence) in enumerate(sentence_items):
            chunk = chunks[chunk_index]
            chunk_id = _make_semantic_chunk_id(source, chunk["title"], chunk["sent_id"])
            sentences_by_chunk[chunk_index].append(
                {
                    "id": _make_question_id(chunk_id, "sentence", ordinal, sentence),
                    "text": sentence,
                    "ordinal": ordinal,
                    "source": source,
                    "title": chunk["title"],
                    "embedding": sentence_embeds[flat_index],
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
                    "author": chunk.get("author"),
                    "publisher": chunk.get("publisher"),
                    "published_at": chunk.get("published_at"),
                    "category": chunk.get("category"),
                    "url": chunk.get("url"),
                    "embedding": body_embedding,
                    "q_minus": q_minus_by_chunk[index],
                    "q_plus": q_plus_by_chunk[index],
                    "sentences": sentences_by_chunk[index],
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
            await self.retry_query(
                f"""
                UNWIND $documents AS document
                CALL (document) {{
                    MERGE (d:{self.doc_label} {{filename: document.doc_id}})
                    SET d.title = document.doc_title, d.updated_at = timestamp()
                    WITH d, document
                    OPTIONAL MATCH (d)-[:CONTAINS]->(old:{self.chunk_label})
                    OPTIONAL MATCH (old)-[:HAS_Q_MINUS|HAS_Q_PLUS|HAS_SENTENCE]->(old_q)
                    WITH d, document,
                         collect(DISTINCT old_q) AS old_questions,
                         collect(DISTINCT old) AS old_chunks
                    FOREACH (q IN old_questions | DETACH DELETE q)
                    FOREACH (c IN old_chunks | DETACH DELETE c)
                    WITH d, document
                    UNWIND document.data AS item
                    MERGE (c:{self.chunk_label} {{id: item.id}})
                    SET c.text = item.text, c.source = item.source,
                        c.title = item.title,
                        c.sent_id = item.sent_id, c.page = item.page,
                        c.author = item.author, c.publisher = item.publisher,
                        c.published_at = item.published_at,
                        c.category = item.category, c.url = item.url,
                        c.embedding = item.embedding
                    MERGE (d)-[:CONTAINS]->(c)
                    FOREACH (question IN item.q_minus |
                        MERGE (q:{self.q_minus_label} {{id: question.id}})
                        SET q.text = question.text, q.ordinal = question.ordinal,
                            q.source = question.source, q.title = question.title,
                            q.embedding = question.embedding,
                            q.question_schema = question.question_schema,
                            q.grounding_quote = question.grounding_quote,
                            q.anchor_entities = question.anchor_entities,
                            q.answer = question.answer,
                            q.continuation_anchor = question.continuation_anchor
                        MERGE (c)-[:HAS_Q_MINUS]->(q)
                    )
                    FOREACH (question IN item.q_plus |
                        MERGE (q:{self.q_plus_label} {{id: question.id}})
                        SET q.text = question.text, q.ordinal = question.ordinal,
                            q.source = question.source, q.title = question.title,
                            q.embedding = question.embedding,
                            q.query_embedding = question.query_embedding,
                            q.question_schema = question.question_schema,
                            q.grounding_quote = question.grounding_quote,
                            q.anchor_entities = question.anchor_entities,
                            q.missing_information = question.missing_information
                        MERGE (c)-[:HAS_Q_PLUS]->(q)
                    )
                    FOREACH (sentence IN item.sentences |
                        MERGE (s:{self.sentence_label} {{id: sentence.id}})
                        SET s.text = sentence.text, s.ordinal = sentence.ordinal,
                            s.source = sentence.source, s.title = sentence.title,
                            s.embedding = sentence.embedding
                        MERGE (c)-[:HAS_SENTENCE]->(s)
                    )
                    RETURN count(c) AS chunks_written
                }}
                CALL (document) {{
                    UNWIND range(0, size(document.data) - 2) AS i
                    MATCH (c1:{self.chunk_label} {{id: document.data[i].id}})
                    MATCH (c2:{self.chunk_label} {{id: document.data[i + 1].id}})
                    MERGE (c1)-[:NEXT]->(c2)
                    RETURN count(*) AS next_edges_written
                }}
                RETURN count(document) AS documents_written
                """,
                {"documents": current_batch},
            )
        except Exception:
            # MERGE makes replay safe. Restore the whole wave, including items
            # already written before the failure, so a later flush cannot
            # silently lose documents when one Neo4j write exhausts retries.
            self._pending_batch = current_batch + self._pending_batch
            raise

        # Evidence edges are built only after every document is visible.
