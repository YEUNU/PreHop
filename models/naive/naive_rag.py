import asyncio
import hashlib
import logging
import re

from core.config import RAGConfig
from core.neo4j_service import Neo4jService
from core.vllm_client import VLLMClient
from models.prehop.indexing.chunking import parse_pages_offline, split_fixed_sentence_windows
from utils.prompts.shared import build_answer_prompt


class NaiveRAG:
    """
    [Baseline] Standard RAG implementation for comparison.
    - The same fixed sentence-window chunking as Prehop.
    - Standard Vector Search.
    """

    def __init__(self, strategy: str = "naive", corpus_tag: str | None = "default"):
        self.logger = logging.getLogger(__name__)
        self.strategy = strategy.lower()
        self.prefix = self.strategy[:2].upper() + "_"
        self.corpus_tag = corpus_tag or "default"
        branch_token = self._safe_token(self.corpus_tag)
        self.chunk_label = f"{self.prefix}{branch_token}_Chunk"
        self.vector_index = f"{self.strategy}_{branch_token}_vector_idx"
        self.branch_namespace = self.corpus_tag

        self.neo4j = Neo4jService()
        self.vllm = VLLMClient()
        self._index_ready = False
        self._lock = asyncio.Lock()
        self._index_setup_lock = asyncio.Lock()

    @staticmethod
    def _safe_token(value: str) -> str:
        token = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
        token = re.sub(r"_+", "_", token).strip("_")
        return token or "default"

    async def setup_index(self):
        try:
            await self.neo4j.execute_query(
                f"""
                CREATE VECTOR INDEX {self.vector_index} IF NOT EXISTS
                FOR (n:{self.chunk_label}) ON (n.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: $dimensions,
                    `vector.similarity_function`: 'cosine'
                }}}}
            """,
                {"dimensions": RAGConfig.EMBEDDING_DIMENSIONS},
            )
            # Property index for MERGE/MATCH performance
            await self.neo4j.execute_query(
                f"CREATE INDEX {self.chunk_label}_id_idx IF NOT EXISTS FOR (n:{self.chunk_label}) ON (n.id)"
            )
        except Exception as e:
            # EquivalentSchemaRuleAlreadyExists: race condition when multiple workers
            # concurrently hit IF NOT EXISTS — index already exists, safe to ignore.
            if "EquivalentSchemaRuleAlreadyExists" in str(e) or "equivalent index already exists" in str(e).lower():
                self.logger.debug(f"Index already exists (race condition, ignored): {self.vector_index}")
            else:
                raise

    async def _ensure_index_ready(self) -> None:
        if self._index_ready:
            return
        async with self._index_setup_lock:
            if self._index_ready:
                return
            await self.setup_index()
            self._index_ready = True

    async def reconcile_dataset_files(self, filenames: list[str]) -> int:
        """Remove chunks whose source file is absent from this dataset snapshot."""
        removed = 0
        while True:
            rows = await self.neo4j.execute_query(
                f"""
                MATCH (c:{self.chunk_label})
                WHERE NOT c.source IN $filenames
                WITH c LIMIT 1000
                DETACH DELETE c
                RETURN count(c) AS deleted
                """,
                {"filenames": filenames},
            )
            deleted = int(rows[0].get("deleted", 0) or 0) if rows else 0
            removed += deleted
            if deleted == 0:
                break
        if removed:
            self.logger.info("NaiveRAG: removed %d stale chunk(s) before indexing.", removed)
        return removed

    @staticmethod
    def _parse_document(filename: str, content: str) -> tuple[str, list[dict]]:
        parsed = parse_pages_offline(filename, content)
        chunks: list[dict] = []
        sent_id = 0
        for page in parsed["pages"]:
            for text in split_fixed_sentence_windows(page["content"]):
                chunks.append({"text": text, "page": page["num"], "sent_id": sent_id})
                sent_id += 1
        return parsed["title"], chunks

    async def index_documents(self, documents: list[tuple[str, str]]) -> int:
        """Embed and atomically replace a batch of source documents."""
        if not documents:
            return 0
        prepared: list[tuple[str, str, list[dict]]] = []
        texts: list[str] = []
        for filename, content in documents:
            title, chunks = self._parse_document(filename, content)
            if not chunks:
                raise ValueError(f"No indexable chunks found for {filename!r}")
            prepared.append((filename, title, chunks))
            # Keep document/version scope in the vector representation.
            texts.extend(f"Document title: {title}\n{chunk['text']}" for chunk in chunks)

        # Flattening across source files is essential for short-document
        # short-document corpora usually have one chunk per file, so per-document
        # calls silently turned a configured batch size of 32 into 66k
        # one-item requests.
        embeddings = await self.vllm.get_embeddings(texts)
        if len(embeddings) != len(texts) or any(
            len(embedding) != RAGConfig.EMBEDDING_DIMENSIONS for embedding in embeddings
        ):
            raise ValueError(
                "NaiveRAG batch embedding failure: "
                f"expected {len(texts)} vectors of dimension {RAGConfig.EMBEDDING_DIMENSIONS}, "
                f"got {len(embeddings)}"
            )

        # Validate the external embedding shape before creating a fixed-size
        # vector index, then serialize schema creation across file tasks.
        await self._ensure_index_ready()

        batch_data = []
        embedding_index = 0
        for filename, title, chunks in prepared:
            for chunk in chunks:
                embedding = embeddings[embedding_index]
                embedding_index += 1
                sent_id = chunk["sent_id"]
                chunk_id = hashlib.md5(
                    f"naive|{self.branch_namespace}|{filename}|{title}|{sent_id}".encode()
                ).hexdigest()
                batch_data.append(
                    {
                        "id": chunk_id,
                        "text": chunk["text"],
                        "source": filename,
                        "title": title,
                        "sent_id": sent_id,
                        "page": chunk["page"],
                        "embedding": embedding,
                    }
                )

        async with self._lock, self.neo4j.driver.session() as session:
            query = f"""
                    OPTIONAL MATCH (old:{self.chunk_label})
                    WHERE old.source IN $sources
                    WITH collect(old) AS old_chunks, $batch AS replacement
                    FOREACH (old IN old_chunks | DETACH DELETE old)
                    WITH replacement
                    UNWIND replacement AS item
                    MERGE (c:{self.chunk_label} {{id: item.id}})
                    SET c.text = item.text,
                        c.source = item.source,
                        c.title = item.title,
                        c.sent_id = item.sent_id,
                        c.page = item.page,
                        c.embedding = item.embedding
                """
            result = await session.run(  # type: ignore
                query,
                batch=batch_data,
                sources=[filename for filename, _title, _chunks in prepared],
            )
            await result.consume()

        self.logger.info(
            "NaiveRAG: indexed %d fixed-window chunks for %d documents in one embedding/write batch",
            len(batch_data),
            len(prepared),
        )
        return len(prepared)

    async def index_document(self, filename: str, content: str):
        """Compatibility wrapper for one-document callers and live smoke tests."""
        await self.index_documents([(filename, content)])

    async def retrieve(self, query: str, top_k: int = RAGConfig.DEFAULT_TOP_K) -> tuple:
        query_embedding = await self.vllm.get_embedding(query)
        if not query_embedding:
            raise ValueError(f"NaiveRAG query embedding is missing for query={query!r}")

        async with self.neo4j.driver.session() as session:
            cypher_query = f"""
                CALL db.index.vector.queryNodes('{self.vector_index}', $k, $embedding)
                YIELD node, score
                RETURN node.text as text, node.title as title, node.sent_id as sent_id, node.page as page, score
            """
            result = await session.run(
                cypher_query,
                {  # type: ignore
                    "k": top_k,
                    "embedding": query_embedding,
                },
            )

            nodes = [dict(rec) async for rec in result]

        context_parts = [f"[[{n['title']}, {n['sent_id']}]]\n{n['text']}" for n in nodes]
        return "\n\n---\n\n".join(context_parts), nodes

    async def run_workflow(self, query: str, history: list[dict] | None = None) -> tuple:
        """Entry point for benchmark. Returns (answer, sources, trace)."""
        _ = history
        # Naive RAG has no upstream/official top-k to preserve. Use the same
        # synthesis-context budget as Prehop so the benchmark comparison does
        # not give the two in-repo methods different evidence counts by default.
        context, nodes = await self.retrieve(query, top_k=RAGConfig.DEFAULT_TOP_K)
        if not context:
            return "Insufficient evidence.", [], [{"step": "naive_qa", "output": "empty_context"}]

        prompt = build_answer_prompt(context, query)
        messages = [{"role": "user", "content": prompt}]

        answer = await self.vllm.generate_response(messages)
        if not str(answer or "").strip():
            raise ValueError("Answer synthesis returned an empty response")
        # Format sources for metric evaluation: {"doc": title, "page": page, "text": text, "sent_id": sent_id}
        trace = [{"step": "naive_qa", "input": messages, "output": answer}]
        sources = [
            {"doc": n["title"], "page": n.get("page", 0), "text": n["text"], "sent_id": n["sent_id"]} for n in nodes
        ]
        return answer, sources, trace
