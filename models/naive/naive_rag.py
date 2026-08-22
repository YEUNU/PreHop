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

    async def index_document(self, filename: str, content: str):
        if not self._index_ready:
            await self.setup_index()
            self._index_ready = True

        title, chunks = self._parse_document(filename, content)
        if not chunks:
            raise ValueError(f"No indexable chunks found for {filename!r}")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = await self.vllm.get_embeddings(texts)
        if len(embeddings) != len(chunks) or any(not embedding for embedding in embeddings):
            raise ValueError(
                f"NaiveRAG embedding failure for {title!r}: "
                f"expected {len(chunks)} non-empty vectors, got {len(embeddings)}"
            )

        batch_data = []
        for chunk, emb in zip(chunks, embeddings):
            i = chunk["sent_id"]
            # Namespace by corpus + ablation profile to avoid cross-branch collisions in Neo4j.
            chunk_id = hashlib.md5(f"naive|{self.branch_namespace}|{filename}|{title}|{i}".encode()).hexdigest()
            batch_data.append(
                {
                    "id": chunk_id,
                    "text": chunk["text"],
                    "source": filename,
                    "title": title,
                    "sent_id": i,
                    "page": chunk["page"],
                    "embedding": emb,
                }
            )

        async with self._lock, self.neo4j.driver.session() as session:
            query = f"""
                    UNWIND $batch AS item
                    MERGE (c:{self.chunk_label} {{id: item.id}})
                    SET c.text = item.text,
                        c.source = item.source,
                        c.title = item.title,
                        c.sent_id = item.sent_id,
                        c.page = item.page,
                        c.embedding = item.embedding
                """
            await session.run(query, batch=batch_data)  # type: ignore

        self.logger.info("NaiveRAG: indexed %d fixed-window chunks for %s", len(batch_data), title)

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
