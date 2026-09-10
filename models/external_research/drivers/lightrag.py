from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from core.benchmark_failures import BenchmarkIntegrityError

from .base import canonical_semantic_env, load_rows


class LightRAGDriver:
    """Pinned LightRAG mix-mode retrieval and native answer generation."""

    def __init__(self, official_root: Path, output_dir: Path):
        sys.path.insert(0, str(official_root))
        import numpy as np
        from lightrag import LightRAG, QueryParam
        from lightrag.llm.openai import openai_complete_if_cache
        from lightrag.utils import EmbeddingFunc
        from tenacity import stop_after_attempt

        from core.inference_transport import InferenceTransport
        from core.strategy_registry import get_strategy
        from core.vllm_client import VLLMClient

        transport = InferenceTransport.resolve("lightrag")
        registry_policy = dict(get_strategy("lightrag").paper_index_policy)
        embedding_dim = transport.embedding_dimensions
        embedding_model = transport.embedding_model
        embedding_base = transport.embedding_base_url
        generation_model = transport.generation_model
        generation_base = transport.generation_base_url
        embedding_client = VLLMClient(generation_model)
        embedding_client.embed_url = embedding_base
        embedding_client.embed_model_name = embedding_model
        retry_with = getattr(openai_complete_if_cache, "retry_with", None)
        if not callable(retry_with):
            raise TypeError("Pinned LightRAG completion API does not expose its typed retry hook")
        complete_with_typed_retry = retry_with(stop=stop_after_attempt(transport.retry_attempts))

        async def llm(prompt, system_prompt=None, history_messages=None, **kwargs):
            if transport.generation_seed is not None:
                kwargs.setdefault("seed", transport.generation_seed)
            return await complete_with_typed_retry(
                model=generation_model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                base_url=generation_base,
                api_key=transport.api_key,
                timeout=transport.timeout_seconds,
                **kwargs,
            )

        async def embed(texts, context="document", **_kwargs):
            vectors = await embedding_client.get_embeddings(
                texts, encoding_type="query" if context == "query" else "document"
            )
            result = np.asarray(vectors, dtype=np.float32)
            if result.ndim != 2 or result.shape != (len(texts), embedding_dim) or not np.isfinite(result).all():
                raise ValueError("LightRAG embedding callback returned an invalid ndarray")
            return result

        embedder = EmbeddingFunc(
            model_name=embedding_model,
            send_dimensions=False,
            embedding_dim=embedding_dim,
            max_token_size=int(canonical_semantic_env("RAG_EMBEDDING_MAX_TOKENS", registry_policy["embedding_max_token_size"])),
            supports_asymmetric=True,
            func=embed,
        )
        self.rows = load_rows(output_dir)
        self.by_id = {row["source_id"]: row for row in self.rows}
        self.param = QueryParam(
            mode=str(canonical_semantic_env("RAG_LIGHTRAG_QUERY_MODE", registry_policy["query_mode"])),
            top_k=int(canonical_semantic_env("RAG_LIGHTRAG_TOP_K", registry_policy["retrieval_top_k"])),
            chunk_top_k=int(canonical_semantic_env("RAG_LIGHTRAG_CHUNK_TOP_K", registry_policy["chunk_top_k"])),
            stream=False,
        )
        self.engine = LightRAG(
            working_dir=str(output_dir / "artifacts"),
            llm_model_func=llm,
            llm_model_name=generation_model,
            llm_model_max_async=transport.generation_concurrency,
            embedding_func=embedder,
            embedding_batch_num=transport.embedding_batch_size,
            embedding_func_max_async=transport.embedding_concurrency,
            **self._producer_options(),
        )
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self.engine.initialize_storages())

    @staticmethod
    def _producer_options():
        from core.execution_profile import execution_profile
        settings = execution_profile()['settings']
        return ({'max_parallel_insert': settings['lightrag_document_concurrency']}
                if 'lightrag_document_concurrency' in settings else {})

    def index(self) -> dict[str, Any]:
        track_id = self.loop.run_until_complete(
            self.engine.ainsert(
                [f"{row['title']}\n{row['text']}" for row in self.rows],
                ids=[row["source_id"] for row in self.rows],
                file_paths=[row["source_id"] for row in self.rows],
            )
        )
        statuses = self.loop.run_until_complete(self.engine.aget_docs_by_track_id(track_id))
        if len(statuses) != len(self.rows) or any(str(getattr(status, "status", "")).lower().split(".")[-1] != "processed" for status in statuses.values()):
            raise RuntimeError("LightRAG did not mark every staged source as processed")
        return {"source_count": len(self.rows), "coverage_complete": True, "query_mode": self.param.mode,
                "native_top_k": self.param.top_k, "document_concurrency": self.engine.max_parallel_insert}

    def query(self, question: str) -> dict[str, Any]:
        return self.loop.run_until_complete(self._query_async(question))

    def query_batch(self, questions):
        async def run():
            return await asyncio.gather(*(self._query_async(q) for q in questions), return_exceptions=True)
        return self.loop.run_until_complete(run())

    async def _query_async(self, question):
        from copy import deepcopy
        result = await self.engine.aquery_llm(question, deepcopy(self.param))
        if not isinstance(result, dict) or result.get("status") != "success":
            raise RuntimeError(f"LightRAG query failed: {(result or {}).get('message', 'malformed response')}")
        chunks = (result.get("data") or {}).get("chunks")
        if not isinstance(chunks, list):
            raise TypeError("LightRAG query omitted structured chunks")
        documents = []
        seen: set[str] = set()
        for chunk in chunks:
            source_id = str(chunk.get("file_path", ""))
            row = self.by_id.get(source_id)
            if row is None:
                raise BenchmarkIntegrityError("LightRAG returned a foreign or missing file_path source identity")
            if source_id not in seen:
                documents.append(
                    {"source_id": source_id, "title": row["title"], "text": chunk.get("content", row["text"])}
                )
                seen.add(source_id)
        answer = (result.get("llm_response") or {}).get("content")
        return {"documents": documents, "answer": answer}

    def close(self) -> None:
        if self.loop.is_closed():
            return
        pending = ()
        async def drain():
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        try:
            self.loop.run_until_complete(self.engine.finalize_storages())
        finally:
            # Native limiter workers outlive finalize_storages. They belong to
            # this adapter's private loop, and must be joined before closing it.
            # Snapshot before wait_for creates its own tasks (Python 3.10 has
            # a separate timeout supervisor which must not cancel itself).
            pending = tuple(asyncio.all_tasks(self.loop))
            self.loop.run_until_complete(asyncio.wait_for(drain(), timeout=10))
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()
