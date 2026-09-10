from __future__ import annotations

import asyncio
from typing import Any

from core.generation_profiles import request_settings
from core.vllm_client import get_llm_client
from models.official_baseline_runtime import OfficialQueryWorker
from utils.prompts.shared import build_answer_prompt, mark_answer_boundary


class ExternalResearchAdapter:
    def __init__(self, strategy: str, model_id: str = "default", corpus_tag: str = "default"):
        self.strategy = strategy
        self.corpus_tag = corpus_tag
        self.llm = None
        self.model_id = model_id
        self._worker = OfficialQueryWorker(strategy, corpus_tag)
        self._batcher = None
        if strategy in {"linear_rag", "lightrag", "gfm_rag"}:
            from .query_batching import NativeQueryBatcher
            self._batcher = NativeQueryBatcher(self._worker)


    def _documents(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        return response.get("documents")

    async def _query(self, query: str) -> dict[str, Any]:
        if self._batcher is not None:
            return await self._batcher.request(query)
        return await asyncio.to_thread(self._worker.request, {"operation": "query", "query": query})

    async def retrieve(self, query: str) -> list[dict[str, Any]]:
        response = await self._query(query)
        return self._documents(response)

    async def run_workflow(self, query: str, history: list[dict] | None = None) -> tuple[str, list, list]:
        _ = history
        response = await self._query(query)
        documents = self._documents(response)
        # The worker returns the method-native retrieval cut-off. Do not apply
        # a cross-method top-k here: it would silently change graph traversal
        # and context semantics.
        native_answer = response.get("answer")
        context_documents = documents
        context = "\n\n".join(
            f"[{idx}] {row.get('title', '')}\n{row.get('text', '')}" for idx, row in enumerate(context_documents, 1)
        )
        answer = native_answer
        if answer is None:
            if self.llm is None:
                self.llm = get_llm_client(self.model_id)
            answer = await self.llm.generate_response(
                [{"role": "user", "content": build_answer_prompt(context, query)}],
                **request_settings("answer"),
            )
        sources = [
            {
                "doc": row.get("title") or row["source_id"],
                "source": row["source_id"],
                "page": 0,
                "sent_id": 0,
                "text": row.get("text", ""),
                **({"score": row["score"]} if row.get("score") is not None else {}),
            }
            for row in documents
        ]
        return (
            mark_answer_boundary(answer),
            sources,
            [{
                "step": f"{self.strategy}_official_retrieval",
                "retrieved": len(documents),
                "worker_queue_seconds": float(response.get("worker_queue_seconds", 0.0)),
                "native_query_batch_size": int(response.get("native_query_batch_size", 1)),
                **({"native_qa_max_workers": response["native_qa_max_workers"]}
                   if response.get("native_qa_max_workers") is not None else {}),
            }],
        )

    def close(self) -> None:
        self._worker.close()
