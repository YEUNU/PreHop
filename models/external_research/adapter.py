from __future__ import annotations

import asyncio
from typing import Any

from core.generation_profiles import request_settings
from core.vllm_client import get_llm_client
from models.official_baseline_runtime import OfficialQueryWorker, verify_snapshot
from utils.prompts.shared import build_answer_prompt, mark_answer_boundary


class ExternalResearchAdapter:
    def __init__(self, strategy: str, model_id: str = "default", corpus_tag: str = "default"):
        self.strategy = strategy
        self.corpus_tag = corpus_tag
        self.llm = None
        self.model_id = model_id
        self._worker = OfficialQueryWorker(strategy, corpus_tag)

    def verify_active_snapshot(self, expected_source_ids: list[str], corpus_manifest: dict | None) -> dict:
        return verify_snapshot(self.strategy, self.corpus_tag, expected_source_ids, corpus_manifest)

    def _documents(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        documents = response.get("documents")
        if not isinstance(documents, list):
            raise TypeError(f"{self.strategy} official worker returned malformed documents")
        seen: set[str] = set()
        for row in documents:
            if not isinstance(row, dict) or not str(row.get("source_id", "")).strip():
                raise TypeError(f"{self.strategy} official worker returned a document without source_id")
            if row["source_id"] in seen:
                raise ValueError(f"{self.strategy} official worker returned duplicate source_id")
            seen.add(row["source_id"])
        return documents

    async def retrieve(self, query: str) -> list[dict[str, Any]]:
        response = await asyncio.to_thread(self._worker.request, {"operation": "query", "query": query})
        return self._documents(response)

    async def run_workflow(self, query: str, history: list[dict] | None = None) -> tuple[str, list, list]:
        _ = history
        response = await asyncio.to_thread(self._worker.request, {"operation": "query", "query": query})
        documents = self._documents(response)
        # The worker returns the method-native retrieval cut-off. Do not apply
        # a cross-method top-k here: it would silently change graph traversal
        # and context semantics.
        if not documents:
            return (
                mark_answer_boundary("Insufficient evidence."),
                [],
                [{"step": f"{self.strategy}_retrieval", "output": "empty"}],
            )
        native_answer = response.get("answer")
        context_documents = documents
        context = "\n\n".join(
            f"[{idx}] {row.get('title', '')}\n{row.get('text', '')}" for idx, row in enumerate(context_documents, 1)
        )
        answer = native_answer
        if answer is None:
            if self.strategy in {"lightrag", "hipporag2", "linear_rag", "youtu_graphrag"}:
                raise ValueError(f"{self.strategy} native query omitted its required answer")
            if self.llm is None:
                self.llm = get_llm_client(self.model_id)
            answer = await self.llm.generate_response(
                [{"role": "user", "content": build_answer_prompt(context, query)}],
                **request_settings("answer"),
            )
        if not str(answer or "").strip():
            raise ValueError(f"{self.strategy} answer synthesis returned an empty response")
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
            }],
        )

    def close(self) -> None:
        self._worker.close()
