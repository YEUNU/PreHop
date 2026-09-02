from __future__ import annotations

import asyncio
from typing import Any

from core.config import RAGConfig
from core.vllm_client import get_llm_client
from models.official_baseline_runtime import OfficialQueryWorker, verify_snapshot
from utils.prompts.shared import build_answer_prompt, mark_answer_boundary


class PropRAGAdapter:
    def __init__(self, model_id: str = "default", corpus_tag: str = "default"):
        self.corpus_tag = corpus_tag
        self.llm = get_llm_client(model_id)
        self._worker = OfficialQueryWorker("proprag", corpus_tag)

    def verify_active_snapshot(self, expected_source_ids: list[str], corpus_manifest: dict | None) -> dict:
        return verify_snapshot("proprag", self.corpus_tag, expected_source_ids, corpus_manifest)

    async def retrieve(self, query: str) -> list[dict[str, Any]]:
        response = await asyncio.to_thread(self._worker.request, {"operation": "query", "query": query})
        documents = response.get("documents")
        if not isinstance(documents, list):
            raise TypeError("PropRAG official worker returned malformed documents")
        return documents

    async def run_workflow(self, query: str, history: list[dict] | None = None) -> tuple[str, list, list]:
        _ = history
        documents = await self.retrieve(query)
        if not documents:
            return mark_answer_boundary("Insufficient evidence."), [], [{"step": "proprag_retrieval", "output": "empty"}]
        context_documents = documents[:5]
        context = "\n\n".join(
            f"[{idx}] {row.get('title', '')}\n{row.get('text', '')}" for idx, row in enumerate(context_documents, 1)
        )
        answer = await self.llm.generate_response(
            [{"role": "user", "content": build_answer_prompt(context, query)}],
            temperature=0.0,
            max_tokens=RAGConfig.SYNTHESIS_MAX_OUTPUT_TOKENS,
        )
        if not str(answer or "").strip():
            raise ValueError("PropRAG answer synthesis returned an empty response")
        sources = [
            {
                "doc": row.get("title") or row.get("source_id", ""),
                "source": row.get("source_id", ""),
                "page": 0,
                "sent_id": 0,
                "text": row.get("text", ""),
                **({"score": row["score"]} if row.get("score") is not None else {}),
            }
            for row in documents
        ]
        return mark_answer_boundary(answer), sources, [
            {"step": "proprag_official_retrieval", "retrieved": len(documents), "answer_context": len(context_documents)}
        ]

    def close(self) -> None:
        self._worker.close()
