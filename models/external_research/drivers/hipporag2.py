from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

from .base import canonical_semantic_env, load_rows, positive_env


def configure_hippo_openie(config: Any) -> None:
    """Use native structured extraction settings under a new semantic profile."""
    from core.strategy_registry import get_strategy

    policy = dict(get_strategy("hipporag2").paper_index_policy)
    canonical_semantic_env("RAG_HIPPORAG2_OPENIE_PROFILE", policy["openie_generation_profile"])
    response_type = canonical_semantic_env("RAG_HIPPORAG2_OPENIE_RESPONSE_FORMAT", policy["openie_response_format"])
    config.response_format = {"type": response_type}
    config.openie_ner_max_tokens = int(canonical_semantic_env("RAG_HIPPORAG2_NER_MAX_TOKENS", policy["openie_ner_max_tokens"]))
    config.openie_triple_max_tokens = int(canonical_semantic_env("RAG_HIPPORAG2_TRIPLE_MAX_TOKENS", policy["openie_triple_max_tokens"]))


class HippoRAG2Driver:
    def __init__(self, official_root: Path, output_dir: Path):
        sys.path.insert(0, str(official_root / "src"))
        import numpy as np
        from hipporag import HippoRAG
        from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
        from hipporag.utils.config_utils import BaseConfig

        from core.config import RAGConfig
        from core.inference_transport import InferenceTransport

        transport = InferenceTransport.resolve("hipporag2")
        from core.strategy_registry import get_strategy

        registry_policy = dict(get_strategy("hipporag2").paper_index_policy)
        # HippoRAG reads this conventional name internally. It is deliberately
        # mapped from the one repository transport credential rather than
        # imposing a second user-facing secret contract.
        os.environ["OPENAI_API_KEY"] = transport.api_key
        config = BaseConfig()
        configure_hippo_openie(config)
        config.embedding_batch_size = transport.embedding_batch_size
        config.embedding_request_timeout = transport.timeout_seconds
        config.openie_max_workers = min(
            positive_env("RAG_HIPPORAG2_OPENIE_WORKERS", transport.generation_concurrency),
            transport.generation_concurrency,
        )
        config.retrieval_top_k = int(
            canonical_semantic_env("RAG_HIPPORAG2_TOP_K", registry_policy["retrieval_top_k"])
        )
        config.embedding_base_url = transport.embedding_base_url
        config.embedding_model_name = transport.embedding_model
        config.max_retry_attempts = transport.retry_attempts
        config.seed = transport.generation_seed
        embedding_slots = threading.BoundedSemaphore(transport.embedding_concurrency)

        class BoundedOpenAIEmbeddingModel(OpenAIEmbeddingModel):
            """Official encoder plus fail-closed size bisection/shape checks."""

            @staticmethod
            def _splittable(exc: Exception) -> bool:
                status = getattr(exc, "status_code", None)
                text = str(exc).lower()
                return status == 413 or (
                    status == 400
                    and any(
                        token in text
                        for token in (
                            "messagepack",
                            "trailing characters",
                            "context length",
                            "too many tokens",
                            "request too large",
                        )
                    )
                )

            def encode(inner, texts):
                try:
                    with embedding_slots:
                        vectors = super(BoundedOpenAIEmbeddingModel, inner).encode(texts)
                except Exception as exc:
                    if len(texts) <= 1 or not inner._splittable(exc):
                        raise
                    middle = len(texts) // 2
                    left = inner.encode(texts[:middle])
                    left_usage = dict(inner.last_usage or {})
                    right = inner.encode(texts[middle:])
                    right_usage = dict(inner.last_usage or {})
                    vectors = np.concatenate((left, right), axis=0)
                    if left_usage.get("usage_unknown") or right_usage.get("usage_unknown"):
                        inner.last_usage = {"usage_unknown": True, "complete": False}
                    else:
                        inner.last_usage = {
                            "prompt_tokens": int(left_usage.get("prompt_tokens", 0))
                            + int(right_usage.get("prompt_tokens", 0)),
                            "total_tokens": int(left_usage.get("total_tokens", 0))
                            + int(right_usage.get("total_tokens", 0)),
                            "complete": left_usage.get("complete", True) is not False
                            and right_usage.get("complete", True) is not False,
                        }
                vectors = np.asarray(vectors, dtype=np.float32)
                if (
                    vectors.ndim != 2
                    or vectors.shape != (len(texts), RAGConfig.EMBEDDING_DIMENSIONS)
                    or not np.isfinite(vectors).all()
                ):
                    raise ValueError("HippoRAG2 embedding response has invalid count, dimension, or finite values")
                return vectors

        embedding_model = BoundedOpenAIEmbeddingModel(config, config.embedding_model_name)
        self.rows = load_rows(output_dir)
        self.engine = HippoRAG(
            global_config=config,
            save_dir=str(output_dir / "artifacts"),
            llm_model_name=transport.generation_model,
            llm_base_url=transport.generation_base_url,
            embedding_model_name=transport.embedding_model,
            embedding_provider="openai",
            embedding_base_url=transport.embedding_base_url,
            embedding_model=embedding_model,
            index_identity=__import__("os").environ.get("RAG_SEMANTIC_CONFIG_SHA256", registry_policy["semantic_config_id"]),
        )

    def index(self) -> dict[str, Any]:
        from hipporag.utils.misc_utils import Chunk

        docs = [
            Chunk(
                content=f"{r['title']}\n{r['text']}",
                source_id=r["source_id"],
                metadata={"source_id": r["source_id"], "title": r["title"]},
            )
            for r in self.rows
        ]
        self.engine.index(docs)
        indexed_sources = {
            str(metadata.get("source_id", ""))
            for metadata in self.engine.chunk_metadata.values()
            if isinstance(metadata, dict)
        }
        expected_sources = {row["source_id"] for row in self.rows}
        if indexed_sources != expected_sources:
            raise RuntimeError("HippoRAG2 stored chunk metadata does not cover the exact staged source set")
        return {
            "source_count": len(self.rows), "coverage_complete": True,
            "native_top_k": self.engine.global_config.retrieval_top_k,
            "openie_response_format": self.engine.global_config.response_format,
            "openie_ner_max_tokens": self.engine.global_config.openie_ner_max_tokens,
            "openie_triple_max_tokens": self.engine.global_config.openie_triple_max_tokens,
        }

    def query(self, question: str) -> dict[str, Any]:
        solutions, _messages, _metadata = self.engine.rag_qa([question])
        if len(solutions) != 1:
            raise RuntimeError("HippoRAG2 rag_qa returned an unexpected result count")
        result = solutions[0]
        scores = result.doc_scores.tolist() if hasattr(result.doc_scores, "tolist") else list(result.doc_scores)
        metadata = result.doc_metadata or [{} for _ in result.docs]
        if not (len(result.docs) == len(scores) == len(metadata)):
            raise ValueError("HippoRAG2 returned inconsistent document/score/metadata counts")
        documents = []
        for text, score, meta in zip(result.docs, scores, metadata):
            source_id = str(meta.get("source_id", ""))
            if not source_id:
                raise RuntimeError("HippoRAG2 retrieval lost source_id metadata")
            documents.append(
                {"source_id": source_id, "title": meta.get("title", source_id), "text": text, "score": float(score)}
            )
        if not isinstance(result.answer, str) or not result.answer.strip():
            raise RuntimeError("HippoRAG2 native QA returned an empty answer")
        return {"documents": documents, "answer": result.answer}

    def close(self) -> None:
        self.engine.close()
