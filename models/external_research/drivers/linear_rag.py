from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from core.generation_profiles import request_settings

from .base import canonical_semantic_env, load_rows, positive_env


def resolve_pinned_model_path(official_root: Path) -> str:
    """Resolve the strategy-local snapshot without consulting an ambient cache."""
    from core.runtime_requirements import runtime_requirement

    path = official_root.parent / "artifacts" / runtime_requirement("linear_rag")["embedding_snapshot_subdir"]
    if not path.is_dir():
        raise RuntimeError("Pinned LinearRAG MPNet snapshot is absent; rerun setup_official_baselines.sh")
    return str(path)


class LinearNativeInference:
    """Transport-only implementation of the native config.llm_model interface."""

    def __init__(self):
        from openai import OpenAI

        from core.inference_transport import InferenceTransport

        self.transport = InferenceTransport.resolve("linear_rag")
        self.client = OpenAI(base_url=self.transport.generation_base_url, api_key=self.transport.api_key,
                             timeout=self.transport.timeout_seconds, max_retries=self.transport.retry_attempts)

    def infer(self, messages):
        response = self.client.chat.completions.create(
            model=self.transport.generation_model, messages=messages, **request_settings("linear_native_qa"),
            seed=self.transport.generation_seed,
        )
        return response.choices[0].message.content

    def close(self):
        self.client.close()


class LinearRAGDriver:
    """Official relation-free Tri-Graph with the published MPNet backbone."""

    def __init__(self, official_root: Path, output_dir: Path):
        from core.strategy_registry import get_strategy

        strategy_spec = get_strategy("linear_rag")
        registry_policy = dict(strategy_spec.paper_index_policy)
        canonical_semantic_env("RAG_LINEAR_RAG_BACKBONE_MODE", registry_policy["backbone_mode"])
        sys.path.insert(0, str(official_root))
        from sentence_transformers import SentenceTransformer
        from src.config import LinearRAGConfig
        from src.LinearRAG import LinearRAG

        str(
            canonical_semantic_env(
                "RAG_LINEAR_RAG_MPNET_REVISION", registry_policy["official_embedding_revision"]
            )
        )
        str(
            canonical_semantic_env("RAG_LINEAR_RAG_MPNET_MODEL", registry_policy["official_embedding_model"])
        )
        pinned_model_path = resolve_pinned_model_path(official_root)

        self.rows = load_rows(output_dir)
        # The official runner requires the numeric ``idx:`` prefix. Keep that
        # exact representation and resolve it through adapter-owned metadata;
        # never inject provenance markers into text seen by NER/embeddings.
        self.passages = [f"{i}:{r['title']}\n{r['text']}" for i, r in enumerate(self.rows)]
        self.by_index = {i: r for i, r in enumerate(self.rows)}
        config = LinearRAGConfig(
            dataset_name="corpus",
            embedding_model=SentenceTransformer(
                pinned_model_path,
            ),
            llm_model=LinearNativeInference(),
            spacy_model=str(canonical_semantic_env("RAG_LINEAR_RAG_SPACY_MODEL", registry_policy["spacy_model"])),
            working_dir=str(output_dir / "artifacts"),
            batch_size=positive_env("RAG_EMBEDDING_BATCH_SIZE", 16),
            max_workers=positive_env("RAG_LINEAR_RAG_NER_WORKERS", 1),
            retrieval_top_k=int(canonical_semantic_env("RAG_LINEAR_RAG_TOP_K", registry_policy["retrieval_top_k"])),
            use_vectorized_retrieval=bool(canonical_semantic_env("RAG_LINEAR_RAG_VECTORIZED", registry_policy["vectorized_retrieval"])),
        )
        self.engine = LinearRAG(global_config=config)

    def index(self) -> dict[str, Any]:
        self.engine.index(self.passages)
        indexed_passages = self.engine.passage_embedding_store.get_hash_id_to_text()
        if len(indexed_passages) != len(self.passages) or set(indexed_passages.values()) != set(self.passages):
            raise RuntimeError("LinearRAG stored passage index does not cover the exact staged corpus")
        return {
            "source_count": len(self.rows),
            "coverage_complete": True,
            "backbone_mode": "official_faithful",
            "native_top_k": self.engine.config.retrieval_top_k,
        }

    def query(self, question: str) -> dict[str, Any]:
        if self.engine.graph.vcount() == 0:
            self.engine.index(self.passages)
        result = self.engine.qa([{"question": question, "answer": ""}])[0]
        passages, scores = result["sorted_passage"], result["sorted_passage_scores"]
        if len(passages) != len(scores):
            raise ValueError("LinearRAG returned inconsistent passage/score counts")
        documents = []
        for passage, score in zip(passages, scores):
            prefix, separator, _ = passage.partition(":")
            if not separator or not prefix.isdecimal():
                raise RuntimeError("LinearRAG retrieval lost its official numeric passage identity")
            row = self.by_index.get(int(prefix))
            if row is None or passage != self.passages[int(prefix)]:
                raise RuntimeError("LinearRAG returned a foreign or altered passage identity")
            source_id = row["source_id"]
            documents.append(
                {"source_id": source_id, "title": row["title"], "text": row["text"], "score": float(score)}
            )
        answer = result.get("pred_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("LinearRAG native qa omitted its answer")
        return {"documents": documents, "answer": answer}

    def close(self) -> None:
        self.engine.llm_model.close()
