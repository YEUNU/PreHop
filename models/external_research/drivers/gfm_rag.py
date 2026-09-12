from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from core.benchmark_failures import BenchmarkIntegrityError
from core.generation_profiles import request_settings

from .base import canonical_semantic_env, load_rows, positive_env


def configure_entity_linker(retriever_cfg: Any, model_path: Path, output_dir: Path) -> None:
    """Keep both interpolated native entity-linker instances in this run."""
    retriever_cfg.el_model.model_name_or_path = str(model_path)
    retriever_cfg.el_model.root = str((output_dir / "artifacts/gfm_entity_linker").resolve())


class GFMRAGDriver:
    def __init__(self, official_root: Path, output_dir: Path):
        sys.path.insert(0, str(official_root))
        from gfmrag import GFMRetriever
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from langchain_openai import ChatOpenAI

        from core.inference_transport import InferenceTransport
        from core.runtime_requirements import runtime_requirement

        transport = InferenceTransport.resolve("gfm_rag")
        from core.strategy_registry import get_strategy

        registry_policy = dict(get_strategy("gfm_rag").paper_index_policy)

        runtime_spec = runtime_requirement("gfm_rag")
        checkpoint_path = official_root.parent / "artifacts" / str(runtime_spec["checkpoint_snapshot_subdir"])
        checkpoint = str(checkpoint_path)
        config_file = checkpoint_path / "config.json"
        model_file = checkpoint_path / "model.pth"
        if not checkpoint_path.is_dir() or not config_file.is_file() or not model_file.is_file():
            raise RuntimeError(
                "GFM-RAG pinned checkpoint snapshot must contain config.json and model.pth"
            )
        with model_file.open("rb") as checkpoint_stream:
            actual_checkpoint_sha = hashlib.file_digest(checkpoint_stream, "sha256").hexdigest()
        actual_config_sha = hashlib.sha256(config_file.read_bytes()).hexdigest()
        config_path = official_root / "gfmrag/workflow/config/gfm_rag/qa_ircot_inference.yaml"
        if not config_path.is_file():
            raise RuntimeError(f"GFM-RAG official inference config is missing: {config_path}")
        rows = load_rows(output_dir)
        raw_dir = output_dir / "artifacts/data/corpus/raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "documents.json").write_text(
            json.dumps({r["source_id"]: f"{r['title']}\n{r['text']}" for r in rows}, ensure_ascii=False),
            encoding="utf-8",
        )
        # A direct OmegaConf.load leaves Hydra defaults unresolved. Compose the
        # pinned package config exactly as the official workflow entrypoint does.
        with initialize_config_dir(config_dir=str(config_path.parent.resolve()), version_base=None):
            cfg = compose(config_name=config_path.stem)
        retriever_cfg = cfg.graph_retriever
        entity_model = str(runtime_spec["entity_linker_model"])
        entity_revision = str(runtime_spec["entity_linker_revision"])
        entity_model_path = official_root.parent / "artifacts" / str(
            runtime_spec["entity_linker_snapshot_subdir"]
        )
        if not entity_model_path.is_dir():
            raise RuntimeError("GFM-RAG strategy-local entity-linker snapshot is missing")
        configure_entity_linker(retriever_cfg, entity_model_path, output_dir)
        retriever_cfg.graph_constructor.root = str(output_dir / "artifacts/gfm_constructor_tmp")
        retriever_cfg.graph_constructor.num_processes = min(
            positive_env("RAG_GFM_CONSTRUCTION_CONCURRENCY", transport.generation_concurrency),
            transport.generation_concurrency,
        )
        generation_model = transport.generation_model
        from models.external_research.extraction_contract import ExtractionAudit
        from models.external_research.native_observation import observed_chat_type as guarded_chat_type

        self.extraction_audit = ExtractionAudit(output_dir / "artifacts" / "extraction_audit.jsonl", profile="native-observation-v1")
        compatible_chat = guarded_chat_type(ChatOpenAI)(
            model=generation_model,
            base_url=transport.generation_base_url,
            api_key=transport.api_key,
            **request_settings("gfm_construction"),
            max_retries=transport.retry_attempts,
            timeout=transport.timeout_seconds,
            seed=transport.generation_seed,
        )
        ner_model = instantiate(retriever_cfg.ner_model)
        graph_constructor = instantiate(retriever_cfg.graph_constructor)
        # Retain official NER/OpenIE prompts and parsers while replacing only
        # their transport with the configured OpenAI-compatible Gemma route.
        compatible_chat.configure_observation(self.extraction_audit)
        ner_model.client = compatible_chat
        graph_constructor.open_ie_model.client = compatible_chat
        self.retriever = GFMRetriever.from_index(
            data_dir=str(output_dir / "artifacts/data"),
            data_name="corpus",
            model_path=checkpoint,
            ner_model=ner_model,
            el_model=instantiate(retriever_cfg.el_model),
            graph_constructor=graph_constructor,
        )

        self.rows = {r["source_id"]: r for r in rows}
        self.top_k = int(canonical_semantic_env("RAG_GFM_RAG_TOP_K", registry_policy["retrieval_top_k"]))
        # The primary single-pass workflow is qa.py/qa_inference, not IRCOT.
        with initialize_config_dir(config_dir=str(config_path.parent.resolve()), version_base=None):
            qa_cfg = compose(config_name="qa_inference")
        from gfmrag.prompt_builder import QAPromptBuilder

        from .gfm_answer import native_qa_client

        if self.top_k != qa_cfg.test.top_k:
            raise RuntimeError("GFM single-pass top_k differs from its native QA configuration")
        self.native_qa_max_workers = int(qa_cfg.test.n_threads)
        self.qa_prompt = QAPromptBuilder(qa_cfg.qa_prompt)
        self.qa_llm, self.qa_audit, self.qa_sdk = native_qa_client(
            transport, qa_cfg.llm.model_name_or_path, qa_cfg.llm.retry,
            output_dir / 'artifacts' / 'answer_audit.jsonl')
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = actual_checkpoint_sha
        self.config_sha256 = actual_config_sha
        self.checkpoint_model = str(runtime_spec["checkpoint_model"])
        self.checkpoint_revision = str(runtime_spec["checkpoint_revision"])
        self.entity_linker_model = entity_model
        self.entity_linker_revision = entity_revision

    def index(self) -> dict[str, Any]:
        document_nodes = {
            str(node_id)
            for node_id, row in self.retriever.node_info.iterrows()
            if str(row.get("type", "")) == "document"
        }
        if document_nodes != set(self.rows):
            raise RuntimeError("GFM-RAG stored graph document nodes do not cover the exact staged source set")
        from models.external_research.extraction_contract import audit_evidence

        return {
            "extraction_audit_evidence": audit_evidence(self.extraction_audit),
            "source_count": len(self.rows),
            "extraction_validation_profile": "native-observation-v1",
            "coverage_complete": True,
            "native_top_k": self.top_k,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "checkpoint_model": self.checkpoint_model,
            "checkpoint_revision": self.checkpoint_revision,
            "entity_linker_model": self.entity_linker_model,
            "entity_linker_revision": self.entity_linker_revision,
        }

    def query(self, question: str) -> dict[str, Any]:
        return self._answer_prepared(self._prepare_query(question))

    def query_batch(self, questions):
        from multiprocessing.dummy import Pool as ThreadPool
        prepared = []
        for question in questions:
            try:
                prepared.append(self._prepare_query(question))
            except Exception as exc:  # noqa: BLE001 - report native failures without repair
                prepared.append(exc)
        def answer(item):
            if isinstance(item, Exception):
                return item
            try:
                return self._answer_prepared(item)
            except Exception as exc:  # noqa: BLE001 - report native failures without repair
                return exc
        # Match the native qa.py ThreadPool; retrieval remains sequential.
        with ThreadPool(self.native_qa_max_workers) as pool:
            return list(pool.imap(answer, prepared))

    def _prepare_query(self, question):
        result = self.retriever.retrieve(question, top_k=self.top_k)

        candidates = result.get("document") if isinstance(result, dict) else None
        if not isinstance(candidates, list):
            raise TypeError("GFM-RAG retrieval returned malformed documents")
        documents = []
        for candidate in candidates:
            source_id = str(candidate.get("id", ""))
            row = self.rows.get(source_id)
            if row is None:
                raise BenchmarkIntegrityError("GFM-RAG returned a foreign source identity")
            documents.append(
                {"source_id": source_id, "title": row["title"], "text": row["text"], "score": float(candidate["score"])}
            )
        # Match qa.py's native name plus flattened attributes, not the
        # repository's generic RAG answer prompt.
        prompt_docs = [{'name': candidate['id'], **candidate['attributes']} for candidate in candidates]
        return documents, self.qa_prompt.build_input_prompt(question, {'document': prompt_docs})

    def _answer_prepared(self, prepared):
        documents, prompt = prepared
        answer = self.qa_llm.generate_sentence(prompt)

        if isinstance(answer, Exception):
            raise answer
        if not isinstance(answer, str):
            raise RuntimeError('Native GFM QA returned no answer')  # noqa: TRY004 - native execution failure contract
        return {'documents': documents, 'answer': answer}

    def close(self) -> None:
        self.qa_sdk.close()
