from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

from .base import canonical_semantic_env, load_rows, positive_env

_NATIVE_ERROR_PREFIXES = ("error:", "[error")
_NATIVE_EVIDENCE_SENTINELS = {
    "No relevant information found",
    "No relevant chunks found",
}
_NATIVE_ANSWER_RETRY_ATTEMPTS = 20


def canonical_observation_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def classify_native_extraction(parsed: Any) -> str:
    """Classify the value returned by upstream's tolerant JSON parser.

    This is post-validation only: callers return ``parsed`` to upstream
    unchanged. The classification makes parse failures that upstream silently
    turns into ``None`` distinguishable from a valid, non-empty extraction.
    """
    if not isinstance(parsed, dict):
        return "malformed"
    attributes = parsed.get("attributes", {})
    triples = parsed.get("triples", [])
    entity_types = parsed.get("entity_types", {})
    new_schema_types = parsed.get("new_schema_types", {})
    if (
        not isinstance(attributes, dict)
        or not isinstance(triples, list)
        or not isinstance(entity_types, dict)
        or not isinstance(new_schema_types, dict)
        or any(
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value.strip() for value in values)
            for values in attributes.values()
        )
        or any(
            not isinstance(triple, list)
            or len(triple) != 3
            or any(not isinstance(value, str) or not value.strip() for value in triple)
            for triple in triples
        )
        or any(not isinstance(values, list) for values in new_schema_types.values())
    ):
        return "malformed"
    if not any(attributes.values()) and not triples:
        return "empty"
    return "success"


def configure_native_openai_client(component: Any, *, retry_attempts: int, timeout_seconds: float | None) -> None:
    """Apply typed SDK controls through the OpenAI client's public clone API."""
    if retry_attempts < 1:
        raise ValueError("Youtu client retry attempts must be positive")
    llm_client = getattr(component, "llm_client", None)
    client = getattr(llm_client, "client", None)
    with_options = getattr(client, "with_options", None)
    if not callable(with_options):
        raise TypeError("Pinned Youtu component does not expose its OpenAI client injection point")
    options: dict[str, Any] = {"max_retries": retry_attempts}
    if timeout_seconds is not None:
        options["timeout"] = timeout_seconds
    llm_client.client = with_options(**options)


def validate_native_query_result(result: Any, chunk_sources: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    """Validate, without repairing or reordering, one upstream query result."""
    if not isinstance(result, dict):
        raise TypeError("Youtu official query API returned a malformed result")
    answer = result.get("initial_answer")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Youtu official query API returned an empty answer")
    if answer.strip().lower().startswith(_NATIVE_ERROR_PREFIXES):
        raise RuntimeError("Youtu official query API exhausted generation retries")

    chunk_ids = result.get("chunk_ids")
    chunk_contents = result.get("chunk_contents")
    triples = result.get("triples")
    if not isinstance(chunk_ids, list) or not isinstance(chunk_contents, list) or not isinstance(triples, list):
        raise TypeError("Youtu official query API returned malformed evidence")
    if not chunk_ids or len(chunk_ids) != len(chunk_contents):
        raise RuntimeError("Youtu official query API returned empty or misaligned chunk evidence")
    normalized_ids = [str(chunk_id) for chunk_id in chunk_ids]
    if any(not chunk_id for chunk_id in normalized_ids) or len(set(normalized_ids)) != len(normalized_ids):
        raise RuntimeError("Youtu official query API returned empty or duplicate chunk IDs")
    if any(chunk_id not in chunk_sources for chunk_id in normalized_ids):
        raise RuntimeError("Youtu retrieval returned chunk IDs absent from the provenance sidecar")
    if any(
        not isinstance(content, str)
        or not content.strip()
        or content.strip() in _NATIVE_EVIDENCE_SENTINELS
        or content.strip().lower().startswith(_NATIVE_ERROR_PREFIXES)
        or content.strip().startswith("[Missing content for chunk ")
        for content in chunk_contents
    ) or any(
        not isinstance(triple, str)
        or not triple.strip()
        or triple.strip() in _NATIVE_EVIDENCE_SENTINELS
        or triple.strip().lower().startswith(_NATIVE_ERROR_PREFIXES)
        for triple in triples
    ):
        raise RuntimeError("Youtu official query API returned sentinel, empty, or malformed evidence")
    sub_results = result.get("sub_question_results")
    if not isinstance(sub_results, list) or not sub_results:
        raise TypeError("Youtu official query API omitted sub-question status")
    for sub_result in sub_results:
        if not isinstance(sub_result, dict):
            raise TypeError("Youtu official query API returned malformed sub-question status")
        triples_count = sub_result.get("triples_count")
        chunks_count = sub_result.get("chunk_ids_count")
        if (
            isinstance(triples_count, bool)
            or not isinstance(triples_count, int)
            or isinstance(chunks_count, bool)
            or not isinstance(chunks_count, int)
            or triples_count < 0
            or chunks_count < 0
        ):
            raise TypeError("Youtu official query API returned malformed sub-question counts")
        if triples_count == 0 and chunks_count == 0:
            raise RuntimeError("Youtu official query API reported an empty or swallowed sub-question failure")
    return answer.strip(), list(zip(normalized_ids, chunk_contents))


def derive_native_source_reachability(
    graph_output: Any,
    chunk_sources: dict[str, str],
    expected_source_ids: set[str],
) -> dict[str, Any]:
    """Observe source reachability from the unmodified upstream graph JSON."""
    if not isinstance(graph_output, list) or not graph_output:
        raise RuntimeError("Youtu construction persisted an empty graph artifact")
    source_chunks: dict[str, set[str]] = {source_id: set() for source_id in expected_source_ids}
    for relationship in graph_output:
        if not isinstance(relationship, dict):
            raise TypeError("Youtu graph artifact contains a malformed relationship")
        for endpoint in (relationship.get("start_node"), relationship.get("end_node")):
            if not isinstance(endpoint, dict) or not isinstance(endpoint.get("properties"), dict):
                raise TypeError("Youtu graph artifact contains a malformed endpoint")
            chunk_id = endpoint["properties"].get("chunk id")
            if not chunk_id:
                continue
            source_id = chunk_sources.get(str(chunk_id))
            if source_id is None:
                raise RuntimeError("Youtu graph edge refers to a chunk absent from staged provenance")
            if source_id not in source_chunks:
                raise RuntimeError("Youtu graph edge refers to a source absent from the staged corpus")
            source_chunks[source_id].add(str(chunk_id))
    missing = sorted(source_id for source_id, chunk_ids in source_chunks.items() if not chunk_ids)
    reachable = {
        source_id: sorted(chunk_ids)
        for source_id, chunk_ids in sorted(source_chunks.items())
        if chunk_ids
    }
    return {
        "reachable_sources": reachable,
        "unreachable_source_ids": missing,
        "complete": not missing,
    }


class YoutuGraphRAGDriver:
    """Pinned Youtu construction/retrieval APIs inside a run-local cwd."""

    def __init__(self, official_root: Path, output_dir: Path):
        from core.paper_policy import approved_youtu_schema

        self.dataset_name = output_dir.name
        if self.dataset_name not in {"multihoprag", "musique"}:
            raise ValueError("Youtu paper driver requires the actual multihoprag or musique corpus tag")
        approved_schema = approved_youtu_schema(self.dataset_name)
        self.native_dataset_name = approved_schema["native_dataset"]
        schema_source = official_root / approved_schema["path"]
        if not schema_source.is_file():
            raise RuntimeError("Youtu pinned checkout is missing its approved dataset schema")
        expected_schema_sha = approved_schema["sha256"]
        self.expected_schema_sha = expected_schema_sha
        actual_schema_sha = hashlib.sha256(schema_source.read_bytes()).hexdigest()
        if expected_schema_sha != actual_schema_sha:
            raise RuntimeError("Youtu checkout schema does not match the approved runtime manifest digest")
        # Upstream uses an absolute top-level package also named ``models``.
        # This worker is already process-isolated, so after this driver module
        # is loaded, evict the project package aliases and admit only the
        # pinned checkout. Without this, Python silently imports our package.
        for module_name in list(sys.modules):
            if module_name == "models" or module_name.startswith("models."):
                del sys.modules[module_name]
        sys.path.insert(0, str(official_root))
        from core.inference_transport import InferenceTransport

        transport = InferenceTransport.resolve("youtu_graphrag")
        self.transport = transport
        os.environ["OPENAI_PROVIDER"] = "openai"
        for ambient_vendor_setting in (
            "API_VERSION",
            "OPENAI_API_VERSION",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_API_KEY",
            "AZURE_API_BASE",
        ):
            os.environ.pop(ambient_vendor_setting, None)
        os.environ["LLM_BASE_URL"] = transport.generation_base_url
        os.environ["LLM_MODEL"] = transport.generation_model
        os.environ["LLM_API_KEY"] = transport.api_key
        from config import get_config

        from core.runtime_requirements import runtime_requirement
        from core.strategy_registry import get_strategy

        module_spec = importlib.util.spec_from_file_location("_prehop_youtu_official_main", official_root / "main.py")
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError("Youtu pinned main.py could not be loaded")
        official_main = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = official_main
        module_spec.loader.exec_module(official_main)
        self.official_main = official_main
        self.rows = load_rows(output_dir)
        self.by_id = {r["source_id"]: r for r in self.rows}
        self.workdir = output_dir / "artifacts"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.old_cwd = Path.cwd()
        os.chdir(self.workdir)
        schema_dir = self.workdir / "schemas"
        schema_dir.mkdir(parents=True, exist_ok=True)
        # Agent-mode schema evolution hardcodes cwd-relative
        # schemas/{dataset}.json upstream. Place the approved starting schema
        # at that exact run-local path so evolution works without mutating the
        # pinned checkout or falling back to a demo schema.
        self.schema_path = schema_dir / f"{self.native_dataset_name}.json"
        schema_evolution_path = self.workdir / "schema_evolution.json"
        completed_graph_path = self.workdir / f"output/graphs/{self.native_dataset_name}_new.json"
        if self.schema_path.exists():
            observed_schema_sha = hashlib.sha256(self.schema_path.read_bytes()).hexdigest()
            if observed_schema_sha != expected_schema_sha:
                try:
                    evolution = json.loads(schema_evolution_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise RuntimeError("Youtu evolved schema has no valid run-local provenance") from exc
                if (
                    not completed_graph_path.is_file()
                    or evolution.get("starting_sha256") != expected_schema_sha
                    or evolution.get("evolved_sha256") != observed_schema_sha
                ):
                    raise RuntimeError("Youtu evolved schema provenance does not match the completed graph")
        else:
            shutil.copyfile(schema_source, self.schema_path)
        self.schema_evolution_path = schema_evolution_path
        self.corpus_path = self.workdir / "corpus.json"
        self.graph_path = completed_graph_path
        self.sidecar_path = self.workdir / "chunk_source_ids.json"
        self.staged_evidence_path = self.workdir / "staged_input_evidence.json"
        self.extraction_evidence_path = self.workdir / "source_extraction_evidence.json"
        self.source_evidence_path = self.workdir / "source_graph_evidence.json"
        self.config = get_config(str(official_root / "config/base_config.yaml"))
        strategy_spec = get_strategy("youtu_graphrag")
        registry_policy = dict(strategy_spec.paper_index_policy)
        self.backbone_mode = str(registry_policy["backbone_mode"])
        if not strategy_spec.local_embedding_revision:
            raise RuntimeError("Youtu registry is missing its pinned embedding revision")
        snapshot_subdir = runtime_requirement("youtu_graphrag")["embedding_snapshot_subdir"]
        self.pinned_embedding_path = official_root.parent / "artifacts" / snapshot_subdir
        if not self.pinned_embedding_path.is_dir():
            raise RuntimeError("Youtu strategy-local MiniLM snapshot is missing")
        self.config.embeddings.model_name = str(self.pinned_embedding_path)
        self.config.tree_comm.embedding_model = str(self.pinned_embedding_path)
        no_chunk_datasets = set(self.config.construction.datasets_no_chunk or [])
        if self.native_dataset_name not in no_chunk_datasets:
            raise RuntimeError("Pinned Youtu no-chunk policy no longer matches the paper corpus units")
        if self.dataset_name == "musique" and any(not source_id.startswith("musique_") for source_id in self.by_id):
            raise RuntimeError("Youtu MuSiQue input must contain one prepared paragraph per staged source")
        self.config.construction.max_workers = int(
            canonical_semantic_env(
                "RAG_YOUTU_CONSTRUCTION_CONCURRENCY", registry_policy["construction_concurrency"]
            )
        )
        self.config.construction.mode = str(registry_policy["construction_mode"])
        self.config.retrieval.top_k = int(registry_policy["retrieval_top_k"])
        self.config.retrieval.top_k_filter = int(registry_policy["retrieval_top_k_filter"])
        self.config.retrieval.faiss.max_workers = positive_env("RAG_YOUTU_RETRIEVAL_CONCURRENCY", 1)
        self.config.triggers.mode = str(registry_policy["query_mode"])
        if self.config.construction.mode not in {"agent", "noagent"} or self.config.triggers.mode not in {
            "agent",
            "noagent",
        }:
            raise ValueError("Youtu construction/query modes must be agent or noagent")
        if self.config.triggers.mode != "noagent":
            raise RuntimeError(
                "Pinned Youtu exposes structured per-query results only through its public "
                "initial_question_decomposition noagent API"
            )
        self.retriever = None
        self.graphq = None
        if self.graph_path.exists():
            self._load_retriever()

    def _load_retriever(self) -> None:
        from models.retriever.agentic_decomposer import GraphQ
        from models.retriever.enhanced_kt_retriever import KTRetriever
        from sentence_transformers import SentenceTransformer

        if not self.sidecar_path.is_file():
            raise RuntimeError("Youtu chunk/source sidecar is missing")
        try:
            chunk_sources = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("Youtu chunk/source sidecar is unreadable") from exc
        if (
            not isinstance(chunk_sources, dict)
            or not chunk_sources
            or any(not isinstance(key, str) or not key for key in chunk_sources)
            or any(not isinstance(value, str) or not value for value in chunk_sources.values())
            or set(chunk_sources.values()) != set(self.by_id)
        ):
            raise RuntimeError("Youtu chunk/source sidecar does not cover the exact staged source set")
        self.chunk_sources = chunk_sources
        self.retriever = KTRetriever(
            self.native_dataset_name,
            json_path=str(self.graph_path),
            qa_encoder=SentenceTransformer(str(self.pinned_embedding_path)),
            device=self.config.embeddings.device,
            cache_dir=str(self.workdir / "retriever/faiss_cache_new"),
            top_k=self.config.retrieval.top_k_filter,
            recall_paths=self.config.retrieval.recall_paths,
            schema_path=str(self.schema_path),
            mode=self.config.triggers.mode,
            config=self.config,
        )
        # The upstream retrieval entrypoint calls this public lifecycle API
        # after construction even when an on-disk index already exists.
        # Query workers must load every native FAISS index into memory too.
        self.retriever.build_indices()
        self.graphq = GraphQ(self.native_dataset_name, config=self.config)
        configure_native_openai_client(
            self.retriever,
            retry_attempts=self.transport.retry_attempts,
            timeout_seconds=self.transport.timeout_seconds,
        )
        configure_native_openai_client(
            self.graphq,
            retry_attempts=self.transport.retry_attempts,
            timeout_seconds=self.transport.timeout_seconds,
        )

    def index(self) -> dict[str, Any]:
        from models.constructor.kt_gen import KTBuilder

        # KTBuilder ignores extra dict fields when forming semantic text. A
        # thin subclass captures the source_id before delegating chunking, so
        # provenance is exact without adding markers to embeddings/prompts.
        corpus = [{"source_id": r["source_id"], "title": r["title"], "text": r["text"]} for r in self.rows]
        self.corpus_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")

        class ObservedKTBuilder(KTBuilder):
            def __init__(inner, *args, **kwargs):
                super().__init__(*args, **kwargs)
                inner.chunk_sources: dict[str, str] = {}
                inner.failed_sources: set[str] = set()
                inner._active_source = threading.local()
                inner.extraction_by_source = {
                    source_id: {
                        "source_id": source_id,
                        "expected_chunk_count": 0,
                        "extraction_attempt_count": 0,
                        "success_count": 0,
                        "malformed_count": 0,
                        "empty_count": 0,
                        "upstream_error_count": 0,
                        "process_error_count": 0,
                    }
                    for source_id in self.by_id
                }

            def _current_observation(inner):
                source_id = getattr(inner._active_source, "source_id", "")
                if source_id not in inner.extraction_by_source:
                    raise RuntimeError("Youtu extraction lost its active staged source identity")
                return inner.extraction_by_source[source_id]

            def chunk_text(inner, text):
                source_id = str(text.get("source_id", "")) if isinstance(text, dict) else ""
                if source_id not in self.by_id:
                    raise RuntimeError("Youtu corpus record lost its source identity before chunking")
                chunks, chunk2id = super().chunk_text(text)
                inner.chunk_sources.update({chunk_id: source_id for chunk_id in chunk2id})
                inner.extraction_by_source[source_id]["expected_chunk_count"] += len(chunk2id)
                return chunks, chunk2id

            def extract_with_llm(inner, prompt):
                observation = inner._current_observation()
                observation["extraction_attempt_count"] += 1
                try:
                    return super().extract_with_llm(prompt)
                except Exception:
                    observation["upstream_error_count"] += 1
                    raise

            def _validate_and_parse_llm_response(inner, prompt, llm_response):
                parsed = super()._validate_and_parse_llm_response(prompt, llm_response)
                observation = inner._current_observation()
                observation[f"{classify_native_extraction(parsed)}_count"] += 1
                # Return the exact native value. This observer does not repair
                # parsing, add facts, or change graph construction behavior.
                return parsed

            def process_document(inner, document):
                source_id = str(document.get("source_id", "")) if isinstance(document, dict) else ""
                if source_id not in inner.extraction_by_source:
                    raise RuntimeError("Youtu construction received a foreign staged source")
                inner._active_source.source_id = source_id
                try:
                    return super().process_document(document)
                except Exception:
                    inner.failed_sources.add(source_id)
                    inner.extraction_by_source[source_id]["process_error_count"] += 1
                    raise
                finally:
                    inner._active_source.source_id = ""

        builder = ObservedKTBuilder(
            self.native_dataset_name,
            schema_path=str(self.schema_path),
            mode=self.config.construction.mode,
            config=self.config,
        )
        configure_native_openai_client(
            builder,
            retry_attempts=self.transport.retry_attempts,
            timeout_seconds=self.transport.timeout_seconds,
        )
        from core.native_structured_profile import YoutuConstructionClient

        builder.llm_client.client = YoutuConstructionClient(builder.llm_client.client)
        builder.build_knowledge_graph(str(self.corpus_path))
        chunk_sources = builder.chunk_sources
        staged_complete = (
            set(chunk_sources) == set(builder.all_chunks)
            and set(chunk_sources.values()) == set(self.by_id)
        )
        staged_observation = {
            "expected_source_count": len(self.by_id),
            "observed_source_count": len(set(chunk_sources.values())),
            "chunk_count": len(chunk_sources),
            "source_ids_sha256": hashlib.sha256(
                "\n".join(sorted(self.by_id)).encode("utf-8")
            ).hexdigest(),
            "chunk_sources_sha256": canonical_observation_sha256(chunk_sources),
            "complete": staged_complete,
        }
        self.staged_evidence_path.write_text(
            json.dumps(staged_observation, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        if not staged_complete:
            raise RuntimeError("Youtu chunk provenance does not cover the exact staged source/chunk set")

        extraction_rows = []
        extraction_counts = {"success": 0, "malformed": 0, "empty": 0, "swallowed_error": 0}
        for source_id in sorted(builder.extraction_by_source):
            observation = dict(builder.extraction_by_source[source_id])
            expected = observation["expected_chunk_count"]
            attempts = observation["extraction_attempt_count"]
            has_swallowed_error = bool(
                observation["upstream_error_count"]
                or observation["process_error_count"]
                or source_id in builder.failed_sources
            )
            has_malformed = bool(observation["malformed_count"])
            has_empty = bool(observation["empty_count"])
            if has_swallowed_error:
                status = "swallowed_error"
            elif has_malformed:
                status = "malformed"
            elif has_empty:
                status = "empty"
            elif expected < 1 or attempts != expected or observation["success_count"] != expected:
                status = "malformed"
                has_malformed = True
            else:
                status = "success"
            observation.update(
                {
                    "success": status == "success",
                    "malformed": has_malformed,
                    "empty": has_empty,
                    "swallowed_error": has_swallowed_error,
                }
            )
            observation["status"] = status
            extraction_counts["success"] += int(status == "success")
            extraction_counts["malformed"] += int(has_malformed)
            extraction_counts["empty"] += int(has_empty)
            extraction_counts["swallowed_error"] += int(has_swallowed_error)
            extraction_rows.append(observation)
        extraction_observation = {
            "expected_source_count": len(self.by_id),
            "source_count": len(extraction_rows),
            "success_source_count": extraction_counts["success"],
            "malformed_source_count": extraction_counts["malformed"],
            "empty_source_count": extraction_counts["empty"],
            "swallowed_error_source_count": extraction_counts["swallowed_error"],
            "complete": extraction_counts["success"] == len(self.by_id),
            "sources": extraction_rows,
        }
        self.extraction_evidence_path.write_text(
            json.dumps(extraction_observation, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        if not extraction_observation["complete"]:
            raise RuntimeError(
                "Youtu native extraction was empty, malformed, or swallowed an upstream source failure "
                f"(malformed={extraction_counts['malformed']}, empty={extraction_counts['empty']}, "
                f"swallowed={extraction_counts['swallowed_error']})"
            )
        if not self.schema_path.is_file():
            raise RuntimeError("Youtu agent construction removed its run-local schema")
        try:
            graph_output = json.loads(self.graph_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Youtu construction did not persist a valid graph artifact") from exc
        native_reachability = derive_native_source_reachability(graph_output, chunk_sources, set(self.by_id))
        self.schema_evolution_path.write_text(
            json.dumps(
                {
                    "starting_sha256": self.expected_schema_sha,
                    "evolved_sha256": hashlib.sha256(self.schema_path.read_bytes()).hexdigest(),
                    "native_dataset": self.native_dataset_name,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.sidecar_path.write_text(json.dumps(chunk_sources, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        self.source_evidence_path.write_text(
            json.dumps(native_reachability, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        persisted_source_evidence = json.loads(self.source_evidence_path.read_text(encoding="utf-8"))
        if persisted_source_evidence != native_reachability:
            raise RuntimeError("Youtu source-level graph evidence sidecar failed round-trip validation")
        self._load_retriever()
        from core.native_structured_profile import YOUTU_STRUCTURED_PROFILE, youtu_profile_sha256

        return {
            "extraction_generation_profile": YOUTU_STRUCTURED_PROFILE,
            "extraction_schema_sha256": youtu_profile_sha256(),
            "source_count": len(self.rows),
            "coverage_complete": True,
            "chunk_count": len(chunk_sources),
            "query_mode": self.config.triggers.mode,
            "backbone_mode": self.backbone_mode,
            "native_answer_retry_attempts": _NATIVE_ANSWER_RETRY_ATTEMPTS,
            "client_retry_attempts": self.transport.retry_attempts,
            "client_timeout_seconds": self.transport.timeout_seconds,
            "staged_input_source_count": staged_observation["observed_source_count"],
            "staged_input_source_ids_sha256": staged_observation["source_ids_sha256"],
            "staged_input_evidence_sha256": canonical_observation_sha256(staged_observation),
            "staged_input_coverage_complete": staged_observation["complete"],
            "input_chunk_coverage_complete": True,
            "native_extraction_source_count": extraction_observation["source_count"],
            "native_extraction_success_count": extraction_observation["success_source_count"],
            "native_extraction_malformed_count": extraction_observation["malformed_source_count"],
            "native_extraction_empty_count": extraction_observation["empty_source_count"],
            "native_extraction_swallowed_error_count": extraction_observation["swallowed_error_source_count"],
            "native_extraction_success_complete": extraction_observation["complete"],
            "native_extraction_evidence_sha256": canonical_observation_sha256(extraction_observation),
            "native_source_reachability_observational": True,
            "native_source_reachability_complete": native_reachability["complete"],
            "native_source_reachability_count": len(native_reachability["reachable_sources"]),
            "native_unreachable_source_count": len(native_reachability["unreachable_source_ids"]),
            "native_source_reachability_evidence_sha256": canonical_observation_sha256(native_reachability),
        }

    def query(self, question: str) -> dict[str, Any]:
        # The pinned checkout has one public per-query function that returns
        # both its answer and retrieval evidence. Its agent batch entrypoint
        # returns no result and unconditionally runs an evaluator, so this
        # controlled adapter uses the native noagent API instead of recreating
        # the agent loop in project code.
        if getattr(getattr(self.config, "triggers", None), "mode", None) != "noagent":
            raise RuntimeError("Youtu controlled adapter query mode changed after initialization")
        query_api = getattr(self.official_main, "initial_question_decomposition", None)
        if not callable(query_api):
            raise TypeError("Pinned Youtu checkout lacks its public per-query API")
        self.official_main.config = self.config
        native_result = query_api(self.graphq, self.retriever, question, str(self.schema_path))
        answer, ordered_evidence = validate_native_query_result(native_result, self.chunk_sources)
        documents = []
        seen_sources: set[str] = set()
        for chunk_id, content in ordered_evidence:
            source_id = self.chunk_sources[chunk_id]
            if source_id in seen_sources:
                raise RuntimeError("Youtu no-chunk retrieval returned duplicate source evidence")
            row = self.by_id[source_id]
            documents.append(
                {"source_id": source_id, "title": row["title"], "text": content}
            )
            seen_sources.add(source_id)
        return {"documents": documents, "answer": answer}

    def close(self) -> None:
        os.chdir(self.old_cwd)
