"""Typed, dependency-free strategy registry used by Python and shell entrypoints."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PaperTransportSpec:
    """Canonical public inference contract for every primary paper target."""

    generation_model: str = "gemma-4-31b-it"
    embedding_model: str = "qwen3-embedding-4b"
    embedding_dimensions: int = 2560
    generation_context_tokens: int = 262144
    embedding_max_input_tokens: int = 32768
    embedding_token_reserve: int = 0
    embedding_batch_size: int = 16
    embedding_concurrency: int = 1
    generation_concurrency: int = 30
    retry_attempts: int = 5
    timeout_seconds: float = 600.0
    benchmark_seed: int = 42
    benchmark_concurrency: int = 1
    benchmark_checkpoint_every: int = 10
    query_instruction: str = "Given a web search query, retrieve relevant passages that answer the query"
    query_template: str = "Instruct: {instruction}\nQuery:{query}"


# Also support the dependency-free `python core/strategy_registry.py` CLI.
if __package__ in {None, ""}:
    from runtime_requirements import runtime_requirement
else:
    from core.runtime_requirements import runtime_requirement

# Defaults are independent of import order. Runtime profile overrides are
# resolved by core.execution_profile at the point of use.
PAPER_TRANSPORT = PaperTransportSpec()


@dataclass(frozen=True)
class StrategySpec:
    name: str
    primary: bool = False
    external: bool = False
    repository: str | None = None
    revision: str | None = None
    worker: str | None = None
    output_env: str | None = None
    output_default: str | None = None
    driver: str | None = None
    license_note: str | None = None
    adapter_variant: str = "official_faithful"
    transport_profile: str = "openai_compatible_litellm"
    paper_generation_model: str = PAPER_TRANSPORT.generation_model
    paper_generation_seed: int | None = None
    paper_embedding_model: str = PAPER_TRANSPORT.embedding_model
    paper_embedding_dimensions: int | None = PAPER_TRANSPORT.embedding_dimensions
    local_embedding_revision: str | None = None
    paper_index_policy: tuple[tuple[str, Any], ...] = ()
    # (semantic field, controlling environment variable); defaults live in the policy.
    paper_index_environment: tuple[tuple[str, str], ...] = ()
    # (provenance field, controlling environment variable, canonical value).
    # Query policies also describe historical ablation controls.
    paper_query_policy: tuple[tuple[str, str | None, Any], ...] = ()

    def index_environment_defaults(self) -> dict[str, Any]:
        policy = dict(self.paper_index_policy)
        return {name: policy[field] for field, name in self.paper_index_environment}


def _external(
    name: str,
    repository: str,
    revision: str,
    *,
    primary: bool,
    worker: str,
    driver: str | None = None,
    license_note: str | None = None,
    adapter_variant: str = "official_faithful",
    transport_profile: str = "openai_compatible_litellm",
    paper_embedding_model: str = PAPER_TRANSPORT.embedding_model,
    paper_generation_seed: int | None = None,
    paper_embedding_dimensions: int | None = PAPER_TRANSPORT.embedding_dimensions,
    local_embedding_revision: str | None = None,
    paper_index_policy: tuple[tuple[str, Any], ...] = (),
    paper_index_environment: tuple[tuple[str, str], ...] = (),
    paper_query_policy: tuple[tuple[str, str | None, Any], ...] = (),
) -> StrategySpec:
    return StrategySpec(
        name=name,
        primary=primary,
        external=True,
        repository=repository,
        revision=revision,
        worker=worker,
        output_env=f"RAG_{name.upper()}_OUTPUT_ROOT",
        output_default=f"data/{name}_output",
        driver=driver,
        license_note=license_note,
        adapter_variant=adapter_variant,
        transport_profile=transport_profile,
        paper_embedding_model=paper_embedding_model,
        paper_generation_seed=paper_generation_seed,
        paper_embedding_dimensions=paper_embedding_dimensions,
        local_embedding_revision=local_embedding_revision,
        paper_index_policy=paper_index_policy,
        paper_index_environment=paper_index_environment,
        paper_query_policy=paper_query_policy,
    )


_GFM_RUNTIME = runtime_requirement('gfm_rag')
_LINEAR_RUNTIME = runtime_requirement('linear_rag')


_CORE_QUERY_POLICY: tuple[tuple[str, str | None, Any], ...] = (
    ("q_minus", None, True),
    ("q_plus", None, True),
    ("sentence_channel_enabled", None, False),
    ("chunk_sentences", None, 6),
    ("questions_per_direction", None, 3),
    ("graph_hop_depth", None, 1),
    ("graph_path_decay", None, 0.5),
    ("graph_edge_variant", "RAG_GRAPH_EDGE_VARIANT", "full"),
    ("hop_edge_filter", None, "none"),
    ("hop_seed_policy", None, "all"),
    ("qplus_hop_activation", None, "owner"),
    ("continuation_edges_enabled", None, False),
    ("continuation_anchor_policy", None, "named_only"),
    ("hop_semantic_variant", "RAG_HOP_SEMANTIC_VARIANT", "body_only"),
    ("question_schema", None, "legacy"),
    ("precompute_reciprocal_hops", None, True),
    ("default_top_k", None, 12),
    ("candidate_pool_multiplier", None, 1),
    ("fulltext_analyzer", "NEO4J_FULLTEXT_ANALYZER", "english"),
    ("hypo_channel_variant", "RAG_HYPO_CHANNEL_VARIANT", "full"),
    ("source_selection_variant", "RAG_SOURCE_SELECTION_VARIANT", "role_body_list_ranking"),
    ("candidate_order_input_order", None, "search"),
    ("candidate_order_shuffle_seed", None, 0),
    ("final_rank_variant", "RAG_FINAL_RANK_VARIANT", "fused"),
)


_CORE_DEFAULTS = {field: value for field, _, value in _CORE_QUERY_POLICY}

STRATEGIES = (
    StrategySpec(
        "prehop",
        primary=True,
        paper_index_policy=(
            ("structured_output_profile", "prehop-json-schema-v3"),
            ("chunk_sentences", _CORE_DEFAULTS["chunk_sentences"]),
            ("default_top_k", _CORE_DEFAULTS["default_top_k"]),
            ("questions_per_direction", _CORE_DEFAULTS["questions_per_direction"]),
            ("question_schema", _CORE_DEFAULTS["question_schema"]),
            ("q_minus_enabled", _CORE_DEFAULTS["q_minus"]),
            ("q_plus_enabled", _CORE_DEFAULTS["q_plus"]),
            ("sentence_channel_enabled", _CORE_DEFAULTS["sentence_channel_enabled"]),
            ("fulltext_analyzer", _CORE_DEFAULTS["fulltext_analyzer"]),
            ("precompute_reciprocal_hops", _CORE_DEFAULTS["precompute_reciprocal_hops"]),
            ("continuation_edges_materialized", False),
            ("continuation_anchor_policy", _CORE_DEFAULTS["continuation_anchor_policy"]),
            ("hop_construction", "qplus_to_qminus_owner"),
        ),
        paper_query_policy=_CORE_QUERY_POLICY + (("structured_output_profile", None, "prehop-json-schema-v3"),),
    ),
    StrategySpec(
        "naive",
        primary=True,
        paper_index_policy=(
            ("structured_output_profile", "prehop-json-schema-v3"),
            ("chunk_sentences", _CORE_DEFAULTS["chunk_sentences"]),
            ("default_top_k", _CORE_DEFAULTS["default_top_k"]),
            ("question_schema", _CORE_DEFAULTS["question_schema"]),
            ("q_minus_enabled", _CORE_DEFAULTS["q_minus"]),
            ("q_plus_enabled", _CORE_DEFAULTS["q_plus"]),
            ("sentence_channel_enabled", _CORE_DEFAULTS["sentence_channel_enabled"]),
            ("precompute_reciprocal_hops", _CORE_DEFAULTS["precompute_reciprocal_hops"]),
            ("fulltext_analyzer", _CORE_DEFAULTS["fulltext_analyzer"]),
        ),
        paper_query_policy=_CORE_QUERY_POLICY + (("structured_output_profile", None, "prehop-json-schema-v3"),),
    ),
    StrategySpec("hoprag", primary=True,
        repository="https://github.com/LIU-Hao-2002/HopRAG.git",
        revision="a6e425b8f8a5d8131dd7805db40185ac76e09903",
        output_env="RAG_HOP_OUTPUT_ROOT", output_default="data/hoprag_output",
        paper_generation_seed=None,
        paper_index_policy=(("native_observation_profile", "adapter-json-recovery-v2"), ("adapter_response_attempts", 1),
                            ("edge_input_scope", "whole-corpus-without-query-or-gold"),
                            ("native_chunk_workers", 1), ("document_workers", 10), ("native_retry_attempts", 2),
                            ("pos_tagger", "paddlenlp-2.8.1-pos_tagging")),
        paper_index_environment=(
            ('document_workers', 'RAG_HOP_DOC_WORKERS'),
        ),
        paper_query_policy=(("max_hop", None, 5), ("topk", None, 8),
                            ("entry_type", None, "node"), ("tol", None, 20),
                            ("traversal", None, "bfs"), ("mode", None, "common"))),
    StrategySpec(
        "ms_graphrag",
        primary=True,
        revision="pypi:3.1.2",
        output_env="RAG_MS_OUTPUT_ROOT",
        output_default="data/ms_graphrag_output",
        transport_profile="openai_compatible_litellm",
        paper_index_policy=(
            ("extraction_validation_profile", "native-observation-v1"),
            ("extract_max_tokens", None),
            ("query_max_tokens", None),
            ("length_retry_policy", "fail-without-identical-retry"),
            ("report_max_tokens", None),
            ("local_search_top_k_entities", 10),
            ("local_search_top_k_relationships", 10),
            ("local_search_max_context_tokens", 12000),
        ),
    ),
    _external(
        "lightrag",
        "https://github.com/HKUDS/LightRAG.git",
        "440d25b0cbfdd94b3c92f7bfb0c3c2989c779671",
        primary=True,
        worker="research_baseline_worker.py",
        driver="models.external_research.drivers.lightrag:LightRAGDriver",
        paper_index_policy=(
            ("backbone_mode", "official_faithful"),
            ("native_retrieval", True),
            ("query_mode", "mix"),
            ("retrieval_top_k", 40),
            ("chunk_top_k", 20),
            ("embedding_max_token_size", 8192),
        ),
        paper_index_environment=(
            ('query_mode', 'RAG_LIGHTRAG_QUERY_MODE'),
            ('retrieval_top_k', 'RAG_LIGHTRAG_TOP_K'),
            ('chunk_top_k', 'RAG_LIGHTRAG_CHUNK_TOP_K'),
            ('embedding_max_token_size', 'RAG_EMBEDDING_MAX_TOKENS'),
        ),
    ),
    _external(
        "gfm_rag",
        "https://github.com/RManLuo/gfm-rag.git",
        "5354731bb68d23a7548eb1ef79f6dc067c5a21e6",
        primary=True,
        worker="research_baseline_worker.py",
        driver="models.external_research.drivers.gfm_rag:GFMRAGDriver",
        paper_embedding_model="checkpoint-defined",
        paper_embedding_dimensions=None,
        paper_index_policy=(
            ("extraction_validation_profile", "native-observation-v1"),
            ("backbone_mode", "official_faithful"),
            ("native_retrieval", True),
            ("retrieval_top_k", 5),
            ("ner_model", "official_hydra_qa_ircot"),
            ("query_mode", "native_single_pass_qa"),
            ("qa_generation_profile", "native-qa-inference-v1"),
            ("qa_max_tokens", None),
            ("entity_linker", "official_hydra_qa_ircot"),
            ("entity_linker_model", _GFM_RUNTIME["entity_linker_model"]),
            ("entity_linker_revision", _GFM_RUNTIME["entity_linker_revision"]),
            ("gfm_checkpoint_model", _GFM_RUNTIME["checkpoint_model"]),
            ("gfm_checkpoint_revision", _GFM_RUNTIME["checkpoint_revision"]),
        ),
        paper_index_environment=(
            ('retrieval_top_k', 'RAG_GFM_RAG_TOP_K'),
        ),
    ),
    _external(
        "linear_rag",
        "https://github.com/DEEP-PolyU/LinearRAG.git",
        "bcc94e66c221f798801255efba09311d6fbcd8d6",
        primary=True,
        worker="research_baseline_worker.py",
        driver="models.external_research.drivers.linear_rag:LinearRAGDriver",
        license_note="GPL upstream remains process-isolated",
        paper_embedding_model=_LINEAR_RUNTIME["embedding_model"],
        paper_embedding_dimensions=768,
        local_embedding_revision=_LINEAR_RUNTIME["embedding_revision"],
        paper_index_policy=(
            ("index_validation_profile", "strict-linear-stores-v1"),
            ("backbone_mode", "official_faithful"),
            ("native_retrieval", True),
            ("spacy_model", "en_core_web_trf"),
            ("official_embedding_model", _LINEAR_RUNTIME["embedding_model"]),
            ("official_embedding_revision", _LINEAR_RUNTIME["embedding_revision"]),
            ("retrieval_top_k", 5),
            ("vectorized_retrieval", False),
        ),
        paper_index_environment=(
            ('spacy_model', 'RAG_LINEAR_RAG_SPACY_MODEL'),
            ('retrieval_top_k', 'RAG_LINEAR_RAG_TOP_K'),
            ('vectorized_retrieval', 'RAG_LINEAR_RAG_VECTORIZED'),
        ),
        paper_query_policy=(
            ("iteration_threshold", None, 0.4),
            ("passage_ratio", None, 2.0),
            ("top_k_sentence", None, 3),
        ),
    ),
)

BY_NAME = {spec.name: spec for spec in STRATEGIES}
ALL_STRATEGIES = tuple(BY_NAME)
PRIMARY_STRATEGIES = tuple(spec.name for spec in STRATEGIES if spec.primary)
EXTERNAL_STRATEGIES = tuple(spec.name for spec in STRATEGIES if spec.external)
RESEARCH_EXTERNAL_STRATEGIES = tuple(spec.name for spec in STRATEGIES if spec.driver)


def get_strategy(name: str) -> StrategySpec:
    try:
        return BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown strategy: {name}") from exc


def method_setting(strategy: str, field: str) -> str | int | bool:
    """Resolve active overrides from the same registry used for provenance."""
    from core.semantic_config import parse_strict_bool
    spec = get_strategy(strategy)
    expected = dict(spec.paper_index_policy)[field]
    name = dict(spec.paper_index_environment).get(field)
    raw = os.environ.get(name) if name else None
    if raw is None or not raw.strip():
        return expected
    if isinstance(expected, bool):
        try:
            normalized: str | int | bool = parse_strict_bool(raw, name=name)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
    elif isinstance(expected, int):
        normalized = int(raw)
    else:
        normalized = raw.strip()
    return normalized


def paper_environment_defaults() -> dict[str, str]:
    """Non-secret shell defaults derived from the canonical transport policy."""
    fields = {
        "RAG_INFERENCE_RETRY_ATTEMPTS": "retry_attempts",
        "RAG_GENERATION_CONCURRENCY": "generation_concurrency",
        "RAG_EMBEDDING_BATCH_SIZE": "embedding_batch_size",
        "RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS": "embedding_concurrency",
        "RAG_INFERENCE_TIMEOUT": "timeout_seconds",
        "RAG_BENCHMARK_CONCURRENCY": "benchmark_concurrency",
        "RAG_BENCHMARK_CHECKPOINT_EVERY": "benchmark_checkpoint_every",
        "RAG_BENCHMARK_SEEDS": "benchmark_seed",
        "RAG_SEED": "benchmark_seed",
        "PYTHONHASHSEED": "benchmark_seed",
    }
    return {name: str(getattr(PAPER_TRANSPORT, field)) for name, field in fields.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--primary-lines", action="store_true")
    group.add_argument("--all-lines", action="store_true")
    group.add_argument("--external-tsv", action="store_true")
    group.add_argument("--paper-defaults-tsv", action="store_true")
    group.add_argument("--local-model-tsv", action="store_true")
    group.add_argument("--is-primary")
    group.add_argument("--is-external")
    group.add_argument("--is-valid")
    args = parser.parse_args()
    if args.paper_defaults_tsv:
        for name, value in paper_environment_defaults().items():
            print(f"{name}\t{value}")
        return 0
    if args.is_primary is not None:
        return 0 if args.is_primary in PRIMARY_STRATEGIES else 1
    if args.is_external is not None:
        return 0 if args.is_external in EXTERNAL_STRATEGIES else 1
    if args.is_valid is not None:
        return 0 if args.is_valid in ALL_STRATEGIES else 1
    if args.external_tsv:
        for name in EXTERNAL_STRATEGIES:
            spec = BY_NAME[name]
            print(f"{name}\t{spec.repository}\t{spec.revision}")
        return 0
    if args.local_model_tsv:
        for spec in STRATEGIES:
            if spec.local_embedding_revision:
                print(f"{spec.name}\t{spec.paper_embedding_model}\t{spec.local_embedding_revision}")
        return 0
    for name in PRIMARY_STRATEGIES if args.primary_lines else ALL_STRATEGIES:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
