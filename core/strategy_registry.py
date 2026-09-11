"""Typed, dependency-free strategy registry used by Python and shell entrypoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
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
    from execution_profile import execution_profile, transport_settings
else:
    from core.execution_profile import execution_profile, transport_settings

PAPER_TRANSPORT = replace(PaperTransportSpec(), **{
    key: value for key, value in transport_settings(execution_profile()).items()
    if key in PaperTransportSpec.__dataclass_fields__
})


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
    # (semantic field, controlling environment variable, canonical value).
    paper_index_environment: tuple[tuple[str, str, Any], ...] = ()
    # (provenance field, controlling environment variable, canonical value).
    # Keeping all three together prevents preflight and admission from
    # drifting into separate, partial method-policy lists.
    paper_query_policy: tuple[tuple[str, str | None, Any], ...] = ()


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
    paper_index_environment: tuple[tuple[str, str, Any], ...] = (),
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


_CORE_QUERY_POLICY: tuple[tuple[str, str | None, Any], ...] = (
    ("q_minus", "RAG_ABLATION_Q_MINUS", True),
    ("q_plus", "RAG_ABLATION_Q_PLUS", True),
    ("sentence_channel_enabled", "RAG_SENTENCE_CHANNEL_ENABLED", False),
    ("chunk_sentences", None, 6),
    ("questions_per_direction", None, 3),
    ("graph_hop_depth", "RAG_GRAPH_HOP_DEPTH", 1),
    ("graph_path_decay", "RAG_GRAPH_PATH_DECAY", 0.5),
    ("graph_edge_variant", "RAG_GRAPH_EDGE_VARIANT", "full"),
    ("hop_edge_filter", "RAG_HOP_EDGE_FILTER", "none"),
    ("hop_seed_policy", "RAG_HOP_SEED_POLICY", "all"),
    ("qplus_hop_activation", "RAG_QPLUS_HOP_ACTIVATION", "owner"),
    ("continuation_edges_enabled", "RAG_CONTINUATION_EDGES_ENABLED", False),
    ("continuation_anchor_policy", "RAG_CONTINUATION_ANCHOR_POLICY", "named_only"),
    ("hop_semantic_variant", "RAG_HOP_SEMANTIC_VARIANT", "body_only"),
    ("question_schema", "RAG_QUESTION_SCHEMA", "legacy"),
    ("precompute_reciprocal_hops", "RAG_PRECOMPUTE_RECIPROCAL_HOPS", True),
    ("default_top_k", None, 12),
    ("candidate_pool_multiplier", "RAG_CANDIDATE_POOL_MULTIPLIER", 1),
    ("fulltext_analyzer", "NEO4J_FULLTEXT_ANALYZER", "english"),
    ("hypo_channel_variant", "RAG_HYPO_CHANNEL_VARIANT", "full"),
    ("source_selection_variant", "RAG_SOURCE_SELECTION_VARIANT", "role_body_list_ranking"),
    ("candidate_order_input_order", "RAG_CANDIDATE_ORDER_INPUT_ORDER", "search"),
    ("candidate_order_shuffle_seed", "RAG_CANDIDATE_ORDER_SHUFFLE_SEED", 0),
    ("final_rank_variant", "RAG_FINAL_RANK_VARIANT", "fused"),
)


STRATEGIES = (
    StrategySpec(
        "prehop",
        primary=True,
        paper_index_policy=(
            ("structured_output_profile", "prehop-json-schema-v3"),
            ("chunk_sentences", 6),
            ("default_top_k", 12),
            ("questions_per_direction", 3),
            ("question_schema", "legacy"),
            ("q_minus_enabled", True),
            ("q_plus_enabled", True),
            ("sentence_channel_enabled", False),
            ("fulltext_analyzer", "english"),
            ("precompute_reciprocal_hops", True),
            ("continuation_edges_materialized", False),
            ("continuation_anchor_policy", "named_only"),
            ("hop_construction", "qplus_to_qminus_owner"),
        ),
        paper_query_policy=_CORE_QUERY_POLICY + (("structured_output_profile", None, "prehop-json-schema-v3"),),
    ),
    StrategySpec(
        "naive",
        primary=True,
        paper_index_policy=(
            ("structured_output_profile", "prehop-json-schema-v3"),
            ("chunk_sentences", 6),
            ("default_top_k", 12),
            ("question_schema", "legacy"),
            ("q_minus_enabled", True),
            ("q_plus_enabled", True),
            ("sentence_channel_enabled", False),
            ("precompute_reciprocal_hops", True),
            ("fulltext_analyzer", "english"),
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
        paper_index_environment=(("document_workers", "RAG_HOP_DOC_WORKERS", 10),),
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
        paper_index_environment=(
            ("embedding_dimensions", "RAG_MS_EMBED_DIM", 2560),
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
            ("backbone_mode", "RAG_LIGHTRAG_BACKBONE_MODE", "official_faithful"),
            ("native_retrieval", "RAG_LIGHTRAG_NATIVE_RETRIEVAL", True),
            ("query_mode", "RAG_LIGHTRAG_QUERY_MODE", "mix"),
            ("retrieval_top_k", "RAG_LIGHTRAG_TOP_K", 40),
            ("chunk_top_k", "RAG_LIGHTRAG_CHUNK_TOP_K", 20),
            ("embedding_max_token_size", "RAG_EMBEDDING_MAX_TOKENS", 8192),
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
            ("entity_linker_model", "colbert-ir/colbertv2.0"),
            ("entity_linker_revision", "c1e84128e85ef755c096a95bdb06b47793b13acf"),
            ("gfm_checkpoint_model", "rmanluo/GFM-RAG-8M"),
            ("gfm_checkpoint_revision", "4da9e4655d126a783ae2b795ab73b7c7a7c3f4ac"),
        ),
        paper_index_environment=(
            ("backbone_mode", "RAG_GFM_RAG_BACKBONE_MODE", "official_faithful"),
            ("native_retrieval", "RAG_GFM_RAG_NATIVE_RETRIEVAL", True),
            ("retrieval_top_k", "RAG_GFM_RAG_TOP_K", 5),
            ("ner_model", "RAG_GFM_RAG_NER_MODEL", "official_hydra_qa_ircot"),
            ("entity_linker", "RAG_GFM_RAG_ENTITY_LINKER", "official_hydra_qa_ircot"),
            ("entity_linker_model", "RAG_GFM_RAG_COLBERT_MODEL", "colbert-ir/colbertv2.0"),
            (
                "entity_linker_revision",
                "RAG_GFM_RAG_COLBERT_REVISION",
                "c1e84128e85ef755c096a95bdb06b47793b13acf",
            ),
            ("gfm_checkpoint_model", "RAG_GFM_RAG_CHECKPOINT_MODEL", "rmanluo/GFM-RAG-8M"),
            (
                "gfm_checkpoint_revision",
                "RAG_GFM_RAG_CHECKPOINT_REVISION",
                "4da9e4655d126a783ae2b795ab73b7c7a7c3f4ac",
            ),
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
        paper_embedding_model="sentence-transformers/all-mpnet-base-v2",
        paper_embedding_dimensions=768,
        local_embedding_revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
        paper_index_policy=(
            ("index_validation_profile", "strict-linear-stores-v1"),
            ("backbone_mode", "official_faithful"),
            ("native_retrieval", True),
            ("spacy_model", "en_core_web_trf"),
            ("official_embedding_model", "sentence-transformers/all-mpnet-base-v2"),
            ("official_embedding_revision", "e8c3b32edf5434bc2275fc9bab85f82640a19130"),
            ("retrieval_top_k", 5),
            ("vectorized_retrieval", False),
        ),
        paper_index_environment=(
            ("backbone_mode", "RAG_LINEAR_RAG_BACKBONE_MODE", "official_faithful"),
            ("native_retrieval", "RAG_LINEAR_RAG_NATIVE_RETRIEVAL", True),
            ("spacy_model", "RAG_LINEAR_RAG_SPACY_MODEL", "en_core_web_trf"),
            (
                "official_embedding_model",
                "RAG_LINEAR_RAG_MPNET_MODEL",
                "sentence-transformers/all-mpnet-base-v2",
            ),
            (
                "official_embedding_revision",
                "RAG_LINEAR_RAG_MPNET_REVISION",
                "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            ),
            ("retrieval_top_k", "RAG_LINEAR_RAG_TOP_K", 5),
            ("vectorized_retrieval", "RAG_LINEAR_RAG_VECTORIZED", False),
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
    group.add_argument("--output-tsv", action="store_true")
    group.add_argument("--paper-defaults-tsv", action="store_true")
    group.add_argument("--generation-seed")
    group.add_argument("--query-instruction")
    group.add_argument("--local-model-tsv", action="store_true")
    group.add_argument("--is-primary")
    group.add_argument("--is-external")
    group.add_argument("--is-valid")
    args = parser.parse_args()
    if args.paper_defaults_tsv:
        for name, value in paper_environment_defaults().items():
            print(f"{name}\t{value}")
        return 0
    if args.query_instruction is not None:
        spec = get_strategy(args.query_instruction)
        print(dict(spec.paper_index_policy).get("embedding_query_instruction", PAPER_TRANSPORT.query_instruction))
        return 0
    if args.generation_seed is not None:
        value = get_strategy(args.generation_seed).paper_generation_seed
        print("" if value is None else value)
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
    if args.output_tsv:
        for spec in STRATEGIES:
            if spec.output_env and spec.output_default:
                print(f"{spec.name}\t{spec.output_env}\t{spec.output_default}")
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
