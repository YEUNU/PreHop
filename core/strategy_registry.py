"""Typed, dependency-free strategy registry used by Python and shell entrypoints."""

from __future__ import annotations

import argparse
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


# Defaults are independent of import order. Runtime profile overrides are
# resolved by core.execution_profile at the point of use.
PAPER_TRANSPORT = PaperTransportSpec()


@dataclass(frozen=True)
class StrategySpec:
    name: str
    paper_generation_model: str = PAPER_TRANSPORT.generation_model
    paper_index_policy: tuple[tuple[str, Any], ...] = ()
    # (provenance field, controlling environment variable, canonical value).
    # Query policies also describe historical ablation controls.
    paper_query_policy: tuple[tuple[str, str | None, Any], ...] = ()


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
)

BY_NAME = {spec.name: spec for spec in STRATEGIES}
ALL_STRATEGIES = tuple(BY_NAME)
# Both retained strategies are paper-primary targets; shell launchers keep the
# primary/valid distinction so the two matrices stay explicit.
PRIMARY_STRATEGIES = ALL_STRATEGIES


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
    group.add_argument("--paper-defaults-tsv", action="store_true")
    group.add_argument("--is-primary")
    group.add_argument("--is-valid")
    args = parser.parse_args()
    if args.paper_defaults_tsv:
        for name, value in paper_environment_defaults().items():
            print(f"{name}\t{value}")
        return 0
    if args.is_primary is not None:
        return 0 if args.is_primary in PRIMARY_STRATEGIES else 1
    if args.is_valid is not None:
        return 0 if args.is_valid in ALL_STRATEGIES else 1
    for name in PRIMARY_STRATEGIES:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
