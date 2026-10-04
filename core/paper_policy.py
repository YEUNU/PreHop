"""Checked-in semantic policy contract for primary paper indexes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from core.strategy_registry import PAPER_TRANSPORT, get_strategy

PAPER_EMBEDDING_QUERY_INSTRUCTION = PAPER_TRANSPORT.query_instruction
PAPER_EMBEDDING_QUERY_TEMPLATE = PAPER_TRANSPORT.query_template
PAPER_EMBEDDING_MAX_INPUT_TOKENS = PAPER_TRANSPORT.embedding_max_input_tokens


def resolved_target_environment(
    strategy: str, dataset: str, run_id: str, environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return target settings without changing the caller's process environment."""
    from core.execution_profile import resolved_execution_environment
    from core.inference_transport import preserve_provider_environment
    environment = resolved_execution_environment(environment)
    get_strategy(strategy)
    preserve_provider_environment(environment)
    environment.update({
        "RAG_PAPER_MODE": "true", "RAG_RUN_ID": run_id,
        "RAG_INDEX_NAMESPACE": f"{dataset}_{run_id}",
        "RAG_INDEX_STATS_PATH": f"data/index_stats/{strategy}_{dataset}_{run_id}.json",
        "RAG_CHUNK_CACHE_DIR": f"data/index_cache/runs/{run_id}/{strategy}/{dataset}",
        # Paper generation omits the LLM seed for every retained strategy.
        "RAG_LLM_SEED": "",
        "EMBEDDING_QUERY_INSTRUCTION": PAPER_TRANSPORT.query_instruction,
    })
    return environment


def configure_target_environment(strategy: str, dataset: str, run_id: str) -> None:
    """Apply the resolved settings only when launching a target."""
    os.environ.update(resolved_target_environment(strategy, dataset, run_id))


def structured_query_identity(strategy: str) -> dict[str, str]:
    get_strategy(strategy)
    from core.structured_outputs import PREHOP_STRUCTURED_PROFILE, structured_bundle_sha256

    return {"structured_output_profile": PREHOP_STRUCTURED_PROFILE,
            "structured_schema_bundle_sha256": structured_bundle_sha256()}


def canonical_query_policy(strategy: str, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return the sole checked-in benchmark policy for a core strategy."""
    policy = {field: value for field, _environment, value in get_strategy(strategy).paper_query_policy}
    if strategy == "prehop":
        policy["query_execution"] = "original-question-single-retrieval-v1"
    policy.update(structured_query_identity(strategy))
    from core.paper_compatibility import method_identity
    policy.update(method_identity(strategy, environment))
    return policy


def canonical_semantic_index_policy(strategy: str, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    spec = get_strategy(strategy)
    policy: dict[str, Any] = {
        "strategy": strategy,
        "indexing_model": None if strategy == "naive" else spec.paper_generation_model,
        "generation_revision": spec.paper_generation_model,
        "generation_seed": None,
        "embedding_model": PAPER_TRANSPORT.embedding_model,
        "embedding_revision": PAPER_TRANSPORT.embedding_model,
        "embedding_query_instruction": PAPER_EMBEDDING_QUERY_INSTRUCTION,
        "embedding_query_template": PAPER_EMBEDDING_QUERY_TEMPLATE,
        "embedding_dimensions": PAPER_TRANSPORT.embedding_dimensions,
        "embedding_max_input_tokens": PAPER_EMBEDDING_MAX_INPUT_TOKENS,
        "embedding_token_reserve": PAPER_TRANSPORT.embedding_token_reserve,
        "generation_max_context_tokens": PAPER_TRANSPORT.generation_context_tokens,
        "fulltext_analyzer": "english",
    }
    from core.paper_compatibility import index_method_identity
    policy.update(index_method_identity(strategy, environment))
    policy.update(dict(spec.paper_index_policy))
    from core.structured_outputs import PREHOP_STRUCTURED_PROFILE, structured_index_bundle_sha256
    policy.update(structured_output_profile=PREHOP_STRUCTURED_PROFILE,
                  structured_schema_bundle_sha256=structured_index_bundle_sha256())
    return policy


def canonical_operational_policy(strategy: str, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Resolve the one non-secret effective transport contract."""
    from core.inference_transport import InferenceTransport
    from core.runtime_requirements import runtime_identity

    return {**InferenceTransport.resolve(strategy, environment).policy_dict(), **runtime_identity(strategy, environment)}
