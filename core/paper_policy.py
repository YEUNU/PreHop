"""Checked-in semantic policy contract for primary paper indexes."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from core.runtime_requirements import runtime_requirement
from core.strategy_registry import PAPER_TRANSPORT, get_strategy

PAPER_EMBEDDING_QUERY_INSTRUCTION = PAPER_TRANSPORT.query_instruction
PAPER_EMBEDDING_QUERY_TEMPLATE = PAPER_TRANSPORT.query_template
PAPER_EMBEDDING_MAX_INPUT_TOKENS = PAPER_TRANSPORT.embedding_max_input_tokens


_METHOD_PREFIXES = {
    "hoprag": ("RAG_HOP_",),
    "ms_graphrag": ("RAG_MS_",), "lightrag": ("RAG_LIGHTRAG_",),
    "gfm_rag": ("RAG_GFM_RAG_",),
    "linear_rag": ("RAG_LINEAR_RAG_",),
}

# These configure Prehop graph traversal/building, not the HopRAG adapter.
_PREHOP_HOP_ENVIRONMENT = {"RAG_HOP_GATHER_WAVE", "RAG_HOP_BUILD_CONCURRENCY",
                           "RAG_HOP_SEMANTIC_VARIANT", "RAG_HOP_EDGE_FILTER"}


def method_environment_defaults(env_path: Path | None = None) -> dict[str, str]:
    """Freeze registry defaults and unknown dotenv method keys without reading values."""
    defaults = {}
    allowed = set()
    for strategy in _METHOD_PREFIXES:
        spec = get_strategy(strategy)
        for _field, name, value in spec.paper_index_environment:
            defaults[name] = str(value).lower() if isinstance(value, bool) else str(value)
        allowed.update({spec.output_env, f"RAG_{strategy.upper()}_ROOT", f"RAG_{strategy.upper()}_PYTHON"})
    allowed.update(defaults)
    allowed.update(_PREHOP_HOP_ENVIRONMENT)
    path = env_path if env_path is not None else Path(__file__).resolve().parents[1] / ".env"
    prefixes = tuple(prefix for values in _METHOD_PREFIXES.values() for prefix in values)
    if path.is_file():
        with path.open() as stream:
            for line in stream:
                match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
                if match and match[1].startswith(prefixes) and match[1] not in allowed:
                    defaults[match[1]] = ""
    return defaults


def preserve_method_environment(environment=None, env_path: Path | None = None) -> None:
    environment = os.environ if environment is None else environment
    for name, value in method_environment_defaults(env_path).items():
        environment.setdefault(name, value)


def configure_target_environment(strategy: str, dataset: str, run_id: str) -> None:
    """Resolve target identity before standalone admission reads live policy."""
    from core.execution_profile import apply_execution_profile
    from core.inference_transport import preserve_provider_environment
    from core.strategy_registry import paper_environment_defaults
    apply_execution_profile()
    spec = get_strategy(strategy)
    preserve_provider_environment()
    preserve_method_environment()
    for name, value in paper_environment_defaults().items():
        os.environ.setdefault(name, value)
    os.environ.update({
        "RAG_PAPER_MODE": "true", "RAG_RUN_ID": run_id,
        "RAG_INDEX_NAMESPACE": f"{dataset}_{run_id}",
        "RAG_INDEX_STATS_PATH": f"data/index_stats/{strategy}_{dataset}_{run_id}.json",
        "RAG_CHUNK_CACHE_DIR": f"data/index_cache/runs/{run_id}/{strategy}/{dataset}",
    })
    if spec.paper_generation_seed is None:
        os.environ["RAG_LLM_SEED"] = ""
    else:
        os.environ["RAG_LLM_SEED"] = str(spec.paper_generation_seed)
    os.environ["EMBEDDING_QUERY_INSTRUCTION"] = str(dict(spec.paper_index_policy).get("embedding_query_instruction", PAPER_TRANSPORT.query_instruction))
    if spec.output_env and spec.output_default:
        os.environ[spec.output_env] = f"{spec.output_default}/runs/{run_id}"


def structured_query_identity(strategy: str) -> dict[str, str]:
    if strategy not in {"prehop", "naive"}:
        return {}
    from core.structured_outputs import PREHOP_STRUCTURED_PROFILE, structured_bundle_sha256

    return {"structured_output_profile": PREHOP_STRUCTURED_PROFILE,
            "structured_schema_bundle_sha256": structured_bundle_sha256()}


def canonical_query_policy(strategy: str) -> dict[str, Any]:
    """Return the sole checked-in benchmark policy for a core strategy."""
    policy = {field: value for field, _environment, value in get_strategy(strategy).paper_query_policy}
    if strategy == "prehop":
        policy["query_execution"] = "original-question-single-retrieval-v1"
    policy.update(structured_query_identity(strategy))
    from core.paper_compatibility import method_identity
    policy.update(method_identity(strategy))
    return policy


def canonical_semantic_index_policy(strategy: str, dataset: str) -> dict[str, Any]:
    spec = get_strategy(strategy)
    local_embedding = spec.local_embedding_revision is not None or strategy == "gfm_rag"
    policy: dict[str, Any] = {
        "strategy": strategy,
        "indexing_model": None if strategy == "naive" else spec.paper_generation_model,
        "generation_revision": spec.paper_generation_model,
        "generation_seed": spec.paper_generation_seed,
        "embedding_model": spec.paper_embedding_model,
        "embedding_revision": (
            spec.local_embedding_revision
            if spec.local_embedding_revision is not None
            else (None if strategy == "gfm_rag" else spec.paper_embedding_model)
        ),
        "embedding_query_instruction": None if local_embedding else PAPER_EMBEDDING_QUERY_INSTRUCTION,
        "embedding_query_template": None if local_embedding else PAPER_EMBEDDING_QUERY_TEMPLATE,
        "embedding_dimensions": spec.paper_embedding_dimensions,
        "embedding_max_input_tokens": PAPER_EMBEDDING_MAX_INPUT_TOKENS,
        "embedding_token_reserve": 0,
        "generation_max_context_tokens": PAPER_TRANSPORT.generation_context_tokens,
        "fulltext_analyzer": "english",
    }
    if spec.revision is not None:
        policy["official_revision"] = spec.revision
    from core.paper_compatibility import index_method_identity
    policy.update(index_method_identity(strategy))
    policy.update(dict(spec.paper_index_policy))
    if strategy in {"prehop", "naive"}:
        from core.structured_outputs import PREHOP_STRUCTURED_PROFILE, structured_index_bundle_sha256
        policy.update(structured_output_profile=PREHOP_STRUCTURED_PROFILE,
                      structured_schema_bundle_sha256=structured_index_bundle_sha256())
    if strategy == "gfm_rag":
        approved = runtime_requirement("gfm_rag").get("checkpoint_file_sha256", {})
        if not isinstance(approved, dict):
            raise RuntimeError("GFM approved checkpoint digest manifest is malformed")
        policy.update(
            {
                "gfm_checkpoint_sha256": approved.get("model.pth"),
                "gfm_config_sha256": approved.get("config.json"),
            }
        )
    return policy


def canonical_operational_policy(strategy: str) -> dict[str, Any]:
    """Resolve the one non-secret effective transport contract."""
    from core.inference_transport import InferenceTransport
    from core.runtime_requirements import runtime_identity

    return {**InferenceTransport.resolve(strategy).policy_dict(), **runtime_identity(strategy)}
