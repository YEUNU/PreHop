"""Checked-in semantic policy contract for primary paper indexes."""

from __future__ import annotations

import os
import re
import string
from pathlib import Path
from typing import Any

from core.runtime_requirements import runtime_requirement
from core.semantic_config import parse_strict_bool, semantic_config_sha256, semantic_index_policy
from core.strategy_registry import PAPER_TRANSPORT, get_strategy

PAPER_EMBEDDING_QUERY_INSTRUCTION = PAPER_TRANSPORT.query_instruction
PAPER_EMBEDDING_QUERY_TEMPLATE = PAPER_TRANSPORT.query_template
PAPER_EMBEDDING_MAX_INPUT_TOKENS = PAPER_TRANSPORT.embedding_max_input_tokens


_METHOD_PREFIXES = {
    "ms_graphrag": ("RAG_MS_",), "lightrag": ("RAG_LIGHTRAG_",),
    "hipporag2": ("RAG_HIPPORAG2_",), "gfm_rag": ("RAG_GFM_RAG_",),
    "linear_rag": ("RAG_LINEAR_RAG_",), "youtu_graphrag": ("RAG_YOUTU_",),
}


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


def _strict_env_value(name: str, expected: str | float | bool) -> None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return
    if isinstance(expected, bool):
        try:
            observed: str | float | bool = parse_strict_bool(raw, name=name)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
    elif isinstance(expected, int):
        try:
            observed = int(raw)
        except ValueError as exc:
            raise RuntimeError(f"{name} must be an integer") from exc
    elif isinstance(expected, float):
        try:
            observed = float(raw)
        except ValueError as exc:
            raise RuntimeError(f"{name} must be a number") from exc
    else:
        observed = raw.strip()
    if observed != expected:
        raise RuntimeError(f"{name} differs from the checked-in paper semantic policy")


def validate_paper_semantic_environment(strategy: str, dataset: str) -> None:
    """Reject caller overrides that would diverge from recorded paper policy."""
    policy = canonical_semantic_index_policy(strategy, dataset)
    mappings: dict[str, str | float | bool] = {
        "MAX_EMBEDDING_LENGTH": policy["embedding_max_input_tokens"],
        "RAG_MAX_CONTEXT_LENGTH": policy["generation_max_context_tokens"],
        "RAG_EMBEDDING_TOKEN_RESERVE": policy["embedding_token_reserve"],
        "NEO4J_FULLTEXT_ANALYZER": policy["fulltext_analyzer"],
    }
    if policy["embedding_query_instruction"] is not None:
        mappings["EMBEDDING_QUERY_INSTRUCTION"] = policy["embedding_query_instruction"]
        if policy["embedding_dimensions"] is not None:
            mappings["NEO4J_VECTOR_DIMENSIONS"] = policy["embedding_dimensions"]
    spec = get_strategy(strategy)
    mappings.update({environment: expected for _field, environment, expected in spec.paper_index_environment})
    for name, expected in mappings.items():
        _strict_env_value(name, expected)
    if strategy in {"prehop", "naive"}:
        for _field, name, expected in spec.paper_query_policy:
            if name is not None:
                _strict_env_value(name, expected)
    method_prefixes = _METHOD_PREFIXES.get(strategy, ())
    allowed = set(mappings)
    allowed.update(
        {
            spec.output_env,
            f"RAG_{strategy.upper()}_ROOT",
            f"RAG_{strategy.upper()}_PYTHON",
        }
    )
    unknown = sorted(
        name
        for name, value in os.environ.items()
        if value.strip() and any(name.startswith(prefix) for prefix in method_prefixes) and name not in allowed
    )
    if unknown:
        raise RuntimeError(f"unknown paper method environment override(s): {unknown}")


def configure_target_environment(strategy: str, dataset: str, run_id: str) -> None:
    """Resolve target identity before standalone admission reads live policy."""
    from core.strategy_registry import paper_environment_defaults

    spec = get_strategy(strategy)
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
    policy.update(structured_query_identity(strategy))
    from core.paper_compatibility import method_identity
    policy.update(method_identity(strategy))
    return policy


def native_dataset_name(strategy: str, dataset: str) -> str:
    aliases = dict(get_strategy(strategy).dataset_aliases)
    return aliases.get(dataset, dataset)


def approved_youtu_schema(dataset: str) -> dict[str, str]:
    native = native_dataset_name("youtu_graphrag", dataset)
    schemas = runtime_requirement("youtu_graphrag").get("approved_schemas")
    if not isinstance(schemas, dict) or not isinstance(schemas.get(native), dict):
        raise TypeError(f"Youtu has no approved schema for dataset alias {dataset!r} -> {native!r}")
    schema = schemas[native]
    path = str(schema.get("path") or "")
    digest = str(schema.get("sha256") or "").lower()
    if not path or len(digest) != 64 or any(char not in string.hexdigits.lower() for char in digest):
        raise RuntimeError(f"Youtu approved schema manifest is malformed for {native}")
    return {"native_dataset": native, "path": path, "sha256": digest}


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
    from core.paper_compatibility import method_identity
    policy.update(method_identity(strategy))
    policy.update(dict(spec.paper_index_policy))
    policy.update(structured_query_identity(strategy))
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
    if strategy == "youtu_graphrag":
        from core.native_structured_profile import youtu_profile_sha256
        policy["extraction_schema_sha256"] = youtu_profile_sha256()
        schema = approved_youtu_schema(dataset)
        policy.update(
            {
                "dataset_alias": schema["native_dataset"],
                "no_chunk": schema["native_dataset"] in {"hotpot", "musique"},
                "schema_sha256": schema["sha256"],
                "schema_expected_sha256": schema["sha256"],
            }
        )
    return policy


def canonical_operational_policy(strategy: str) -> dict[str, Any]:
    """Resolve the one non-secret effective transport contract."""
    from core.inference_transport import InferenceTransport
    from core.runtime_requirements import runtime_identity

    return {**InferenceTransport.resolve(strategy).policy_dict(), **runtime_identity(strategy)}


def validate_canonical_index_policy(
    strategy: str,
    dataset: str,
    policy: dict[str, Any] | None,
    recorded_sha256: str | None = None,
) -> str:
    """Validate the complete semantic key set and its content hash."""
    raw_policy = dict(policy or {})
    observed = semantic_index_policy(raw_policy)
    observed_digest = semantic_config_sha256(policy)
    if recorded_sha256 is not None and recorded_sha256 != observed_digest:
        raise RuntimeError("stored index policy digest does not match its policy content")
    if strategy == "youtu_graphrag":
        schema_path = str(raw_policy.get("schema_path", "") or "")
        if schema_path:
            approved_parts = Path(approved_youtu_schema(dataset)["path"]).parts
            observed_parts = Path(schema_path).parts
            if len(observed_parts) < len(approved_parts) or observed_parts[-len(approved_parts) :] != approved_parts:
                raise RuntimeError("Youtu canonical policy has an invalid runtime schema_path")
    expected = canonical_semantic_index_policy(strategy, dataset)
    expected["operational_config"] = canonical_operational_policy(strategy)
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(key for key in set(expected) & set(observed) if expected[key] != observed[key])
        raise RuntimeError(
            "stored index policy differs from the checked-in paper policy "
            f"(missing={missing}, extra={extra}, changed={changed})"
        )
    expected_digest = semantic_config_sha256(expected)
    if expected_digest != observed_digest:
        raise RuntimeError("stored index policy digest differs from the canonical complete policy")
    return expected_digest
