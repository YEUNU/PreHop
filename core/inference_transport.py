"""Effective OpenAI-compatible inference settings shared by every runtime."""
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from core.execution_profile import execution_profile, resolved_execution_environment
from core.semantic_config import parse_strict_bool
from core.strategy_registry import PAPER_TRANSPORT, get_strategy

_FORBIDDEN_AMBIENT_PROVIDER_KEYS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "OPENAI_PROVIDER",
    "VLLM_API_BASE",
    "VLLM_EMBED_API_BASE",
    "VLLM_URL",
    "VLLM_EMBED_URL",
    "VLLM_API_KEY",
    "VLLM_SERVED_MODEL_NAME",
    "VLLM_SERVED_EMBED_MODEL_NAME",
)

PUBLIC_INFERENCE_FIELDS = (
    'RAG_INFERENCE_BASE_URL', 'RAG_INFERENCE_API_KEY', 'RAG_GENERATION_MODEL', 'RAG_EMBEDDING_MODEL',
)
_EMBEDDING_CONTROLS = {
    'BATCH_SIZE': 'RAG_EMBEDDING_BATCH_SIZE',
    'CONCURRENCY': 'RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS',
    'RETRY_ATTEMPTS': 'RAG_INFERENCE_RETRY_ATTEMPTS',
}


def inference_environment_keys() -> set[str]:
    """Inputs that must survive filtering into an isolated paper process."""
    from core.strategy_registry import paper_environment_defaults

    return set(PUBLIC_INFERENCE_FIELDS) | set(paper_environment_defaults()) | {
        'RAG_GENERATION_REVISION', 'RAG_EMBEDDING_REVISION', 'RAG_LLM_SEED',
        'EMBEDDING_QUERY_INSTRUCTION', 'MAX_EMBEDDING_LENGTH', 'NEO4J_VECTOR_DIMENSIONS',
        'RAG_EMBEDDING_TOKEN_RESERVE', 'RAG_MAX_CONTEXT_LENGTH',
    }


def preserve_provider_environment(environment=None) -> None:
    """Freeze the already-canonical paper environment before launching a target."""
    environment = os.environ if environment is None else environment
    for name in _FORBIDDEN_AMBIENT_PROVIDER_KEYS:
        environment.setdefault(name, '')
    # Public LiteLLM configuration prevents DEV-mode load_dotenv from reading
    # the installed package's ancestor .env, outside the execution worktree.
    environment['LITELLM_MODE'] = 'PRODUCTION'


def validate_public_inference_environment(environment: Mapping[str, str]) -> None:
    """Check the public inputs required by shell service/experiment launchers."""
    for name in _FORBIDDEN_AMBIENT_PROVIDER_KEYS:
        if environment.get(name, '').strip():
            raise ValueError(f"{name} is not a public input; configure only the canonical RAG_* LiteLLM contract.")
    missing = [name for name in PUBLIC_INFERENCE_FIELDS if not environment.get(name, '').strip()]
    if missing:
        raise ValueError(f"Canonical LiteLLM settings are required: {', '.join(missing)}")


def _normalized_endpoint(value: str) -> str:
    parsed = urlsplit(value.strip())
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))


@dataclass(frozen=True)
class InferenceTransport:
    strategy: str
    generation_model: str
    generation_base_url: str
    embedding_model: str
    embedding_base_url: str
    api_key: str
    timeout_seconds: float | None
    retry_attempts: int
    embedding_batch_size: int
    embedding_concurrency: int
    generation_concurrency: int
    generation_seed: int | None
    embedding_query_instruction: str
    embedding_max_input_tokens: int
    embedding_dimensions: int
    embedding_token_reserve: int
    generation_max_context_tokens: int
    embedding_query_template: str
    gateway_identity_sha256: str
    execution_profile: dict

    def policy_dict(self) -> dict[str, object]:
        """Return the non-secret effective contract persisted with an index."""
        return {
            "gateway_identity_sha256": self.gateway_identity_sha256,
            "execution_profile": self.execution_profile,
            "generation_model": self.generation_model,
            "gateway_embedding_model": self.embedding_model,
            "timeout_seconds": self.timeout_seconds,
            "inference_retry_attempts": self.retry_attempts,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_concurrency": self.embedding_concurrency,
            "generation_concurrency": self.generation_concurrency,
            "generation_seed": self.generation_seed,
            "embedding_query_instruction": self.embedding_query_instruction,
            "embedding_query_template": self.embedding_query_template,
            "embedding_max_input_tokens": self.embedding_max_input_tokens,
            "gateway_embedding_dimensions": self.embedding_dimensions,
            "embedding_token_reserve": self.embedding_token_reserve,
            "generation_max_context_tokens": self.generation_max_context_tokens,
            "transport_profile": "openai_compatible_litellm",
        }

    @classmethod
    def resolve(cls, strategy: str, environment: Mapping[str, str] | None = None) -> InferenceTransport:
        environment = resolved_execution_environment(environment)
        if strategy != "core":
            get_strategy(strategy)

        def embedding_setting(suffix: str) -> int:
            return int(environment[_EMBEDDING_CONTROLS[suffix]])

        timeout = float(environment["RAG_INFERENCE_TIMEOUT"])
        generation_concurrency = int(
            environment["RAG_GENERATION_CONCURRENCY"]
        )
        seed_raw = environment.get("RAG_LLM_SEED", "").strip()
        paper_mode = parse_strict_bool(environment.get("RAG_PAPER_MODE", "false"), name="RAG_PAPER_MODE")
        query_template = PAPER_TRANSPORT.query_template
        query_instruction = environment.get("EMBEDDING_QUERY_INSTRUCTION", PAPER_TRANSPORT.query_instruction).strip()
        embedding_max_input_tokens = int(
            environment.get("MAX_EMBEDDING_LENGTH", str(PAPER_TRANSPORT.embedding_max_input_tokens))
        )
        embedding_dimensions = int(
            environment.get("NEO4J_VECTOR_DIMENSIONS", str(PAPER_TRANSPORT.embedding_dimensions))
        )
        embedding_token_reserve = int(
            environment.get("RAG_EMBEDDING_TOKEN_RESERVE", str(PAPER_TRANSPORT.embedding_token_reserve))
        )
        generation_max_context_tokens = int(
            environment.get("RAG_MAX_CONTEXT_LENGTH", str(PAPER_TRANSPORT.generation_context_tokens))
        )
        endpoint = _normalized_endpoint(environment.get("RAG_INFERENCE_BASE_URL", ""))
        generation_model = environment.get("RAG_GENERATION_MODEL", "").strip()
        embedding_model = environment.get("RAG_EMBEDDING_MODEL", "").strip()
        observed_seed = None if paper_mode else (int(seed_raw) if seed_raw else None)
        result = cls(
            strategy=strategy,
            generation_model=generation_model,
            generation_base_url=endpoint,
            embedding_model=embedding_model,
            embedding_base_url=endpoint,
            api_key=environment.get("RAG_INFERENCE_API_KEY", "").strip(),
            timeout_seconds=timeout,
            retry_attempts=embedding_setting("RETRY_ATTEMPTS"),
            embedding_batch_size=embedding_setting("BATCH_SIZE"),
            embedding_concurrency=embedding_setting("CONCURRENCY"),
            generation_concurrency=generation_concurrency,
            generation_seed=observed_seed,
            embedding_query_instruction=query_instruction,
            embedding_max_input_tokens=embedding_max_input_tokens,
            embedding_dimensions=embedding_dimensions,
            embedding_token_reserve=embedding_token_reserve,
            generation_max_context_tokens=generation_max_context_tokens,
            embedding_query_template=query_template,
            gateway_identity_sha256=hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
            execution_profile=execution_profile(environment),
        )
        return result
