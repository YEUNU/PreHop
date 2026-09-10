"""Single fail-closed OpenAI-compatible inference transport contract."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from core.embedding_policy import EmbeddingOperationalConfig
from core.execution_profile import execution_profile
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


def preserve_provider_environment(environment=None) -> None:
    """Freeze the already-canonical paper environment before native imports."""
    environment = os.environ if environment is None else environment
    sorted(name for name in _FORBIDDEN_AMBIENT_PROVIDER_KEYS if environment.get(name))
    for name in _FORBIDDEN_AMBIENT_PROVIDER_KEYS:
        environment.setdefault(name, '')
    # Public LiteLLM configuration prevents DEV-mode load_dotenv from reading
    # the installed package's ancestor .env, outside the execution worktree.
    environment['LITELLM_MODE'] = 'PRODUCTION'


def _required(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


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

    def policy_dict(self) -> dict[str, object]:
        """Return the non-secret effective contract persisted with an index."""
        return {
            "gateway_identity_sha256": self.gateway_identity_sha256,
            "execution_profile": execution_profile(),
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
            "transport_profile": (
                "openai_compatible_litellm"
                if self.strategy == "core"
                else get_strategy(self.strategy).transport_profile
            ),
        }

    @classmethod
    def resolve(cls, strategy: str) -> InferenceTransport:
        embedding = EmbeddingOperationalConfig.resolve(strategy)
        timeout = float(os.environ.get("RAG_INFERENCE_TIMEOUT", str(PAPER_TRANSPORT.timeout_seconds)))
        generation_concurrency = int(
            os.environ.get("RAG_GENERATION_CONCURRENCY", str(PAPER_TRANSPORT.generation_concurrency))
        )
        seed_raw = os.environ.get("RAG_LLM_SEED", "").strip()
        paper_mode = parse_strict_bool(os.environ.get("RAG_PAPER_MODE", "false"), name="RAG_PAPER_MODE")
        method_policy = {} if strategy == "core" else dict(get_strategy(strategy).paper_index_policy)
        expected_instruction = method_policy.get("embedding_query_instruction", PAPER_TRANSPORT.query_instruction)
        query_template = method_policy.get("embedding_query_template", PAPER_TRANSPORT.query_template)
        query_instruction = os.environ.get("EMBEDDING_QUERY_INSTRUCTION", expected_instruction).strip()
        embedding_max_input_tokens = int(
            os.environ.get("MAX_EMBEDDING_LENGTH", str(PAPER_TRANSPORT.embedding_max_input_tokens))
        )
        embedding_dimensions = int(
            os.environ.get("NEO4J_VECTOR_DIMENSIONS", str(PAPER_TRANSPORT.embedding_dimensions))
        )
        embedding_token_reserve = int(
            os.environ.get("RAG_EMBEDDING_TOKEN_RESERVE", str(PAPER_TRANSPORT.embedding_token_reserve))
        )
        generation_max_context_tokens = int(
            os.environ.get("RAG_MAX_CONTEXT_LENGTH", str(PAPER_TRANSPORT.generation_context_tokens))
        )
        paper_mode = parse_strict_bool(os.environ.get("RAG_PAPER_MODE", "false"), name="RAG_PAPER_MODE")
        endpoint = _normalized_endpoint(_required("RAG_INFERENCE_BASE_URL"))
        generation_model = _required("RAG_GENERATION_MODEL")
        embedding_model = _required("RAG_EMBEDDING_MODEL")
        observed_seed = None if paper_mode else (int(seed_raw) if seed_raw else None)
        proxy = os.environ.get("RAG_QUEUE_PROXY_URL", "").strip()
        if proxy:
            proxy = _normalized_endpoint(proxy)
        result = cls(
            strategy=strategy,
            generation_model=generation_model,
            generation_base_url=proxy or endpoint,
            embedding_model=embedding_model,
            embedding_base_url=proxy or endpoint,
            api_key=os.environ["RAG_QUEUE_TOKEN"] if proxy else _required("RAG_INFERENCE_API_KEY"),
            timeout_seconds=timeout,
            retry_attempts=embedding.retry_attempts,
            embedding_batch_size=embedding.batch_size,
            embedding_concurrency=embedding.concurrency,
            generation_concurrency=generation_concurrency,
            generation_seed=observed_seed,
            embedding_query_instruction=query_instruction,
            embedding_max_input_tokens=embedding_max_input_tokens,
            embedding_dimensions=embedding_dimensions,
            embedding_token_reserve=embedding_token_reserve,
            generation_max_context_tokens=generation_max_context_tokens,
            embedding_query_template=query_template,
            gateway_identity_sha256=hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
        )
        return result
