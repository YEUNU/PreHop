import asyncio
import copy
import inspect
import json
import logging
import math
import os
import re
import threading
from typing import Any, ClassVar

import httpx
import openai
import tiktoken
from openai import AsyncOpenAI

from utils.parsers import clean_and_unwrap_json

from .config import RAGConfig
from .inference_telemetry import record as record_inference
from .semantic_config import parse_strict_bool


class VLLMClient:
    _client_cache: ClassVar[dict] = {}
    # Request budgets belong to an inference endpoint, not to a client
    # wrapper.  Sharing these objects prevents two VLLMClient instances on
    # the same event loop from each consuming the full server budget.
    # Embedding calls can originate from several event loops/worker threads
    # (notably HopRAG). A thread semaphore is endpoint-global; an
    # asyncio.Semaphore keyed by loop would multiply the configured cap.
    _embed_semaphores: ClassVar[dict[str, threading.BoundedSemaphore]] = {}
    _embed_semaphores_lock: ClassVar[threading.Lock] = threading.Lock()
    _generation_semaphores: ClassVar[dict[tuple[str, int], asyncio.Semaphore]] = {}
    _generation_inflight: ClassVar[dict[tuple[str, int], int]] = {}
    _generation_peak: ClassVar[dict[tuple[str, int], int]] = {}
    _QUERY_EMBED_CACHE_LIMIT = 2048

    def __init__(self, model_name: str | None = None):
        self.logger = logging.getLogger(__name__)
        # One query-time retrieval call re-embeds the identical query string
        # across several independent channel/scoring calls (hybrid.py's RRF
        # channels, scoring.py's body/bridge passes). Embeddings are a pure
        # function of (text, model), so caching single-text query embeddings
        # here is safe for the process lifetime and avoids redundant network
        # round trips; document batches are never cached since their content
        # differs per call.
        self._query_embed_cache: dict[str, list[float]] = {}
        paper_mode = parse_strict_bool(os.environ.get("RAG_PAPER_MODE", "false"), name="RAG_PAPER_MODE")
        if paper_mode:
            from core.inference_transport import InferenceTransport

            transport = InferenceTransport.resolve("core")
            self.vllm_url = transport.generation_base_url
            self.embed_url = transport.embedding_base_url
            if model_name is not None and model_name != transport.generation_model:
                raise RuntimeError("paper generation model override is not registered in the transport policy")
            self.model_name = transport.generation_model
            self.embed_model_name = transport.embedding_model
            self.api_key = transport.api_key
            timeout_val = transport.timeout_seconds or 0
            self._embedding_concurrency = transport.embedding_concurrency
            self._embedding_batch_size = transport.embedding_batch_size
            self._generation_concurrency = transport.generation_concurrency
            self._retry_attempts = transport.retry_attempts
            self._embedding_query_instruction = transport.embedding_query_instruction
            self._embedding_query_template = transport.embedding_query_template
            self._embedding_max_input_tokens = transport.embedding_max_input_tokens
            self._embedding_dimensions = transport.embedding_dimensions
            self._embedding_token_reserve = transport.embedding_token_reserve
            self._generation_max_context_tokens = transport.generation_max_context_tokens
        else:
            self.vllm_url = RAGConfig.VLLM_URL
            self.embed_url = RAGConfig.VLLM_EMBED_URL
            self.model_name = model_name or RAGConfig.DEFAULT_MODEL
            self.embed_model_name = RAGConfig.EMBEDDING_MODEL
            self.api_key = os.environ.get("RAG_INFERENCE_API_KEY", "EMPTY")
            timeout_val = RAGConfig.LLM_REQUEST_TIMEOUT
            self._embedding_concurrency = RAGConfig.MAX_CONCURRENT_EMBEDDING_REQUESTS
            self._embedding_batch_size = RAGConfig.EMBEDDING_BATCH_SIZE
            self._generation_concurrency = RAGConfig.MAX_CONCURRENT_LLM_CALLS
            self._retry_attempts = RAGConfig.LLM_MAX_RETRIES
            self._embedding_query_instruction = RAGConfig.EMBEDDING_QUERY_INSTRUCTION
            from core.strategy_registry import PAPER_TRANSPORT

            self._embedding_query_template = PAPER_TRANSPORT.query_template
            self._embedding_max_input_tokens = RAGConfig.MAX_EMBEDDING_LENGTH
            self._embedding_dimensions = RAGConfig.EMBEDDING_DIMENSIONS
            self._embedding_token_reserve = int(os.environ.get("RAG_EMBEDDING_TOKEN_RESERVE", "0"))
            self._generation_max_context_tokens = RAGConfig.MAX_CONTEXT_LENGTH
        # 0 = infinite timeout (None)
        self._request_timeout = None if timeout_val == 0 else timeout_val
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except ValueError:
            self.tokenizer = None

    def _count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Rough token count for OpenAI messages."""
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                if key == "content":
                    if isinstance(value, list):
                        for item in value:
                            if item.get("type") == "text":
                                content = item.get("text", "")
                                if self.tokenizer:
                                    num_tokens += len(self.tokenizer.encode(content))
                                else:
                                    num_tokens += len(content) // 4
                            elif item.get("type") == "image_url":
                                num_tokens += 85  # rough estimate for image
                    else:
                        content = str(value)
                        if self.tokenizer:
                            num_tokens += len(self.tokenizer.encode(content))
                        else:
                            num_tokens += len(content) // 4
                if key == "name":
                    num_tokens += 1  # role is always 1 token, name adds 1
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def _truncate_messages(self, messages: list[dict[str, Any]], max_tokens: int | None = None) -> list[dict[str, Any]]:
        """
        Truncates messages to fit within max_tokens.
        Strategy: Keep system message and most recent messages.
        """
        # Reserve tokens for completion (e.g., 1024)
        effective_limit = (max_tokens or self._generation_max_context_tokens) - 1024

        if self._count_tokens(messages) <= effective_limit:
            return copy.deepcopy(messages)

        messages = copy.deepcopy(messages)

        self.logger.warning(
            f"Messages too long ({self._count_tokens(messages)} tokens). Truncating to {effective_limit}..."
        )

        # 1. Keep system message if it exists
        system_msg = None
        if messages and messages[0].get("role") == "system":
            system_msg = messages[0]
            messages = messages[1:]

        # 2. Add messages from the end until limit is reached
        truncated = []
        current_tokens = 0
        if system_msg:
            current_tokens = self._count_tokens([system_msg])

        for msg in reversed(messages):
            msg_tokens = self._count_tokens([msg])
            if current_tokens + msg_tokens <= effective_limit:
                truncated.insert(0, msg)
                current_tokens += msg_tokens
            else:
                # If even one message is too long, we might need to truncate its content
                if not truncated:
                    # For the last message (which is actually the most recent user prompt),
                    # we try to keep as much as possible
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        allowed = effective_limit - current_tokens
                        if allowed > 100:
                            # Preserve the most recent part of the prompt (which
                            # contains the question/instructions in shared RAG
                            # prompts) instead of silently dropping its tail.
                            msg["content"] = "...(truncated)" + content[-(allowed * 3) :]
                            truncated.insert(0, msg)
                break

        if system_msg:
            truncated.insert(0, system_msg)

        return truncated

    def _truncate_text(self, text: str, max_tokens: int = RAGConfig.MAX_EMBEDDING_LENGTH) -> str:
        """Truncates a single string to fit within max_tokens."""
        if not text:
            return ""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return self.tokenizer.decode(tokens[:max_tokens])
        else:
            # Fallback to rough character count
            char_limit = max_tokens * 3
            if len(text) <= char_limit:
                return text
            return text[:char_limit]

    @staticmethod
    def _parse_positive_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float):
            value = int(value)
            return value if value > 0 else None
        if isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                parsed = int(value)
                return parsed if parsed > 0 else None
        return None

    def _resolve_output_token_limit(self, requested_max_tokens: Any = None) -> int:
        cap = max(1, int(RAGConfig.MAX_OUTPUT_TOKENS))
        requested = self._parse_positive_int(requested_max_tokens)
        if requested is None:
            return cap
        return min(requested, cap)

    @staticmethod
    def _json_error_context(text: Any, pos: int | None, radius: int = 120) -> str:
        raw = str(text or "")
        if not raw:
            return ""

        if pos is None or pos < 0:
            excerpt = raw[: max(1, radius * 2)]
            return excerpt.replace("\n", "\\n").replace("\r", "\\r")

        idx = min(max(int(pos), 0), max(0, len(raw) - 1))
        start = max(0, idx - radius)
        end = min(len(raw), idx + radius)
        excerpt = raw[start:end]
        marker = idx - start
        if 0 <= marker < len(excerpt):
            excerpt = excerpt[:marker] + "<<<ERR>>>" + excerpt[marker] + "<<<ERR>>>" + excerpt[marker + 1 :]
        return excerpt.replace("\n", "\\n").replace("\r", "\\r")

    @staticmethod
    def _extract_json_payload(raw: str) -> dict[str, Any] | None:
        """Recover a JSON object from noisy model output."""
        if not isinstance(raw, str):
            return None

        text = raw.strip()
        if not text:
            return None

        candidates: list[str] = [text]
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            candidates.append(fenced.group(1).strip())

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _embedding_token_limit(self, aggressive: bool = False) -> int:
        """
        Return a conservative embedding token limit with safety reserve.
        This avoids off-by-one and tokenizer-mismatch overflows at provider side.
        """
        base = max(256, self._embedding_max_input_tokens - max(0, self._embedding_token_reserve))
        if aggressive:
            return max(128, int(base * 0.75))
        return base

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "maximum context length" in msg or "input tokens" in msg or "too many tokens" in msg

    @classmethod
    def _is_splittable_embedding_error(cls, exc: Exception) -> bool:
        """Return true only for failures that a smaller payload can fix.

        A generic HTTP 400 can mean a bad model, credentials, or schema and
        must not be converted into expensive item-wise retries.  The remote
        stack's malformed MessagePack error, context overflow, and HTTP 413
        are the only currently evidenced size-dependent failures.
        """
        return (
            cls._is_context_length_error(exc)
            or getattr(exc, "status_code", None) == 413
            or "messagepack data is malformed" in str(exc).lower()
        )

    def _validated_embedding_response(self, response: Any, expected_count: int) -> list[list[float]]:
        data = getattr(response, "data", None)
        if not isinstance(data, (list, tuple)):
            raise TypeError("Embedding response has no data list")
        indices = [getattr(item, "index", None) for item in data]
        expected_indices = list(range(expected_count))
        if sorted(indices) != expected_indices:
            raise ValueError(
                "Embedding response indices must be an exact permutation of "
                f"0..{expected_count - 1}: got {indices!r}"
            )
        ordered = sorted(data, key=lambda item: item.index)
        vectors = [getattr(item, "embedding", None) for item in ordered]
        expected_dim = getattr(self, "_embedding_dimensions", RAGConfig.EMBEDDING_DIMENSIONS)
        for index, vector in enumerate(vectors):
            if not isinstance(vector, (list, tuple)) or len(vector) != expected_dim:
                raise ValueError(
                    f"Embedding response vector {index} has invalid dimension: "
                    f"expected {expected_dim}, got {len(vector) if isinstance(vector, (list, tuple)) else None}"
                )
            if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in vector):
                raise ValueError(f"Embedding response vector {index} contains non-finite values")
        return [list(vector) for vector in vectors]

    async def _embed_batch_strict(
        self,
        batch: list[str],
        encoding_type: str,
        *,
        allow_truncation: bool,
    ) -> list[list[float]]:
        try:
            response = await self._create_embedding_request(batch)
            return self._validated_embedding_response(response, len(batch))
        except Exception as exc:
            if not self._is_splittable_embedding_error(exc):
                raise
            if len(batch) > 1:
                midpoint = len(batch) // 2
                self._embedding_bisection_count = getattr(self, "_embedding_bisection_count", 0) + 1
                self.logger.warning(
                    "Embedding payload rejected (size=%d, type=%s); bisecting into %d and %d",
                    len(batch),
                    encoding_type,
                    midpoint,
                    len(batch) - midpoint,
                )
                left = await self._embed_batch_strict(
                    batch[:midpoint], encoding_type, allow_truncation=allow_truncation
                )
                right = await self._embed_batch_strict(
                    batch[midpoint:], encoding_type, allow_truncation=allow_truncation
                )
                return left + right
            if self._is_context_length_error(exc) and allow_truncation:
                shortened = self._truncate_text(
                    batch[0], max_tokens=self._embedding_token_limit(aggressive=True)
                )
                if shortened and shortened != batch[0]:
                    response = await self._create_embedding_request([shortened])
                    return self._validated_embedding_response(response, 1)
            if self._is_context_length_error(exc) and not allow_truncation:
                raise ValueError(
                    "Embedding input exceeds the configured server limit and truncation is forbidden "
                    f"for encoding_type={encoding_type!r}"
                ) from exc
            raise

    @staticmethod
    def _is_retryable_inference_error(exc: Exception) -> bool:
        if isinstance(
            exc,
            (
                httpx.TimeoutException,
                httpx.TransportError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.InternalServerError,
            ),
        ):
            return True
        status = getattr(exc, "status_code", None)
        return isinstance(status, int) and (status == 429 or status >= 500)

    def _is_qwen_embedding_model(self) -> bool:
        return "qwen3-embedding" in (self.embed_model_name or "").lower()

    def _format_query_for_embedding(self, query: str) -> str:
        """Apply model-recommended query instruction format for Qwen embedding models."""
        return self._embedding_query_template.format(
            instruction=self._embedding_query_instruction,
            query=query,
        )

    async def _create_embedding_request(self, inputs: list[str]):
        key = self.embed_url.rstrip("/")
        cls = type(self)
        concurrency = getattr(self, "_embedding_concurrency", RAGConfig.MAX_CONCURRENT_EMBEDDING_REQUESTS)
        with cls._embed_semaphores_lock:
            semaphore = cls._embed_semaphores.setdefault(
                key,
                threading.BoundedSemaphore(concurrency),
            )
        acquire_task = asyncio.create_task(asyncio.to_thread(semaphore.acquire))
        try:
            await asyncio.shield(acquire_task)
        except asyncio.CancelledError:
            # Cancelling to_thread() does not stop the worker thread. Return
            # cancellation immediately, while a callback releases any permit
            # the background acquire obtains later. Waiting here would make a
            # cancelled request hang forever behind a wedged holder.
            def _release_cancelled_acquire(task: asyncio.Task) -> None:
                try:
                    acquired = task.result()
                except asyncio.CancelledError:
                    return
                except Exception:  # noqa: BLE001 - done callback must never leak into the event loop
                    return
                if acquired:
                    semaphore.release()

            acquire_task.add_done_callback(_release_cancelled_acquire)
            raise
        try:
            response = await self._retry_with_backoff(
                self.embed_client.embeddings.create,
                model=self.embed_model_name,
                input=inputs,
            )
            record_inference("embedding", response)
            return response
        finally:
            semaphore.release()

    async def _create_generation_request(self, request_client: AsyncOpenAI, params: dict[str, Any]):
        if parse_strict_bool(os.environ.get("RAG_PAPER_MODE", "false"), name="RAG_PAPER_MODE"):
            from core.inference_transport import InferenceTransport, _normalized_endpoint

            approved = InferenceTransport.resolve("core")
            if params.get("model") != approved.generation_model:
                raise RuntimeError("paper generation request model is not registered in the transport policy")
            if _normalized_endpoint(str(request_client.base_url)) != approved.generation_base_url:
                raise RuntimeError("paper generation request endpoint differs from the approved gateway")
            if params.get("seed", approved.generation_seed) != approved.generation_seed:
                raise RuntimeError("paper generation request seed differs from the transport policy")
            if approved.generation_seed is not None:
                params["seed"] = approved.generation_seed
        endpoint = str(getattr(request_client, "base_url", self.vllm_url)).rstrip("/")
        key = (endpoint, self._running_loop_id())
        cls = type(self)
        semaphore = cls._generation_semaphores.setdefault(
            key,
            asyncio.Semaphore(self._generation_concurrency),
        )
        async with semaphore:
            inflight = cls._generation_inflight.get(key, 0) + 1
            cls._generation_inflight[key] = inflight
            previous_peak = cls._generation_peak.get(key, 0)
            if inflight > previous_peak:
                cls._generation_peak[key] = inflight
                if inflight == 1 or inflight % 10 == 0 or inflight == self._generation_concurrency:
                    self.logger.info(
                        "Generation endpoint load | in_flight=%d peak=%d limit=%d endpoint=%s",
                        inflight,
                        inflight,
                        self._generation_concurrency,
                        endpoint,
                    )
            try:
                response = await self._retry_with_backoff(request_client.chat.completions.create, **params)
                record_inference("generation", response)
                return response
            finally:
                cls._generation_inflight[key] -= 1

    async def _retry_with_backoff(self, coro_func, *args, **kwargs):
        """Exponential backoff retry wrapper for handling GPU load spikes."""
        retry_attempts = getattr(self, "_retry_attempts", RAGConfig.LLM_MAX_RETRIES)
        if retry_attempts < 1:
            raise ValueError("LLM_MAX_RETRIES must be at least 1")
        if RAGConfig.LLM_RETRY_DELAY < 0:
            raise ValueError("LLM_RETRY_DELAY must be non-negative")
        for attempt in range(retry_attempts):
            try:
                return await coro_func(*args, **kwargs)
            except Exception as exc:
                if not self._is_retryable_inference_error(exc) or attempt == retry_attempts - 1:
                    raise
                delay = RAGConfig.LLM_RETRY_DELAY * (2**attempt)
                self.logger.warning(
                    "Transient inference error, retrying in %.1fs (%d/%d): %s",
                    delay,
                    attempt + 1,
                    retry_attempts,
                    exc,
                )
                await asyncio.sleep(delay)

    def _get_cached_client(self, url: str) -> AsyncOpenAI:
        if not str(url or "").strip():
            raise ValueError("External inference endpoint is not configured")
        # Key the cache by the *running event loop* as well as the url. httpx's
        # connection pool binds to the loop that first used it, so a client
        # cached on one loop and reused on another (hoprag runs each judge call
        # in a ThreadPoolExecutor worker that spins up a fresh asyncio.run loop)
        # deadlocks forever in select() — the loop-bound read timeout never
        # fires either. Per-loop clients keep prehop/naive (single main loop)
        # unchanged while isolating hoprag's multi-loop path.
        key = (url, self._running_loop_id())
        if key not in self._client_cache:
            timeout = httpx.Timeout(self._request_timeout, connect=60.0)
            self._client_cache[key] = AsyncOpenAI(base_url=url, api_key=self.api_key, timeout=timeout)
        return self._client_cache[key]

    @staticmethod
    def _running_loop_id() -> int:
        try:
            return id(asyncio.get_running_loop())
        except RuntimeError:
            return 0

    @property
    def client(self):
        return self._get_cached_client(self.vllm_url)

    @property
    def embed_client(self):
        return self._get_cached_client(self.embed_url)

    @property
    def judge_client(self):
        return self.client

    @staticmethod
    def _is_openai_model(model: str) -> bool:
        _ = model
        return False

    def think_strip(self, message: str | None) -> str:
        if not message:
            return ""
        if "</think>" in message:
            message = message.split("</think>")[-1]
        return message.replace("<end>", "").strip()

    async def generate_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> Any:
        try:
            apply_default_sampling = bool(kwargs.pop("apply_default_sampling", True))
            # Truncate messages to fit context window
            truncated_messages = self._truncate_messages(messages)
            requested_max_tokens = kwargs.get("max_tokens", kwargs.get("max_completion_tokens"))

            params = {
                "model": kwargs.get("model", self.model_name),
                "messages": truncated_messages,
                "stream": False,
                "max_tokens": self._resolve_output_token_limit(requested_max_tokens),
            }
            if temperature is not None:
                params["temperature"] = temperature
            elif apply_default_sampling:
                params["temperature"] = 0.7
            if RAGConfig.LLM_SEED is not None:
                params["seed"] = RAGConfig.LLM_SEED
            if tools:
                params["tools"] = tools
            if tool_choice:
                params["tool_choice"] = tool_choice
            if kwargs.get("response_format"):
                params["response_format"] = kwargs["response_format"]
            params["extra_body"] = (
                kwargs["extra_body"]
                if "extra_body" in kwargs and kwargs["extra_body"] is not None
                else {"chat_template_kwargs": {"enable_thinking": False}}
            )

            is_openai = self._is_openai_model(str(params["model"]))
            if is_openai:
                params.pop("extra_body", None)
            request_client = self.judge_client if is_openai else self.client
            response = await self._create_generation_request(request_client, params)
            strict_schema = (kwargs.get("response_format") or {}).get("type") == "json_schema"
            if strict_schema:
                from core.structured_outputs import StructuredOutputError

                if len(response.choices) != 1 or response.choices[0].finish_reason != "stop":
                    raise StructuredOutputError("structured response did not finish with one complete answer")
                message = response.choices[0].message
                if getattr(message, "refusal", None) or getattr(message, "tool_calls", None):
                    raise StructuredOutputError("structured response refused or returned tool output")
                if not isinstance(message.content, str) or not message.content.strip():
                    raise StructuredOutputError("structured response has no raw content")
                return message.content
            msg = response.choices[0].message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return msg

            content = msg.content or (msg.reasoning_content if hasattr(msg, "reasoning_content") else "")

            # If JSON format was requested, don't attempt to unwrap common keys
            if kwargs.get("response_format"):
                return self.think_strip(content)

            return self.think_strip(clean_and_unwrap_json(content))
        except Exception as e:
            self.logger.error(f"Error calling vLLM: {e}")
            raise

    async def generate_json(
        self, messages: list[dict[str, str]], max_retries: int | None = None, **kwargs
    ) -> dict[str, Any]:
        contract = kwargs.pop("structured_contract", None)
        if contract is not None:
            from core.structured_outputs import StructuredContract, StructuredOutputError

            if not isinstance(contract, StructuredContract):
                raise TypeError("structured_contract must be a registered contract")
            if any(name in kwargs for name in ("response_format", "extra_body", "structured_outputs", "tools", "tool_choice")) or any(name.startswith("guided_") for name in kwargs):
                raise StructuredOutputError("conflicting structured-output request configuration")
            kwargs.pop("json_debug_label", None)
            from core.inference_telemetry import record_structured_contract
            record_structured_contract(contract.provenance())
            raw = await self.generate_response(messages, response_format=contract.response_format(), **kwargs)
            def object_pairs(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("duplicate JSON property")
                    result[key] = value
                return result
            def invalid_constant(value):
                raise ValueError(f"non-finite JSON constant: {value}")
            try:
                parsed = json.loads(raw, object_pairs_hook=object_pairs, parse_constant=invalid_constant)
            except (ValueError, TypeError) as exc:
                raise StructuredOutputError("structured response is not one raw JSON document") from exc
            validated = contract.validate(parsed)
            self.logger.debug("Structured output validated | %s", contract.provenance())
            return validated
        last_error_hint = ""
        last_parse_error: Exception | None = None
        last_response_preview = ""
        max_retries = max_retries or RAGConfig.RETRY_COUNT
        json_debug_label = str(kwargs.pop("json_debug_label", "") or "").strip()

        def _preview(text: Any, max_len: int = 500) -> str:
            raw = str(text or "")
            if not raw:
                return ""
            raw = raw.replace("\\n", "\\\\n").replace("\\r", "\\\\r")
            return raw if len(raw) <= max_len else raw[:max_len] + "..."

        for attempt in range(max_retries):
            current_messages = copy.deepcopy(messages)
            if last_error_hint:
                current_messages.append({"role": "user", "content": f"SYSTEM: {last_error_hint}"})
            response_text = ""
            try:
                model = kwargs.get("model")
                if model and model == RAGConfig.EVAL_MODEL:
                    parsed = await self.generate_eval_json(current_messages, model=model)
                    if parsed:
                        return parsed
                    last_error_hint = "Output ONLY one non-empty JSON object."
                    last_parse_error = ValueError("evaluation model returned an empty JSON object")
                    last_response_preview = "empty-response"
                    continue

                response_text = await self.generate_response(
                    current_messages, response_format={"type": "json_object"}, **kwargs
                )
                last_response_preview = _preview(response_text)
                parsed = None
                try:
                    parsed = json.loads(response_text)
                except json.JSONDecodeError:
                    parsed = self._extract_json_payload(response_text)
                    if parsed is None:
                        raise
                    self.logger.info(
                        "generate_json fallback parse succeeded [stage=%s] (attempt %d/%d)",
                        json_debug_label or "unknown",
                        attempt + 1,
                        max_retries,
                    )

                if isinstance(parsed, dict):
                    return parsed

                last_parse_error = TypeError(f"expected JSON object, got {type(parsed).__name__}")
                last_error_hint = (
                    f"Invalid JSON type '{type(parsed).__name__}'. "
                    "Output ONLY one JSON object (not array/string/markdown)."
                )
                last_response_preview = _preview(response_text, max_len=800)
                self.logger.warning(
                    "generate_json type mismatch [stage=%s] (attempt %d/%d): %s | preview=%s",
                    json_debug_label or "unknown",
                    attempt + 1,
                    max_retries,
                    type(parsed).__name__,
                    _preview(response_text, max_len=160),
                )
            except json.JSONDecodeError as e:
                last_parse_error = e
                snippet = self._json_error_context(response_text, e.pos)
                last_response_preview = _preview(response_text, max_len=800)
                self.logger.warning(
                    "generate_json parse failed [stage=%s] (attempt %d/%d): %s | len=%d pos=%d line=%d col=%d | snippet=%s",
                    json_debug_label or "unknown",
                    attempt + 1,
                    max_retries,
                    e,
                    len(response_text or ""),
                    e.pos,
                    e.lineno,
                    e.colno,
                    snippet,
                )
                last_error_hint = (
                    f"Invalid JSON near line {e.lineno}, column {e.colno}. "
                    "Output ONLY one raw JSON object with double quotes, no markdown fences, no prose."
                )
        stage = json_debug_label or "unknown"
        raise ValueError(
            f"generate_json exhausted {max_retries} parse attempts for stage={stage}: "
            f"{last_error_hint or 'no further hint'} | last_response={last_response_preview or 'n/a'}"
        ) from last_parse_error

    async def generate_eval_json(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Generate judge JSON synchronously when Batch mode is explicitly disabled."""
        model = kwargs.get("model", RAGConfig.EVAL_MODEL)
        try:
            truncated_messages = self._truncate_messages(messages)
            requested_max_tokens = kwargs.get("max_tokens", kwargs.get("max_completion_tokens"))
            params: dict[str, Any] = {
                "model": model,
                "messages": truncated_messages,
                "response_format": {"type": "json_object"},
                "max_tokens": self._resolve_output_token_limit(requested_max_tokens),
                "temperature": 0.0,
            }
            if not self._is_openai_model(str(model)):
                params["extra_body"] = kwargs.get("extra_body") or {"chat_template_kwargs": {"enable_thinking": False}}
            if RAGConfig.LLM_SEED is not None:
                params["seed"] = RAGConfig.LLM_SEED
            response = await self._create_generation_request(self.judge_client, params)
            content = response.choices[0].message.content or ""
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as e:
                snippet = self._json_error_context(content, e.pos)
                self.logger.warning(
                    "generate_eval_json parse failed: %s | len=%d pos=%d line=%d col=%d | snippet=%s",
                    e,
                    len(content),
                    e.pos,
                    e.lineno,
                    e.colno,
                    snippet,
                )
                raise
            if not isinstance(parsed, dict):
                raise TypeError(f"Evaluation model returned {type(parsed).__name__}, expected JSON object")
            return parsed
        except Exception as e:
            self.logger.error(f"Error calling evaluation LLM ({model}): {e}")
            raise

    async def get_embeddings(
        self,
        texts: list[str],
        encoding_type: str = "document",
        *,
        allow_truncation: bool = True,
    ) -> list[list[float]]:
        if not texts:
            return []

        single_query = encoding_type == "query" and len(texts) == 1
        if single_query:
            cached = self._query_embed_cache.get(texts[0])
            if cached is not None:
                return [cached]

        # Most call sites retain bounded truncation for defensive compatibility.
        # A whole-document index can forbid it so one source is never silently
        # represented by only a prefix.
        embed_max_tokens = self._embedding_token_limit()
        truncated_texts: list[str] = []
        for t in texts:
            candidate = self._truncate_text(t, max_tokens=embed_max_tokens) if allow_truncation else t
            if encoding_type == "query" and self._is_qwen_embedding_model():
                candidate = self._format_query_for_embedding(candidate)
            truncated_texts.append(
                self._truncate_text(candidate, max_tokens=embed_max_tokens) if allow_truncation else candidate
            )

        if self._embedding_batch_size < 1:
            raise ValueError("RAG_EMBEDDING_BATCH_SIZE must be at least 1")
        all_embeddings: list[list[float]] = []
        for i in range(0, len(truncated_texts), self._embedding_batch_size):
            batch = truncated_texts[i : i + self._embedding_batch_size]
            all_embeddings.extend(
                await self._embed_batch_strict(
                    batch,
                    encoding_type,
                    allow_truncation=allow_truncation,
                )
            )

        if len(all_embeddings) != len(texts):
            raise ValueError(
                f"Embedding result count mismatch: expected {len(texts)}, got {len(all_embeddings)}"
            )

        if single_query and all_embeddings and all_embeddings[0]:
            if len(self._query_embed_cache) >= self._QUERY_EMBED_CACHE_LIMIT:
                self._query_embed_cache.clear()
            self._query_embed_cache[texts[0]] = all_embeddings[0]

        return all_embeddings

    async def get_embedding(self, text: str) -> list[float]:
        res = await self.get_embeddings([text], encoding_type="query")
        if not res or not res[0]:
            self.logger.warning("Failed to generate query embedding (empty vector).")
            return []
        return res[0]

    @classmethod
    async def global_close(cls):
        """Close cached API clients and clear cache."""
        logger = logging.getLogger(__name__)
        for key, client in list(cls._client_cache.items()):
            try:
                close_fn = getattr(client, "close", None)
                if callable(close_fn):
                    result = close_fn()
                    if inspect.isawaitable(result):
                        await result
                else:
                    aclose_fn = getattr(client, "aclose", None)
                    if callable(aclose_fn):
                        result = aclose_fn()
                        if inspect.isawaitable(result):
                            await result
            except Exception as e:  # noqa: BLE001 - cleanup must continue through heterogeneous client types
                logger.warning(f"Failed to close client cache entry '{key}': {e}")
        cls._client_cache.clear()
        cls._embed_semaphores.clear()
        cls._generation_semaphores.clear()
        cls._generation_inflight.clear()
        cls._generation_peak.clear()


def get_llm_client(model_id: str = "default"):
    return VLLMClient(model_name=None if model_id == "default" else model_id)
