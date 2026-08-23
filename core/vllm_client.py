import asyncio
import copy
import inspect
import json
import logging
import os
from typing import Any, ClassVar

import httpx
import openai
import tiktoken
from openai import AsyncOpenAI

from utils.parsers import clean_and_unwrap_json

from .config import RAGConfig


class VLLMClient:
    _client_cache: ClassVar[dict] = {}
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
        self.vllm_url = RAGConfig.VLLM_URL
        self.embed_url = RAGConfig.VLLM_EMBED_URL

        self.model_name = model_name or RAGConfig.DEFAULT_MODEL
        self.embed_model_name = RAGConfig.EMBEDDING_MODEL

        self.api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
        # 0 = infinite timeout (None)
        timeout_val = RAGConfig.LLM_REQUEST_TIMEOUT
        self._request_timeout = None if timeout_val == 0 else timeout_val
        # HopRAG's synchronous upstream hooks may call this client from fresh
        # event loops in worker threads. asyncio synchronization primitives
        # are loop-bound after first contention, so keep one semaphore per
        # running loop instead of sharing a single cross-loop object.
        self._embed_semaphores: dict[int, asyncio.Semaphore] = {}
        self._generation_semaphores: dict[int, asyncio.Semaphore] = {}

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

    def _truncate_messages(
        self, messages: list[dict[str, Any]], max_tokens: int = RAGConfig.MAX_CONTEXT_LENGTH
    ) -> list[dict[str, Any]]:
        """
        Truncates messages to fit within max_tokens.
        Strategy: Keep system message and most recent messages.
        """
        # Reserve tokens for completion (e.g., 1024)
        effective_limit = max_tokens - 1024

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

    def _embedding_token_limit(self, aggressive: bool = False) -> int:
        """
        Return a conservative embedding token limit with safety reserve.
        This avoids off-by-one and tokenizer-mismatch overflows at provider side.
        """
        reserve = int(os.environ.get("RAG_EMBEDDING_TOKEN_RESERVE", "0"))
        base = max(256, RAGConfig.MAX_EMBEDDING_LENGTH - max(0, reserve))
        if aggressive:
            return max(128, int(base * 0.75))
        return base

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "maximum context length" in msg or "input tokens" in msg or "too many tokens" in msg

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
        task = os.environ.get(
            "EMBEDDING_QUERY_INSTRUCTION",
            "Given a web search query, retrieve relevant passages that answer the query",
        )
        return f"Instruct: {task}\nQuery:{query}"

    async def _create_embedding_request(self, inputs: list[str]):
        loop_id = self._running_loop_id()
        semaphore = self._embed_semaphores.setdefault(
            loop_id,
            asyncio.Semaphore(RAGConfig.MAX_CONCURRENT_EMBEDDING_REQUESTS),
        )
        async with semaphore:
            return await self._retry_with_backoff(
                self.embed_client.embeddings.create,
                model=self.embed_model_name,
                input=inputs,
            )

    async def _create_generation_request(self, request_client: AsyncOpenAI, params: dict[str, Any]):
        loop_id = self._running_loop_id()
        semaphore = self._generation_semaphores.setdefault(
            loop_id,
            asyncio.Semaphore(RAGConfig.MAX_CONCURRENT_LLM_CALLS),
        )
        async with semaphore:
            return await self._retry_with_backoff(request_client.chat.completions.create, **params)

    async def _embed_batch_itemwise(self, batch: list[str], encoding_type: str) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for idx, text in enumerate(batch):
            try:
                response = await self._create_embedding_request([text])
                if getattr(response, "data", None):
                    embeddings.append(response.data[0].embedding)
                else:
                    self.logger.error("Embedding item response missing data at idx=%d.", idx)
                    embeddings.append([])
            except Exception as e:  # noqa: BLE001 - OpenAI clients expose provider-specific exceptions
                recovered = False
                if self._is_context_length_error(e):
                    aggressive_text = self._truncate_text(text, max_tokens=self._embedding_token_limit(aggressive=True))
                    if aggressive_text and aggressive_text != text:
                        try:
                            response = await self._create_embedding_request([aggressive_text])
                            if getattr(response, "data", None):
                                self.logger.warning(
                                    "Embedding item recovered with aggressive truncation at idx=%d.",
                                    idx,
                                )
                                embeddings.append(response.data[0].embedding)
                                recovered = True
                        except Exception as e2:  # noqa: BLE001 - same provider boundary as the first request
                            self.logger.error(
                                "Embedding item aggressive retry failed at idx=%d: %s",
                                idx,
                                e2,
                            )
                if not recovered:
                    preview = text.replace("\n", " ")[:120]
                    self.logger.error("Embedding item failed at idx=%d text='%s': %s", idx, preview, e)
                    embeddings.append([])
        return embeddings

    async def _retry_with_backoff(self, coro_func, *args, **kwargs):
        """Exponential backoff retry wrapper for handling GPU load spikes."""
        if RAGConfig.LLM_MAX_RETRIES < 1:
            raise ValueError("LLM_MAX_RETRIES must be at least 1")
        if RAGConfig.LLM_RETRY_DELAY < 0:
            raise ValueError("LLM_RETRY_DELAY must be non-negative")
        for attempt in range(RAGConfig.LLM_MAX_RETRIES):
            try:
                return await coro_func(*args, **kwargs)
            except Exception as exc:
                if not self._is_retryable_inference_error(exc) or attempt == RAGConfig.LLM_MAX_RETRIES - 1:
                    raise
                delay = RAGConfig.LLM_RETRY_DELAY * (2**attempt)
                self.logger.warning(
                    "Transient inference error, retrying in %.1fs (%d/%d): %s",
                    delay,
                    attempt + 1,
                    RAGConfig.LLM_MAX_RETRIES,
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
        if self._is_openai_model(RAGConfig.EVAL_MODEL):
            if not RAGConfig.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is required for an OpenAI EVAL_MODEL")
            if "openai_official" not in self._client_cache:
                self._client_cache["openai_official"] = AsyncOpenAI(api_key=RAGConfig.OPENAI_API_KEY)
            return self._client_cache["openai_official"]
        return self.client

    @staticmethod
    def _is_openai_model(model: str) -> bool:
        return str(model or "").lower().startswith(("gpt-", "o1", "o3", "o4", "chatgpt-"))

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
        last_error_hint = ""
        last_parse_error: Exception | None = None
        max_retries = max_retries or RAGConfig.RETRY_COUNT
        json_debug_label = str(kwargs.pop("json_debug_label", "") or "").strip()
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
                    continue

                response_text = await self.generate_response(
                    current_messages, response_format={"type": "json_object"}, **kwargs
                )
                parsed = json.loads(response_text)
                if isinstance(parsed, dict):
                    return parsed
                last_parse_error = TypeError(f"expected JSON object, got {type(parsed).__name__}")
                last_error_hint = (
                    f"Invalid JSON type '{type(parsed).__name__}'. "
                    "Output ONLY one JSON object (not array/string/markdown)."
                )
            except json.JSONDecodeError as e:
                last_parse_error = e
                snippet = self._json_error_context(response_text, e.pos)
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
            f"generate_json exhausted {max_retries} parse attempts for stage={stage}"
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

    async def get_embeddings(self, texts: list[str], encoding_type: str = "document") -> list[list[float]]:
        if not texts:
            return []

        single_query = encoding_type == "query" and len(texts) == 1
        if single_query:
            cached = self._query_embed_cache.get(texts[0])
            if cached is not None:
                return [cached]

        # Truncate and format query texts to prevent embedding model overflow.
        embed_max_tokens = self._embedding_token_limit()
        truncated_texts: list[str] = []
        for t in texts:
            candidate = self._truncate_text(t, max_tokens=embed_max_tokens)
            if encoding_type == "query" and self._is_qwen_embedding_model():
                candidate = self._format_query_for_embedding(candidate)
            truncated_texts.append(self._truncate_text(candidate, max_tokens=embed_max_tokens))

        if RAGConfig.EMBEDDING_BATCH_SIZE < 1:
            raise ValueError("RAG_EMBEDDING_BATCH_SIZE must be at least 1")
        all_embeddings = []
        for i in range(0, len(truncated_texts), RAGConfig.EMBEDDING_BATCH_SIZE):
            batch = truncated_texts[i : i + RAGConfig.EMBEDDING_BATCH_SIZE]
            try:
                response = await self._create_embedding_request(batch)
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as e:
                status = getattr(e, "status_code", None)
                can_split_batch = self._is_context_length_error(e) or status in {400, 413}
                if not can_split_batch:
                    raise
                self.logger.error(
                    "Embedding batch rejected (size=%d, type=%s): %s. Retrying item-wise to isolate the bad input.",
                    len(batch),
                    encoding_type,
                    e,
                )
                all_embeddings.extend(await self._embed_batch_itemwise(batch, encoding_type=encoding_type))

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


def get_llm_client(model_id: str = "default"):
    return VLLMClient(model_name=None if model_id == "default" else model_id)
