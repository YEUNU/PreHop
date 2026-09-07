import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.config import RAGConfig
from core.vllm_client import VLLMClient


def _response(indices, vectors):
    return SimpleNamespace(
        data=[SimpleNamespace(index=index, embedding=vector) for index, vector in zip(indices, vectors, strict=True)]
    )


def _client(monkeypatch, dimension=2):
    client = object.__new__(VLLMClient)
    client.logger = __import__("logging").getLogger("test")
    monkeypatch.setattr(RAGConfig, "EMBEDDING_DIMENSIONS", dimension)
    return client


@pytest.mark.parametrize("indices", [[0, 0], [0, 2], [1]])
def test_embedding_response_requires_exact_indices(monkeypatch, indices):
    client = _client(monkeypatch)
    with pytest.raises(ValueError, match="exact permutation"):
        client._validated_embedding_response(_response(indices, [[1.0, 0.0]] * len(indices)), 2)


@pytest.mark.parametrize("vector", [[1.0], [1.0, float("nan")], [1.0, float("inf")]])
def test_embedding_response_requires_dimension_and_finite_values(monkeypatch, vector):
    client = _client(monkeypatch)
    with pytest.raises(ValueError, match="dimension|non-finite"):
        client._validated_embedding_response(_response([0], [vector]), 1)


def test_messagepack_failure_bisects_and_preserves_order(monkeypatch):
    client = _client(monkeypatch)
    seen = []

    async def request(values):
        seen.append(list(values))
        if len(values) > 1:
            raise RuntimeError("MessagePack data is malformed: trailing characters")
        value = float(ord(values[0]) - ord("a") + 1)
        return _response([0], [[value, 0.0]])

    client._create_embedding_request = request
    result = asyncio.run(client._embed_batch_strict(["a", "b", "c"], "document", allow_truncation=True))

    assert result == [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    assert seen == [["a", "b", "c"], ["a"], ["b", "c"], ["b"], ["c"]]


def test_unrelated_http_400_is_not_split(monkeypatch):
    client = _client(monkeypatch)
    error = RuntimeError("unknown embedding model")
    error.status_code = 400
    client._create_embedding_request = AsyncMock(side_effect=error)

    with pytest.raises(RuntimeError, match="unknown embedding model"):
        asyncio.run(client._embed_batch_strict(["a", "b"], "document", allow_truncation=True))
    client._create_embedding_request.assert_awaited_once()


def test_core_embedding_endpoint_peak_is_one_across_clients(monkeypatch):
    in_flight = 0
    peak = 0

    class Embeddings:
        async def create(self, **kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return kwargs

    class Client(VLLMClient):
        @property
        def embed_client(self):
            return SimpleNamespace(embeddings=Embeddings())

    monkeypatch.setattr(RAGConfig, "MAX_CONCURRENT_EMBEDDING_REQUESTS", 1)
    Client._embed_semaphores.clear()
    clients = [object.__new__(Client), object.__new__(Client)]
    for client in clients:
        client.embed_url = "http://embedding.example/v1"
        client.embed_model_name = "embedding-model"

    async def run():
        await asyncio.gather(*(client._create_embedding_request(["x"]) for client in clients))

    asyncio.run(run())
    assert peak == 1


def test_cancelled_embedding_waiter_returns_eventual_permit(monkeypatch):
    class Client(VLLMClient):
        @property
        def embed_client(self):
            raise AssertionError("cancelled waiter must not issue a request")

    monkeypatch.setattr(RAGConfig, "MAX_CONCURRENT_EMBEDDING_REQUESTS", 1)
    Client._embed_semaphores.clear()
    semaphore = __import__("threading").BoundedSemaphore(1)
    semaphore.acquire()
    Client._embed_semaphores["http://embedding.example/v1"] = semaphore
    client = object.__new__(Client)
    client.embed_url = "http://embedding.example/v1"
    client.embed_model_name = "embedding-model"

    async def run():
        task = asyncio.create_task(client._create_embedding_request(["x"]))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.done()
        semaphore.release()
        await asyncio.sleep(0.01)

    asyncio.run(run())
    assert semaphore.acquire(blocking=False)
    semaphore.release()


def test_paper_client_uses_canonical_litellm_and_never_public_gpt(monkeypatch):
    from core.strategy_registry import paper_environment_defaults

    for name, value in paper_environment_defaults().items():
        monkeypatch.setenv(name, value)
    for name in tuple(os.environ):
        if name.startswith(("VLLM_", "AZURE_OPENAI")) or name in {
            "OPENAI_API_BASE",
            "OPENAI_BASE_URL",
            "OPENAI_PROVIDER",
        }:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAG_PAPER_MODE", "true")
    monkeypatch.setenv("RAG_INFERENCE_BASE_URL", "http://litellm/v1")
    monkeypatch.setattr("core.inference_transport._approved_gateway_identity", lambda: __import__("hashlib").sha256(b"http://litellm/v1").hexdigest())
    monkeypatch.setenv("RAG_INFERENCE_API_KEY", "test-key")
    monkeypatch.setenv("RAG_GENERATION_MODEL", "gemma-4-31b-it")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "qwen3-embedding-4b")
    monkeypatch.setenv("RAG_LLM_SEED", "42")
    client = VLLMClient()
    assert client.vllm_url == client.embed_url == "http://litellm/v1"
    assert client._is_openai_model("gpt-6") is False
