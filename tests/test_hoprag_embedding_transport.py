from types import SimpleNamespace

import pytest

from core.inference_transport import InferenceTransport
from models.hoprag.official_indexer import _VLLMEmbedClient


class Response:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.ok = 200 <= status < 300

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            error = RuntimeError(self.text)
            error.status_code = self.status_code
            raise error


def _client(responses):
    client = _VLLMEmbedClient(InferenceTransport.resolve('hoprag', {
        'RAG_INFERENCE_BASE_URL': 'http://unused', 'RAG_INFERENCE_API_KEY': 'fixture-key',
        'RAG_EMBEDDING_MODEL': 'embedding-model', 'NEO4J_VECTOR_DIMENSIONS': '2',
        'RAG_EMBEDDING_BATCH_SIZE': '2', 'RAG_INFERENCE_TIMEOUT': '19',
    }))
    client._sess.close()
    client._sess = SimpleNamespace(post=lambda *args, **kwargs: responses.pop(0))
    return client


def test_hoprag_messagepack_bisection_preserves_order():
    responses = [
        Response(400, text="MessagePack data is malformed: trailing characters"),
        Response(200, {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}),
        Response(200, {"data": [{"index": 0, "embedding": [2.0, 0.0]}]}),
    ]
    client = _client(responses)
    assert client._request_batch(["a", "b"]) == [[1.0, 0.0], [2.0, 0.0]]


def test_hoprag_unrelated_400_fails_without_bisection():
    responses = [Response(400, text="unknown embedding model")]
    client = _client(responses)
    with pytest.raises(RuntimeError, match="unknown embedding model"):
        client._request_batch(["a", "b"])
    assert responses == []


def test_hoprag_embeddings_use_resolved_batch_and_request_settings():
    client = _client([])
    batches = []
    def post(url, **kwargs):
        assert url == 'http://unused/embeddings'
        assert kwargs['headers'] == {'Authorization': 'Bearer fixture-key'}
        assert kwargs['timeout'] == 19
        batch = kwargs['json']['input']
        batches.append(batch)
        return Response(200, {'data': [{'index': i, 'embedding': [1., 0.]} for i, _ in enumerate(batch)]})
    client._sess.post = post
    result = client.encode(['a', 'b', 'c'])
    assert batches == [['a', 'b'], ['c']]
    assert result.tolist() == [[1., 0.]] * 3
