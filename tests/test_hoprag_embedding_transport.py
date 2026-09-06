from types import SimpleNamespace

import pytest

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
    client = object.__new__(_VLLMEmbedClient)
    client.base_url = "http://unused"
    client.model = "embedding-model"
    client.dim = 2
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


@pytest.mark.parametrize("indices", [[0, 0], [0, 2]])
def test_hoprag_requires_exact_response_indices(indices):
    data = [{"index": index, "embedding": [1.0, 0.0]} for index in indices]
    client = _client([Response(200, {"data": data})])
    with pytest.raises(ValueError, match="exact permutation"):
        client._request_batch(["a", "b"])
