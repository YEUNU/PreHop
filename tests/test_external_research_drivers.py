import json
import sys
import types
from pathlib import Path

import pytest

from models.external_research.drivers.base import canonical_semantic_env, validate_documents


def _stage(tmp_path: Path):
    target = tmp_path / "run"
    (target / "input").mkdir(parents=True)
    (target / "input/corpus.json").write_text(
        json.dumps(
            [
                {"source_id": "a", "title": "A", "text": "alpha"},
                {"source_id": "b", "title": "B", "text": "beta"},
            ]
        ),
        encoding="utf-8",
    )
    return target


def test_document_contract_rejects_foreign_duplicate_and_nonfinite():
    assert validate_documents("x", [{"source_id": "a", "score": 1.0}], {"a"})[0]["source_id"] == "a"
    for rows in (
        [{"source_id": "z"}],
        [{"source_id": "a"}, {"source_id": "a"}],
        [{"source_id": "a", "score": float("nan")}],
    ):
        try:
            validate_documents("x", rows, {"a"})
        except ValueError:
            pass
        else:
            raise AssertionError("malformed retrieval output was admitted")


def test_semantic_environment_must_equal_checked_in_policy(monkeypatch):
    monkeypatch.delenv("RAG_LIGHTRAG_QUERY_MODE", raising=False)
    assert canonical_semantic_env("RAG_LIGHTRAG_QUERY_MODE", "mix") == "mix"
    monkeypatch.setenv("RAG_LIGHTRAG_QUERY_MODE", "mix")
    assert canonical_semantic_env("RAG_LIGHTRAG_QUERY_MODE", "mix") == "mix"
    monkeypatch.setenv("RAG_LIGHTRAG_QUERY_MODE", "local")
    with pytest.raises(RuntimeError, match="checked-in paper semantic policy"):
        canonical_semantic_env("RAG_LIGHTRAG_QUERY_MODE", "mix")


def test_lightrag_uses_pinned_async_contract_and_exact_file_path(monkeypatch, tmp_path):
    calls = {}

    class QueryParam:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class EmbeddingFunc:
        def __init__(self, **kwargs):
            calls["embedding"] = kwargs

    class LightRAG:
        def __init__(self, **kwargs):
            calls["config"] = kwargs
            self.max_parallel_insert = kwargs.get("max_parallel_insert", 2)

        async def initialize_storages(self):
            calls["initialized"] = True

        async def ainsert(self, docs, **kwargs):
            calls["insert"] = (docs, kwargs)
            return "track-1"

        async def aget_docs_by_track_id(self, track_id):
            assert track_id == "track-1"
            return {"a": types.SimpleNamespace(status="processed"), "b": types.SimpleNamespace(status="processed")}

        async def aquery_llm(self, question, param):
            return {
                "status": "success",
                "data": {
                    "chunks": [
                        {"file_path": "a", "content": "one"},
                        {"file_path": "a", "content": "two"},
                    ]
                },
                "llm_response": {"content": "native"},
            }

        async def finalize_storages(self):
            calls["finalized"] = True

    async def complete(*args, **kwargs):
        calls["llm"] = kwargs
        return "x"

    def retry_with(**kwargs):
        calls["retry_stop"] = kwargs["stop"]
        return complete

    complete.retry_with = retry_with

    modules = {
        "lightrag": types.SimpleNamespace(LightRAG=LightRAG, QueryParam=QueryParam),
        "lightrag.llm": types.ModuleType("lightrag.llm"),
        "lightrag.llm.openai": types.SimpleNamespace(openai_complete_if_cache=complete),
        "lightrag.utils": types.SimpleNamespace(EmbeddingFunc=EmbeddingFunc),
        "tenacity": types.SimpleNamespace(
            stop_after_attempt=lambda attempts: types.SimpleNamespace(max_attempt_number=attempts)
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("VLLM_API_BASE", "http://generation/v1")
    monkeypatch.setenv("VLLM_EMBED_API_BASE", "http://embedding/v1")
    monkeypatch.setenv("VLLM_SERVED_MODEL_NAME", "generation")
    monkeypatch.setenv("VLLM_SERVED_EMBED_MODEL_NAME", "embedding")
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    monkeypatch.setenv("RAG_INFERENCE_BASE_URL", "http://litellm/v1")
    monkeypatch.setenv("RAG_INFERENCE_API_KEY", "test-key")
    monkeypatch.setenv("RAG_GENERATION_MODEL", "generation")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "embedding")
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", "16")
    monkeypatch.setenv("RAG_EMBEDDING_CONCURRENCY", "1")
    monkeypatch.setenv("RAG_INFERENCE_RETRY_ATTEMPTS", "7")
    monkeypatch.setenv("RAG_INFERENCE_TIMEOUT", "23")
    monkeypatch.setenv("NEO4J_VECTOR_DIMENSIONS", "2560")
    from core.vllm_client import VLLMClient
    from models.external_research.drivers.lightrag import LightRAGDriver

    async def fake_embeddings(self, texts, encoding_type=None):
        import numpy as np
        return np.ones((len(texts), 2560), dtype=np.float32).tolist()

    monkeypatch.setattr(VLLMClient, "get_embeddings", fake_embeddings)

    driver = LightRAGDriver(tmp_path / "official", _stage(tmp_path))
    driver.index()
    result = driver.query("q")
    assert calls["initialized"]
    assert calls["config"]["embedding_batch_num"] == 16
    assert calls["config"]["embedding_func_max_async"] == 1
    assert calls["embedding"]["embedding_dim"] == 2560
    assert calls["embedding"]["supports_asymmetric"] is True
    assert calls["retry_stop"].max_attempt_number == 7
    assert driver.loop.run_until_complete(calls["config"]["llm_model_func"]("prompt")) == "x"
    assert calls["llm"]["timeout"] == 23.0
    embedded = driver.loop.run_until_complete(calls["embedding"]["func"](["x", "y"]))
    assert embedded.shape == (2, 2560)
    assert embedded.__class__.__module__ == "numpy"
    driver.close()
    assert calls["finalized"]
    assert calls["insert"][1]["ids"] == ["a", "b"]
    assert calls["insert"][1]["file_paths"] == ["a", "b"]
    assert result == {"documents": [{"source_id": "a", "title": "A", "text": "one"}], "answer": "native"}


def test_linear_driver_has_no_visible_source_marker():
    source = Path("models/external_research/drivers/linear_rag.py").read_text(encoding="utf-8")
    assert "[SOURCE_ID=" not in source
    assert "SentenceTransformer(\n                pinned_model_path" in source


def test_linear_resolves_pinned_snapshot_before_old_sentence_transformer(tmp_path):
    from models.external_research.drivers.linear_rag import resolve_pinned_model_path

    source = tmp_path / "source"
    (tmp_path / "artifacts/embedding").mkdir(parents=True)
    assert resolve_pinned_model_path(source) == str(tmp_path / "artifacts/embedding")


def test_linear_constructor_uses_published_runner_query_defaults(monkeypatch, tmp_path):
    import ast

    from core.strategy_registry import get_strategy
    from models.external_research.drivers import linear_rag as driver
    captured = {}
    class Encoder:
        def __init__(self, path): pass
    def config(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, 'sentence_transformers', types.SimpleNamespace(SentenceTransformer=Encoder))
    monkeypatch.setitem(sys.modules, 'src.config', types.SimpleNamespace(LinearRAGConfig=config))
    monkeypatch.setitem(sys.modules, 'src.LinearRAG', types.SimpleNamespace(LinearRAG=lambda **kwargs: types.SimpleNamespace(config=kwargs['global_config'])))
    monkeypatch.setattr(driver, 'resolve_pinned_model_path', lambda _: 'pinned-model')
    monkeypatch.setattr(driver, 'LinearNativeInference', lambda _: object())
    before = sys.path[:]
    try:
        engine = driver.LinearRAGDriver(tmp_path/'source', _stage(tmp_path)).engine
    finally:
        sys.path[:] = before
    expected = {'iteration_threshold': 0.4, 'passage_ratio': 2.0, 'top_k_sentence': 3}
    for field, value in expected.items():
        assert getattr(engine.config, field) == value
        assert field not in dict(get_strategy('linear_rag').paper_index_policy)
    native = Path('data/official_baselines/qwen06-1024/linear_rag/source/run.py')
    if native.exists():
        defaults = {}
        for node in ast.walk(ast.parse(native.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument' and node.args:
                flag = ast.literal_eval(node.args[0])
                for arg in node.keywords:
                    if arg.arg == 'default' and flag.removeprefix('--') in expected:
                        defaults[flag.removeprefix('--')] = ast.literal_eval(arg.value)
        assert defaults == expected
