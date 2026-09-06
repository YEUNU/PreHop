import ast
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from core.strategy_registry import get_strategy
from models.external_research.drivers.base import canonical_semantic_env, validate_documents
from models.external_research.drivers.youtu_graphrag import (
    YoutuGraphRAGDriver,
    classify_native_extraction,
    configure_native_openai_client,
    derive_native_source_reachability,
    validate_native_query_result,
)


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


def test_youtu_duplicate_fact_preserves_native_first_source_limitation():
    # The pinned native graph keeps the first entity chunk when an identical
    # fact is reused. Observation records that limitation without rewriting it.
    graph = [
        {
            "start_node": {"label": "entity", "properties": {"name": "a", "chunk id": "chunk-a"}},
            "relation": "r",
            "end_node": {"label": "entity", "properties": {"name": "b", "chunk id": "chunk-a"}},
        }
    ]
    observed = derive_native_source_reachability(
        graph, {"chunk-a": "source-a", "chunk-b": "source-b"}, {"source-a", "source-b"}
    )
    assert observed == {
        "reachable_sources": {"source-a": ["chunk-a"]},
        "unreachable_source_ids": ["source-b"],
        "complete": False,
    }


@pytest.mark.parametrize(
    ("parsed", "expected"),
    [
        ({"attributes": {"a": ["x"]}, "triples": []}, "success"),
        ({"attributes": {}, "triples": [["a", "r", "b"]]}, "success"),
        ({"attributes": {}, "triples": []}, "empty"),
        (None, "malformed"),
        ({"attributes": [], "triples": []}, "malformed"),
        ({"attributes": {}, "triples": [["too", "short"]]}, "malformed"),
    ],
)
def test_youtu_native_extraction_classification_is_fail_closed(parsed, expected):
    assert classify_native_extraction(parsed) == expected


def test_youtu_client_injection_uses_typed_retry_and_timeout_without_replacing_component():
    calls = []
    original = object()
    client = types.SimpleNamespace(with_options=lambda **kwargs: calls.append(kwargs) or original)
    component = types.SimpleNamespace(llm_client=types.SimpleNamespace(client=client))

    configure_native_openai_client(component, retry_attempts=5, timeout_seconds=600.0)

    assert calls == [{"max_retries": 5, "timeout": 600.0}]
    assert component.llm_client.client is original


def _native_youtu_result(**overrides):
    result = {
        "initial_answer": "answer",
        "chunk_ids": ["chunk-b", "chunk-a"],
        "chunk_contents": ["native B", "native A"],
        "triples": ["(B, relates, A)"],
        "sub_question_results": [
            {"sub_question": "q", "triples_count": 1, "chunk_ids_count": 2, "time_taken": 0.1}
        ],
    }
    result.update(overrides)
    return result


def test_youtu_query_calls_public_top_level_api_and_preserves_native_order():
    calls = []

    def initial_question_decomposition(graphq, retriever, question, schema_path):
        calls.append((graphq, retriever, question, schema_path))
        return _native_youtu_result()

    driver = object.__new__(YoutuGraphRAGDriver)
    driver.official_main = types.SimpleNamespace(initial_question_decomposition=initial_question_decomposition)
    driver.config = types.SimpleNamespace(triggers=types.SimpleNamespace(mode="noagent"))
    driver.graphq = object()
    driver.retriever = object()
    driver.schema_path = Path("schemas/hotpot.json")
    driver.chunk_sources = {"chunk-a": "source-a", "chunk-b": "source-b"}
    driver.by_id = {
        "source-a": {"title": "A", "text": "staged A"},
        "source-b": {"title": "B", "text": "staged B"},
    }

    result = driver.query("question")

    assert calls == [(driver.graphq, driver.retriever, "question", "schemas/hotpot.json")]
    assert result == {
        "documents": [
            {"source_id": "source-b", "title": "B", "text": "native B"},
            {"source_id": "source-a", "title": "A", "text": "native A"},
        ],
        "answer": "answer",
    }


def test_youtu_exact_pinned_checkout_declares_the_structured_top_level_api():
    root = Path("data/official_baselines/youtu_graphrag/source")
    if not root.is_dir():
        pytest.skip("pinned Youtu checkout is prepared only after static GO")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == get_strategy("youtu_graphrag").revision
    module = ast.parse((root / "main.py").read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    query_api = functions["initial_question_decomposition"]
    assert [argument.arg for argument in query_api.args.args] == [
        "graphq",
        "kt_retriever",
        "question",
        "schema_path",
    ]
    returned_keys = {
        key.value
        for node in ast.walk(query_api)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
        for key in node.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert {"chunk_ids", "chunk_contents", "sub_question_results", "initial_answer"} <= returned_keys


@pytest.mark.parametrize(
    ("native_result", "message"),
    [
        (None, "malformed result"),
        (_native_youtu_result(initial_answer=""), "empty answer"),
        (_native_youtu_result(initial_answer="Error: Unable to generate answer"), "exhausted generation retries"),
        (_native_youtu_result(chunk_ids=[], chunk_contents=[]), "empty or misaligned"),
        (_native_youtu_result(chunk_contents=["No relevant chunks found", "native A"]), "sentinel"),
        (
            _native_youtu_result(
                sub_question_results=[
                    {"sub_question": "q", "triples_count": 0, "chunk_ids_count": 0, "time_taken": 0.0}
                ]
            ),
            "swallowed sub-question failure",
        ),
    ],
)
def test_youtu_native_query_validation_rejects_sentinel_malformed_and_swallowed_errors(
    native_result, message
):
    with pytest.raises((RuntimeError, TypeError), match=message):
        validate_native_query_result(native_result, {"chunk-a": "source-a", "chunk-b": "source-b"})


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


def test_youtu_driver_records_source_before_native_chunking():
    source = Path("models/external_research/drivers/youtu_graphrag.py").read_text(encoding="utf-8")
    assert "super().chunk_text(text)" in source
    assert '"source_id": r["source_id"]' in source
    assert "content in document" not in source
    assert "self.dataset_name" in source
    assert "top_k=self.config.retrieval.top_k_filter" in source
    assert '"staged_input_coverage_complete"' in source
    assert '"native_extraction_success_complete"' in source
    assert "source_graph_evidence.json" in source
    assert "source_extraction_evidence.json" in source
    assert "persisted an empty graph artifact" in source
    assert "derive_native_source_reachability(graph_output" in source
    assert "def triple_deduplicate" not in source
    assert "def format_output" not in source
    assert "_extract_chunk_ids_from_nodes" not in source
    assert "LLMCompletionCall.call_api =" not in source
    assert "initial_question_decomposition" in source
    assert "all_chunk_ids = set" not in source
    assert "source_ids = sorted" not in source
