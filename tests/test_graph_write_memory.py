"""Transaction grouping must preserve complete document payloads and fail closed."""
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from neo4j.exceptions import TransientError

from core.config import RAGConfig
from models.prehop.graphrag import GraphRAG
from models.prehop.indexing import graph_writer
from models.prehop.indexing.graph_writer import GraphWriteFailed, _logical_payload_bytes


def memory_error():
    error = TransientError("transaction memory limit")
    error._neo4j_code = "Neo.TransientError.General.MemoryPoolOutOfMemoryError"
    return error


def documents():
    return [
        {"doc_id": f"{i}.txt", "doc_title": "문서", "data": [
            {"id": f"{i}-0", "embedding": [0.25] * 2560,
             "q_minus": [{"id": f"{i}-q", "text": "Full question?", "embedding": [0.5] * 2560}],
             "q_plus": [{"id": f"{i}-p", "embedding": [0.75] * 2560, "query_embedding": [1.0] * 2560}],
             "sentences": []},
            {"id": f"{i}-1", "embedding": [0.125] * 2560, "q_minus": [], "q_plus": [], "sentences": []},
        ]} for i in range(4)
    ]


@pytest.mark.asyncio
async def test_payload_grouping_preserves_every_document_and_next_order(monkeypatch):
    rag = GraphRAG(strategy="prehop")
    original = documents()
    rag._pending_batch = deepcopy(original)
    monkeypatch.setattr(graph_writer, "GRAPH_WRITE_PAYLOAD_BYTES", _logical_payload_bytes(original[0]) + 1)
    rag.retry_query = AsyncMock(return_value=[])
    await rag.flush_graph_batch()
    calls = rag.retry_query.await_args_list
    assert len(calls) == 4
    assert [doc for call in calls for doc in call.args[1]["documents"]] == original
    assert all("MERGE (c1)-[:NEXT]->(c2)" in call.args[0] for call in calls)
    assert not rag._pending_batch
    assert not rag.graph_write_failed


@pytest.mark.asyncio
async def test_memory_split_commits_each_document_once_without_payload_changes():
    rag = GraphRAG(strategy="prehop")
    original = documents()
    rag._pending_batch = deepcopy(original)
    committed = []
    attempts = []

    async def write(_query, parameters):
        batch = parameters["documents"]
        attempts.append(len(batch))
        if len(batch) > 1:
            raise memory_error()
        committed.extend(deepcopy(batch))
        return []

    rag.retry_query = AsyncMock(side_effect=write)
    await rag.flush_graph_batch()
    assert attempts == [4, 2, 1, 1, 2, 1, 1]
    assert committed == original
    assert rag._pending_batch == []
    assert not rag.graph_write_failed


@pytest.mark.asyncio
async def test_failed_suffix_does_not_replay_successful_prefix_or_accept_more_documents():
    rag = GraphRAG(strategy="prehop")
    original = documents()
    rag._pending_batch = deepcopy(original)
    committed = []

    async def write(_query, parameters):
        batch = parameters["documents"]
        if len(batch) > 1 or batch[0]["doc_id"] == "2.txt":
            raise memory_error()
        committed.extend(deepcopy(batch))
        return []

    rag.retry_query = AsyncMock(side_effect=write)
    with pytest.raises(TransientError):
        await rag.flush_graph_batch()
    assert committed == original[:2]
    assert rag._pending_batch == original[2:]
    assert rag.graph_write_failed
    call_count = rag.retry_query.await_count
    with pytest.raises(GraphWriteFailed):
        await rag.flush_graph_batch()
    with pytest.raises(GraphWriteFailed):
        await rag.build_graph({"chunks": []}, source="new", document_filename="new.txt")
    assert rag.retry_query.await_count == call_count
    assert rag._pending_batch == original[2:]


@pytest.mark.asyncio
async def test_nonmemory_error_is_never_split_or_retried_by_grouping():
    rag = GraphRAG(strategy="prehop")
    original = documents()
    rag._pending_batch = deepcopy(original)
    rag.retry_query = AsyncMock(side_effect=ValueError("invalid graph data"))
    with pytest.raises(ValueError):
        await rag.flush_graph_batch()
    rag.retry_query.assert_awaited_once()
    assert rag._pending_batch == original
    assert rag.graph_write_failed


@pytest.mark.asyncio
async def test_memory_limit_bypasses_transient_retry_sleep(monkeypatch):
    rag = GraphRAG(strategy="prehop")
    rag.neo4j.execute_query = AsyncMock(side_effect=memory_error())
    sleep = AsyncMock()
    monkeypatch.setattr(graph_writer.asyncio, "sleep", sleep)
    with pytest.raises(TransientError):
        await rag.retry_query("query", {})
    rag.neo4j.execute_query.assert_awaited_once()
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_document_count_limit_and_oversized_singleton_remain_atomic(monkeypatch):
    rag = GraphRAG(strategy="prehop")
    original = documents()
    rag._pending_batch = deepcopy(original)
    monkeypatch.setattr(RAGConfig, "NEO4J_BATCH_SIZE", 2)
    rag.retry_query = AsyncMock(return_value=[])
    await rag.flush_graph_batch()
    assert [len(c.args[1]["documents"]) for c in rag.retry_query.await_args_list] == [2, 2]
    monkeypatch.setattr(graph_writer, "GRAPH_WRITE_PAYLOAD_BYTES", 1)
    rag._pending_batch = deepcopy(original[:1])
    rag.retry_query.reset_mock()
    await rag.flush_graph_batch()
    assert rag.retry_query.await_args.args[1]["documents"] == original[:1]
