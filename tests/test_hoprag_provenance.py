import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from models.hoprag import official_indexer as hop_official_indexer
from models.hoprag.hoprag_adapter import HopRAGAdapter, _run_coro_sync
from models.hoprag.official_indexer import (
    _build_official_edge_groups,
    _document_cache_digest,
    _hoprag_indexable_text,
    _prune_stale_hoprag_sources,
    _validate_cached_node,
)
from scripts.datasets.prepare_musique import paragraph_identity


class _AsyncRows:
    def __init__(self, rows):
        self._rows = iter(rows)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._rows)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def test_hoprag_sync_hook_reuses_one_worker_event_loop():
    async def current_loop_id():
        return id(asyncio.get_running_loop())

    first = _run_coro_sync(current_loop_id())
    second = _run_coro_sync(current_loop_id())

    assert first == second


def _adapter_with_rows(rows):
    adapter = object.__new__(HopRAGAdapter)
    adapter.chunk_label = "HO_test"
    result = _AsyncRows(rows)
    session = MagicMock()
    session.run = AsyncMock(return_value=result)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=None)
    adapter.neo4j = MagicMock()
    adapter.neo4j.driver.session.return_value = context
    return adapter, session


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            "Title: Display\nParagraph-ID: musique:abc\n\n--- Page 1 ---\nFirst.\n\n--- Page 2 ---\nSecond.",
            "First.\n\nSecond.",
        ),
        ("Title: Display\n\nBody keeps, commas, exactly.", "Body keeps, commas, exactly."),
        ("Paragraph-ID: musique:abc\n\nBody only.", "Body only."),
        ("No metadata.\n\nTitle: this body line is evidence.", "No metadata.\n\nTitle: this body line is evidence."),
        ("\ufeffTitle: BOM title\r\n\r\nBody.\r\n", "Body."),
    ],
)
def test_hoprag_parser_removes_only_corpus_metadata(content, expected):
    assert _hoprag_indexable_text(content) == expected


@pytest.mark.parametrize("content", ["", "Title: Only", "Title: Only\nParagraph-ID: musique:abc"])
def test_hoprag_parser_rejects_metadata_only_documents(content):
    with pytest.raises(ValueError, match="no evidence text"):
        _hoprag_indexable_text(content)


def test_hoprag_cache_digest_changes_with_parser_version(tmp_path, monkeypatch):
    source = tmp_path / "doc.txt"
    source.write_text("Title: T\n\nBody", encoding="utf-8")
    first = _document_cache_digest(source)

    monkeypatch.setattr(hop_official_indexer, "_CACHE_FORMAT_VERSION", 999)

    assert _document_cache_digest(source) != first


def _valid_cached_node(dim: int = 2):
    vector = np.ones(dim, dtype=np.float32)
    node = {"text": "Evidence.", "keywords": ["evidence"], "embed": vector}
    questions = {
        "answerable": [("What is stated?", {"evidence"}, vector)],
        "pending": [("What is needed next?", {"evidence"}, vector)],
    }
    return node, questions


def test_hoprag_cache_validator_accepts_complete_entry(monkeypatch):
    monkeypatch.setattr(hop_official_indexer, "_EMBED_DIM", 2)
    node, questions = _valid_cached_node()

    _validate_cached_node("doc.txt", node, questions)

    node["text"] = "Title: a body label that is not the document title"
    _validate_cached_node("doc.txt", node, questions, "Document title")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda node, _q: node.update(text="Title: leaked"), "contains corpus metadata"),
        (lambda node, _q: node.update(text="--- Page 3 ---"), "contains corpus metadata"),
        (lambda node, _q: node.update(keywords=[]), "invalid keywords"),
        (lambda node, _q: node.update(embed=[1.0]), "invalid node embedding"),
        (lambda node, _q: node.update(embed=[1.0, float("nan")]), "invalid node embedding"),
        (lambda _node, q: q.pop("pending"), "invalid question roles"),
        (lambda _node, q: q.update(pending=[]), "no pending questions"),
        (lambda _node, q: q.update(pending=[("", {"x"}, [1.0, 1.0])]), "invalid pending question"),
        (lambda _node, q: q.update(pending=[("Valid?", {"x"}, [1.0])]), "invalid pending embedding"),
    ],
)
def test_hoprag_cache_validator_rejects_corruption(monkeypatch, mutation, message):
    monkeypatch.setattr(hop_official_indexer, "_EMBED_DIM", 2)
    node, questions = _valid_cached_node()
    mutation(node, questions)

    with pytest.raises(RuntimeError, match=message):
        _validate_cached_node("doc.txt", node, questions, "leaked")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        (("not-a-list", []), TypeError, "expected list"),
        ((["valid", None], []), TypeError, "index 1.*expected str"),
        ((["valid", "  "], []), ValueError, "index 1 is blank"),
    ],
)
async def test_hoprag_retriever_contract_rejects_malformed_results(payload, error, message):
    adapter = object.__new__(HopRAGAdapter)
    adapter._retriever = MagicMock()
    adapter._retriever.search_docs.return_value = payload

    with pytest.raises(error, match=message):
        await adapter._run_official_retrieval("query")


@pytest.mark.asyncio
async def test_hoprag_provenance_keeps_title_and_source_separate():
    adapter, session = _adapter_with_rows(
        [
            {
                "idx": 0,
                "id": 17,
                "title": "Displayed title",
                "source": "corpus_file",
                "sent_id": 0,
                "page": 0,
                "text": "shared evidence",
                "embedding": [1.0],
            }
        ]
    )

    nodes = await adapter._lookup_nodes_by_text(["shared evidence"])

    assert nodes[0]["title"] == "Displayed title"
    assert nodes[0]["source"] == "corpus_file"
    query = session.run.await_args.args[0]
    assert "coalesce(n.title, '') AS title" in query
    assert "coalesce(n.source, '') AS source" in query


@pytest.mark.asyncio
async def test_hoprag_provenance_marks_same_text_from_different_sources_ambiguous():
    adapter, _session = _adapter_with_rows(
        [
            {"idx": 0, "id": 17, "title": "First", "source": "first_doc", "text": "same text"},
            {"idx": 0, "id": 23, "title": "Second", "source": "second_doc", "text": "same text"},
        ]
    )

    nodes = await adapter._lookup_nodes_by_text(["same text"])

    assert nodes[0]["title"] == "Ambiguous exact-text provenance"
    assert nodes[0]["source"] == ""
    assert nodes[0]["provenance_status"] == "ambiguous_exact_text"
    assert [candidate["source"] for candidate in nodes[0]["provenance_candidates"]] == [
        "first_doc",
        "second_doc",
    ]


@pytest.mark.asyncio
async def test_hoprag_provenance_collapses_duplicate_nodes_from_one_source():
    adapter, _session = _adapter_with_rows(
        [
            {"idx": 0, "id": "17", "title": "First", "source": "one_doc", "text": "same text"},
            {"idx": 0, "id": "23", "title": "First", "source": "one_doc", "text": "same text"},
        ]
    )

    nodes = await adapter._lookup_nodes_by_text(["same text"])

    assert nodes[0]["source"] == "one_doc"
    assert "provenance_status" not in nodes[0]


@pytest.mark.asyncio
async def test_hoprag_provenance_rejects_missing_official_result():
    adapter, _session = _adapter_with_rows([])

    with pytest.raises(RuntimeError, match="provenance lookup found no node"):
        await adapter._lookup_nodes_by_text(["missing text"])


@pytest.mark.asyncio
async def test_hoprag_workflow_prefers_title_for_doc_and_keeps_source():
    adapter = object.__new__(HopRAGAdapter)
    adapter.top_k = 20
    adapter.retrieve = AsyncMock(
        return_value=(
            "retrieved context",
            [
                {
                    "title": "Original article title",
                    "source": "musique_aabbccddeeff0011",
                    "page": 0,
                    "sent_id": 0,
                    "text": "supporting evidence",
                }
            ],
        )
    )
    adapter.llm = MagicMock()
    adapter.llm.generate_response = AsyncMock(return_value="answer")

    _answer, sources, _trace = await adapter.run_workflow("question")

    assert sources == [
        {
            "doc": "Original article title",
            "source": "musique_aabbccddeeff0011",
            "page": 0,
            "sent_id": 0,
            "text": "supporting evidence",
        }
    ]


def test_hoprag_musique_edge_groups_resolve_same_title_by_paragraph_identity(tmp_path, monkeypatch):
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    first_id = paragraph_identity("Repeated", "first body")
    second_id = paragraph_identity("Repeated", "second body")
    first_file = f"musique_{first_id.removeprefix('musique:')}.txt"
    second_file = f"musique_{second_id.removeprefix('musique:')}.txt"
    (staged_dir / first_file).write_text(
        f"Title: Repeated\nParagraph-ID: {first_id}\n\nfirst body", encoding="utf-8"
    )
    (staged_dir / second_file).write_text(
        f"Title: Repeated\nParagraph-ID: {second_id}\n\nsecond body", encoding="utf-8"
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw_row = {
        "id": "2hop__same-title",
        "answerable": True,
        "paragraphs": [
            {"title": "Repeated", "paragraph_text": "first body"},
            {"title": "Repeated", "paragraph_text": "second body"},
        ],
    }
    (data_dir / "musique_ans_v1.0_dev.jsonl").write_text(json.dumps(raw_row) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    groups = _build_official_edge_groups("musique", staged_dir, [first_file, second_file])

    assert groups == {"2hop__same-title": [first_file, second_file]}


def test_hoprag_snapshot_prune_is_scoped_and_clears_edges_for_rebuild():
    count_result = MagicMock()
    count_result.single.return_value = {"stale_count": 2}
    delete_result = MagicMock()
    edge_result = MagicMock()
    session = MagicMock()
    session.run.side_effect = [count_result, delete_result, edge_result]
    context = MagicMock()
    context.__enter__.return_value = session
    context.__exit__.return_value = None
    driver = MagicMock()
    driver.session.return_value = context
    builder = SimpleNamespace(label="HO_musique", driver=driver)

    changed = _prune_stale_hoprag_sources(
        builder,
        ["musique_current_a.txt", "musique_current_b.txt"],
        "HO_musique_p2a",
    )

    assert changed is True
    count_query, count_params = session.run.call_args_list[0].args
    delete_query, delete_params = session.run.call_args_list[1].args
    edge_query = session.run.call_args_list[2].args[0]
    assert "MATCH (n:HO_musique)" in count_query
    assert "MATCH (n:HO_musique)" in delete_query
    assert "[r:HO_musique_p2a]" in edge_query
    assert "(a:HO_musique)" in edge_query and "(b:HO_musique)" in edge_query
    expected_sources = ["musique_current_a", "musique_current_b"]
    assert count_params == {"active_sources": expected_sources}
    assert delete_params == {"active_sources": expected_sources}
    delete_result.consume.assert_called_once_with()
    edge_result.consume.assert_called_once_with()
