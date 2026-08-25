import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from data.prepare_musique import paragraph_identity
from models.hoprag.hoprag_adapter import HopRAGAdapter
from models.hoprag.official_indexer import _build_official_edge_groups, _prune_stale_hoprag_sources


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
async def test_hoprag_provenance_rejects_same_text_from_different_sources():
    adapter, _session = _adapter_with_rows(
        [
            {"idx": 0, "id": 17, "title": "First", "source": "first_doc", "text": "same text"},
            {"idx": 0, "id": 23, "title": "Second", "source": "second_doc", "text": "same text"},
        ]
    )

    with pytest.raises(RuntimeError, match=r"provenance is ambiguous.*first_doc.*second_doc"):
        await adapter._lookup_nodes_by_text(["same text"])


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
