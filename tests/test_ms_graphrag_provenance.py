import asyncio
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from models.ms_graphrag import official_indexer as ms_official_indexer
from models.ms_graphrag.ms_adapter import _QA_RESPONSE_TYPE, MSGraphRAGAdapter
from models.ms_graphrag.official_indexer import _ms_indexable_text
from utils.metrics import _source_paragraph_identity


def _adapter_with_document_maps() -> MSGraphRAGAdapter:
    adapter = object.__new__(MSGraphRAGAdapter)
    adapter._short_id_to_doc_id = {"7": "opaque-document-hash"}
    adapter._doc_id_to_title = {
        "opaque-document-hash": "musique_aabbccddeeff00112233445566778899.txt",
    }
    return adapter


def test_ms_sources_keep_staged_filename_as_identity_and_header_as_display_title():
    adapter = _adapter_with_document_maps()
    context_data = {
        "sources": pd.DataFrame(
            [
                {
                    "id": 7,
                    "text": "Title: Repeated Wikipedia Title\nParagraph-ID: musique:aabbccddeeff00112233445566778899\n\nEvidence.",
                }
            ]
        )
    }

    sources = adapter._extract_sources(context_data)

    assert sources == [
        {
            "doc": "Repeated Wikipedia Title",
            "source": "musique_aabbccddeeff00112233445566778899.txt",
            "source_filename": "musique_aabbccddeeff00112233445566778899.txt",
            "document_id": "opaque-document-hash",
            "page": 0,
            "text": "Title: Repeated Wikipedia Title\nParagraph-ID: musique:aabbccddeeff00112233445566778899\n\nEvidence.",
            "sent_id": 0,
        }
    ]
    # The primary ``source`` field, not the display title, now carries the
    # stable filename used by MuSiQue paragraph-support evaluation.
    assert _source_paragraph_identity(sources[0]) == "musique:aabbccddeeff00112233445566778899"


def test_ms_source_without_title_header_uses_filename_as_display_fallback():
    adapter = _adapter_with_document_maps()
    context_data = {"sources": pd.DataFrame([{"id": "7", "text": "Evidence without a title header."}])}

    source = adapter._extract_sources(context_data)[0]

    assert source["doc"] == "musique_aabbccddeeff00112233445566778899"
    assert source["source"] == "musique_aabbccddeeff00112233445566778899.txt"


@pytest.mark.parametrize(
    ("content", "expected_title", "expected_body"),
    [
        (
            "Title: Display\nParagraph-ID: musique:abc\n\n--- Page 1 ---\nFirst.\n\n--- Page 2 ---\nSecond.",
            "Display",
            "First.\n\nSecond.",
        ),
        ("Body without headers.", "doc", "Body without headers."),
        ("\ufeffTitle: BOM\r\n\r\nBody.\r\n", "BOM", "Body."),
    ],
)
def test_ms_parser_separates_display_metadata_from_evidence(content, expected_title, expected_body):
    assert _ms_indexable_text("doc.txt", content) == (expected_title, expected_body)


@pytest.mark.parametrize("content", ["", "Title: Only", "Title: Only\nParagraph-ID: musique:abc"])
def test_ms_parser_rejects_metadata_only_documents(content):
    with pytest.raises(ValueError, match="no evidence text"):
        _ms_indexable_text("doc.txt", content)


def test_ms_staging_writes_clean_body_and_keeps_title_sidecar(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.txt").write_text(
        "Title: Alpha Display\nParagraph-ID: musique:alpha\n\nEvidence A.", encoding="utf-8"
    )
    (corpus / "beta.md").write_text("Title: Beta Display\n\nEvidence B.", encoding="utf-8")
    monkeypatch.setattr(ms_official_indexer, "_OUTPUT_ROOT", tmp_path / "output")

    staged, titles = ms_official_indexer._stage_input_files(str(corpus), "test")

    assert (staged / "alpha.txt").read_text(encoding="utf-8") == "Evidence A."
    assert (staged / "beta.md").read_text(encoding="utf-8") == "Evidence B."
    assert titles == {"alpha": "Alpha Display", "beta": "Beta Display"}


def test_ms_indexing_extends_litellm_client_cache_ttl(monkeypatch):
    import litellm

    client_cache = litellm.in_memory_llm_clients_cache
    monkeypatch.setattr(client_cache, "default_ttl", 600)
    monkeypatch.setenv("RAG_MS_LITELLM_CLIENT_TTL_SECONDS", "7200")

    ms_official_indexer._configure_litellm_client_lifecycle()

    assert client_cache.default_ttl == 7200


def test_ms_indexing_closes_litellm_clients_when_pipeline_fails(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.txt").write_text("Evidence A.", encoding="utf-8")
    monkeypatch.setattr(ms_official_indexer, "_OUTPUT_ROOT", tmp_path / "output")
    monkeypatch.setattr(ms_official_indexer, "_configure_litellm_client_lifecycle", lambda: None)
    monkeypatch.setattr(ms_official_indexer, "build_config", lambda corpus_tag, staged_input: object())

    cleanup_calls = []

    async def fail_build_index(**kwargs):
        del kwargs
        raise RuntimeError("pipeline failed")

    async def record_cleanup():
        cleanup_calls.append(True)

    import graphrag.api.index

    monkeypatch.setattr(graphrag.api.index, "build_index", fail_build_index)
    monkeypatch.setattr(ms_official_indexer, "_close_litellm_async_clients", record_cleanup)

    with pytest.raises(RuntimeError, match="pipeline failed"):
        asyncio.run(ms_official_indexer.run_official_index(str(corpus), "test"))

    assert cleanup_calls == [True]


def test_ms_source_map_uses_dataframe_index_like_official_api():
    text_units = pd.DataFrame(
        [{"human_readable_id": 7, "document_id": "doc-hash"}],
        index=[42],
    )
    documents = pd.DataFrame([{"id": "doc-hash", "title": "source.txt"}])

    short_map, doc_map = MSGraphRAGAdapter._build_source_maps(text_units, documents)

    assert short_map == {"42": "doc-hash"}
    assert "7" not in short_map
    assert doc_map == {"doc-hash": "source.txt"}


@pytest.mark.parametrize(
    ("text_units", "documents", "message"),
    [
        (pd.DataFrame([{"wrong": "x"}]), pd.DataFrame([{"id": "d", "title": "a.txt"}]), "lacks required"),
        (
            pd.DataFrame([{"document_id": "missing"}]),
            pd.DataFrame([{"id": "d", "title": "a.txt"}]),
            "references unknown document",
        ),
        (
            pd.DataFrame([{"document_id": "d"}]),
            pd.DataFrame([{"id": "d", "title": "a.txt"}, {"id": "d", "title": "b.txt"}]),
            "duplicate document id",
        ),
        (
            pd.DataFrame([{"document_id": "d1"}]),
            pd.DataFrame([{"id": "d1", "title": "a.txt"}, {"id": "d2", "title": "a.txt"}]),
            "duplicate source filename",
        ),
    ],
)
def test_ms_source_map_rejects_corrupt_parquet_links(text_units, documents, message):
    with pytest.raises(RuntimeError, match=message):
        MSGraphRAGAdapter._build_source_maps(text_units, documents)


def test_ms_extract_sources_rejects_unknown_or_incomplete_provenance():
    adapter = _adapter_with_document_maps()

    with pytest.raises(RuntimeError, match="unknown source id"):
        adapter._extract_sources({"sources": pd.DataFrame([{"id": 99, "text": "Evidence"}])})
    with pytest.raises(RuntimeError, match="incomplete provenance"):
        adapter._extract_sources({"sources": pd.DataFrame([{"id": 7, "text": ""}])})


def test_ms_source_uses_integrity_sidecar_title_after_header_removal():
    adapter = _adapter_with_document_maps()
    adapter._source_id_to_display_title = {"musique_aabbccddeeff00112233445566778899": "Mapped Display Title"}

    source = adapter._extract_sources({"sources": pd.DataFrame([{"id": 7, "text": "Clean evidence."}])})[0]

    assert source["doc"] == "Mapped Display Title"


@pytest.mark.asyncio
async def test_ms_workflow_uses_fixed_official_local_search_without_keyword_router():
    adapter = object.__new__(MSGraphRAGAdapter)
    adapter.local_search = AsyncMock(return_value=("answer", [], []))
    adapter.global_search = AsyncMock(return_value=("global", [], []))

    result = await adapter.run_workflow("Give an overall summary across documents")

    assert result == ("answer", [], [])
    adapter.local_search.assert_awaited_once()
    adapter.global_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_ms_local_search_requests_metric_compatible_short_answer(monkeypatch):
    import graphrag.api as gapi

    search = AsyncMock(return_value=("Final Answer: Paris", {}))
    monkeypatch.setattr(gapi, "local_search", search)
    adapter = object.__new__(MSGraphRAGAdapter)
    adapter._config = object()
    adapter._entities = object()
    adapter._communities = object()
    adapter._community_reports = object()
    adapter._text_units = object()
    adapter._relationships = object()
    adapter._ensure_loaded = Mock()
    adapter._extract_sources = Mock(return_value=[])

    answer, sources, trace = await adapter.local_search("Where was the person born?")

    assert answer == "Final Answer: Paris"
    assert sources == []
    assert trace == [{"step": "ms_local_search_api", "response_type": _QA_RESPONSE_TYPE}]
    assert search.await_args.kwargs["response_type"] == _QA_RESPONSE_TYPE
