import pandas as pd

from models.ms_graphrag.ms_adapter import MSGraphRAGAdapter
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
