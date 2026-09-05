import pytest

from core.index_namespace import index_namespace


def test_index_namespace_defaults_to_corpus_tag(monkeypatch):
    monkeypatch.delenv("RAG_INDEX_NAMESPACE", raising=False)

    assert index_namespace("multi-hop.rag") == "multi_hop_rag"


def test_index_namespace_is_explicitly_run_scoped(monkeypatch):
    monkeypatch.setenv("RAG_INDEX_NAMESPACE", "multihoprag_paper-run.01")

    assert index_namespace("multihoprag") == "multihoprag_paper_run_01"


def test_index_namespace_rejects_empty_sanitized_value(monkeypatch):
    monkeypatch.setenv("RAG_INDEX_NAMESPACE", "---")

    with pytest.raises(ValueError, match="letter or digit"):
        index_namespace("multihoprag")
