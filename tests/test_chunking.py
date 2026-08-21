import pytest
from unittest.mock import AsyncMock, MagicMock
from core.config import RAGConfig
from models.prehop.graphrag import GraphRAG


@pytest.fixture(autouse=True)
def _disable_chunk_cache(monkeypatch):
    # extract_knowledge persists results to an on-disk cache keyed by content
    # hash + ablation signature. These tests reuse the same fixed content
    # across runs, so without this the second run (in this file or the full
    # suite) would load a stale cache entry instead of exercising the code.
    monkeypatch.setenv("RAG_CHUNK_CACHE", "off")


@pytest.mark.asyncio
async def test_fixed_size_chunking_windows(monkeypatch):
    # Fixed-size chunking (core-only rewrite): pages are windowed into
    # CHUNK_SENTENCES-sentence chunks; a trailing window shorter than
    # MIN_CHUNK_SENTENCES merges into the previous chunk.
    monkeypatch.setattr(RAGConfig, "CHUNK_SENTENCES", 3)
    monkeypatch.setattr(RAGConfig, "MIN_CHUNK_SENTENCES", 2)

    rag = GraphRAG(strategy="prehop")
    rag.vllm = MagicMock()
    rag.indexing_llm = AsyncMock()
    rag.indexing_llm.generate_json = AsyncMock(return_value={
        "summary": "Mock summary",
        "q_minus": ["q1"],
        "q_plus": ["q2"],
    })

    sentences = [f"Sentence {i}." for i in range(1, 9)]  # 8 sentences
    test_content = "Document: Test Document\n--- Page 1 ---\n" + " ".join(sentences)

    knowledge = await rag.extract_knowledge(test_content, source="test")
    chunks = knowledge["chunks"]

    # 8 sentences / window=3 -> [1,2,3] [4,5,6] [7,8]; trailing window of 2
    # meets min_chunk_sentences=2 so it stays its own chunk (3 chunks total).
    assert len(chunks) == 3
    assert chunks[0]["text"] == "Sentence 1. Sentence 2. Sentence 3."
    assert chunks[1]["text"] == "Sentence 4. Sentence 5. Sentence 6."
    assert chunks[2]["text"] == "Sentence 7. Sentence 8."
    assert all(c["page"] == 1 for c in chunks)
    assert chunks[0]["summary"] == "Mock summary"
    assert chunks[0]["q_minus"] == ["q1"]
    assert chunks[0]["q_plus"] == ["q2"]


@pytest.mark.asyncio
async def test_trailing_window_merges_below_minimum(monkeypatch):
    monkeypatch.setattr(RAGConfig, "CHUNK_SENTENCES", 3)
    monkeypatch.setattr(RAGConfig, "MIN_CHUNK_SENTENCES", 2)

    rag = GraphRAG(strategy="prehop")
    rag.vllm = MagicMock()
    rag.indexing_llm = AsyncMock()
    rag.indexing_llm.generate_json = AsyncMock(return_value={
        "summary": "Mock summary",
        "q_minus": [],
        "q_plus": [],
    })

    sentences = [f"Sentence {i}." for i in range(1, 8)]  # 7 sentences
    test_content = "Document: Test Document\n--- Page 1 ---\n" + " ".join(sentences)

    knowledge = await rag.extract_knowledge(test_content, source="test")
    chunks = knowledge["chunks"]

    # 7 sentences / window=3 -> [1,2,3] [4,5,6] [7]; trailing window of 1 is
    # below min_chunk_sentences=2, so it merges into the previous chunk.
    assert len(chunks) == 2
    assert chunks[0]["text"] == "Sentence 1. Sentence 2. Sentence 3."
    assert chunks[1]["text"] == "Sentence 4. Sentence 5. Sentence 6. Sentence 7."


@pytest.mark.asyncio
async def test_table_lines_still_routed_through_table_to_text(monkeypatch):
    monkeypatch.setattr(RAGConfig, "CHUNK_SENTENCES", 6)
    monkeypatch.setattr(RAGConfig, "MIN_CHUNK_SENTENCES", 2)

    rag = GraphRAG(strategy="prehop")
    rag.vllm = MagicMock()
    rag.llm = MagicMock()
    rag.llm.generate_response = AsyncMock(
        return_value="Year 2020: Pandemic.\nYear 2021: Vaccine."
    )
    rag.indexing_llm = AsyncMock()
    rag.indexing_llm.generate_json = AsyncMock(return_value={
        "summary": "Mock summary",
        "q_minus": ["q1"],
        "q_plus": ["q2"],
    })

    # The intro sentence ends in "." before the table so the sentence
    # splitter treats it as a separate unit from the pipe-delimited table
    # blob (table lines carry no terminal punctuation of their own).
    test_content = (
        "Document: Test Document\n--- Page 1 ---\n"
        "Here is a summary table. "
        "| Year | Event |\n| 2020 | Pandemic |\n| 2021 | Vaccine |"
    )

    knowledge = await rag.extract_knowledge(test_content, source="test")
    chunks = knowledge["chunks"]

    rag.llm.generate_response.assert_awaited()
    all_text = " ".join(c["text"] for c in chunks)
    assert "Here is a summary table." in all_text
    assert "Year 2020: Pandemic." in all_text
