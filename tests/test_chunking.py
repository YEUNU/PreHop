import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.config import RAGConfig
from models.prehop.graphrag import GraphRAG
from models.prehop.indexing.chunking import _chunk_cache_load, _chunk_cache_save


@pytest.fixture(autouse=True)
def _disable_chunk_cache(monkeypatch):
    # extract_knowledge persists results to an on-disk cache keyed by content
    # hash + ablation signature. These tests reuse the same fixed content
    # across runs, so without this the second run (in this file or the full
    # suite) would load a stale cache entry instead of exercising the code.
    monkeypatch.setenv("RAG_CHUNK_CACHE", "off")


@pytest.mark.asyncio
async def test_fixed_size_chunking_windows(monkeypatch):
    # Pages are split into fixed sentence windows, including a partial tail.
    monkeypatch.setattr(RAGConfig, "CHUNK_SENTENCES", 3)

    rag = GraphRAG(strategy="prehop")
    rag.vllm = MagicMock()
    rag.indexing_llm = AsyncMock()
    rag.indexing_llm.generate_json = AsyncMock(
        return_value={
            "q_minus": ["q1"],
            "q_plus": ["q2"],
        }
    )

    sentences = [f"Sentence {i}." for i in range(1, 9)]  # 8 sentences
    test_content = "Title: Test Document\n--- Page 1 ---\n" + " ".join(sentences)

    knowledge = await rag.extract_knowledge(test_content, source="test")
    chunks = knowledge["chunks"]

    # 8 sentences / window=3 -> [1,2,3] [4,5,6] [7,8]; trailing window of 2
    # remains as a partial third chunk.
    assert len(chunks) == 3
    assert chunks[0]["text"] == "Sentence 1. Sentence 2. Sentence 3."
    assert chunks[1]["text"] == "Sentence 4. Sentence 5. Sentence 6."
    assert chunks[2]["text"] == "Sentence 7. Sentence 8."
    assert all(c["page"] == 1 for c in chunks)
    assert "summary" not in chunks[0]
    assert chunks[0]["q_minus"] == ["q1"]
    assert chunks[0]["q_plus"] == ["q2"]


@pytest.mark.asyncio
async def test_trailing_window_is_retained(monkeypatch):
    monkeypatch.setattr(RAGConfig, "CHUNK_SENTENCES", 3)

    rag = GraphRAG(strategy="prehop")
    rag.vllm = MagicMock()
    rag.indexing_llm = AsyncMock()
    rag.indexing_llm.generate_json = AsyncMock(
        return_value={
            "q_minus": [],
            "q_plus": [],
        }
    )

    sentences = [f"Sentence {i}." for i in range(1, 8)]  # 7 sentences
    test_content = "Title: Test Document\n--- Page 1 ---\n" + " ".join(sentences)

    knowledge = await rag.extract_knowledge(test_content, source="test")
    chunks = knowledge["chunks"]

    # 7 sentences / window=3 -> [1,2,3] [4,5,6] [7].
    assert len(chunks) == 3
    assert chunks[0]["text"] == "Sentence 1. Sentence 2. Sentence 3."
    assert chunks[1]["text"] == "Sentence 4. Sentence 5. Sentence 6."
    assert chunks[2]["text"] == "Sentence 7."


@pytest.mark.asyncio
async def test_chunks_within_document_do_not_fan_out_generation(monkeypatch):
    monkeypatch.setattr(RAGConfig, "CHUNK_SENTENCES", 1)
    rag = GraphRAG(strategy="prehop")
    active = 0
    peak = 0

    async def extract(_chunk, _title):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return {"q_minus": [], "q_plus": []}

    rag.extract_hoprag_queries = extract
    content = "Title: Ordered\n--- Page 1 ---\nOne. Two. Three. Four."

    knowledge = await rag.extract_knowledge(content, source="ordered")

    assert len(knowledge["chunks"]) == 4
    assert peak == 1


@pytest.mark.asyncio
async def test_pipe_delimited_text_is_preserved_without_conversion(monkeypatch):
    monkeypatch.setattr(RAGConfig, "CHUNK_SENTENCES", 6)

    rag = GraphRAG(strategy="prehop")
    rag.vllm = MagicMock()
    rag.indexing_llm = AsyncMock()
    rag.indexing_llm.generate_json = AsyncMock(
        return_value={
            "q_minus": ["q1"],
            "q_plus": ["q2"],
        }
    )

    # The intro sentence ends in "." before the table so the sentence
    # splitter treats it as a separate unit from the pipe-delimited table
    # blob (table lines carry no terminal punctuation of their own).
    test_content = (
        "Title: Test Document\n--- Page 1 ---\n"
        "Here is a summary table. "
        "| Year | Event |\n| 2020 | Pandemic |\n| 2021 | Vaccine |"
    )

    knowledge = await rag.extract_knowledge(test_content, source="test")
    chunks = knowledge["chunks"]

    all_text = " ".join(c["text"] for c in chunks)
    assert "Here is a summary table." in all_text
    assert "| Year | Event |" in all_text
    assert "| 2020 | Pandemic |" in all_text


def test_legacy_chunk_cache_backfills_per_chunk_title(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_CACHE", "on")
    monkeypatch.setenv("RAG_CHUNK_CACHE_DIR", str(tmp_path))
    content = "Title: Cached Document\nBody."
    legacy_knowledge = {
        "title": "Cached Document",
        "chunks": [
            {
                "sent_id": 0,
                "page": 1,
                "text": "Body.",
                "q_minus": [],
                "q_plus": [],
            }
        ],
    }
    _chunk_cache_save("test", "cached.txt", content, legacy_knowledge)

    loaded = _chunk_cache_load("test", "cached.txt", content)

    assert loaded is not None
    assert loaded["chunks"][0]["title"] == "Cached Document"
    assert "title" not in legacy_knowledge["chunks"][0]
