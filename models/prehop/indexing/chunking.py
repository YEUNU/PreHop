"""Create fixed sentence windows within individual pages.

Each page is split into sentences and grouped into fixed-size windows of
`RAGConfig.CHUNK_SENTENCES` sentences. The final partial window is retained.
No embedding-similarity decisions and no
cross-page grouping — chunk boundaries never cross a page. Source text,
including pipe-delimited fragments, is preserved without an LLM conversion.
"""

import hashlib
import json
import logging
import os
import re
import threading
from typing import Any

from core.config import RAGConfig
from utils.prompts import (
    GROUNDED_HOPRAG_FORMAT_INSTRUCTION,
    GROUNDED_HOPRAG_PROMPT,
    HOPRAG_FORMAT_INSTRUCTION,
    HOPRAG_PROMPT,
)

logger = logging.getLogger(__name__)


def _make_semantic_chunk_id(source, title, sent_id):
    content_sig = f"{source}-{title}-{sent_id}"
    return hashlib.md5(content_sig.encode()).hexdigest()


# Chunk cache for skipping repeated question generation.
# After a successful `extract_knowledge` we persist the resulting chunks
# (Q-/Q+ and text/page/sent_id metadata) to
# `data/index_cache/<version>/<corpus_tag>/<source>__<sha8>__<ablation_sig>.json`.
# Rerunning indexing on the same file under the same generation settings
# flags and prompt text loads the cache and returns the prior knowledge dict
# Embeddings use a separate cache keyed by model, revision, encoding role,
# dimensions, instruction, endpoint, and normalized text.

_CHUNK_CACHE_VERSION = "v1"  # v1 preserves raw table/pipe text; no table-to-text generation


def _chunk_cache_root() -> str:
    return os.environ.get("RAG_CHUNK_CACHE_DIR", os.path.join("data", "index_cache"))


def _chunk_cache_enabled() -> bool:
    return os.environ.get("RAG_CHUNK_CACHE", "on").strip().lower() not in {"off", "false", "0", "no"}


def _prompt_sig() -> str:
    """Hash of every prompt template whose text ends up baked into a cached
    chunk. Editing a prompt changes this hash, so old cache
    entries are never reused under a changed prompt — they just sit
    untouched on disk under their old key (nothing is deleted) while a fresh
    run writes new entries under the new key."""
    combined = (
        GROUNDED_HOPRAG_PROMPT + GROUNDED_HOPRAG_FORMAT_INSTRUCTION
        if RAGConfig.QUESTION_SCHEMA == "grounded_v1"
        else HOPRAG_PROMPT + HOPRAG_FORMAT_INSTRUCTION
    )
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:8]


def _ablation_signature() -> str:
    """Cache key fragment that invalidates when the chunking-relevant
    ablation flags, chunk-size setting, or prompt text change (different
    ablation/prompt = different chunk shape or content)."""
    return (
        f"qm={int(RAGConfig.ABLATION_Q_MINUS)}"
        f"-qp={int(RAGConfig.ABLATION_Q_PLUS)}"
        f"-cs={RAGConfig.CHUNK_SENTENCES}"
        f"-schema={RAGConfig.QUESTION_SCHEMA}"
        f"-prompt={_prompt_sig()}"
    )


def _content_sha8(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]


def _chunk_cache_path(corpus_tag: str, source: str, content_sha: str) -> str:
    safe_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", corpus_tag or "default")
    safe_src = re.sub(r"[^A-Za-z0-9_.-]+", "_", source or "doc")
    abl = re.sub(r"[^A-Za-z0-9=_-]+", "_", _ablation_signature())
    fname = f"{safe_src}__{content_sha}__{abl}.json"
    return os.path.join(_chunk_cache_root(), _CHUNK_CACHE_VERSION, safe_tag, fname)


def _chunk_cache_load(corpus_tag: str, source: str, content: str) -> "dict[str, Any] | None":
    if not _chunk_cache_enabled():
        return None
    path = _chunk_cache_path(corpus_tag, source, _content_sha8(content))
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "chunks" not in data:
        return None
    return data


def _chunk_cache_save(corpus_tag: str, source: str, content: str, knowledge: dict) -> None:
    if not _chunk_cache_enabled():
        return
    path = _chunk_cache_path(corpus_tag, source, _content_sha8(content))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(knowledge, fh, ensure_ascii=False)
    os.replace(tmp_path, path)


# Top-level (picklable) page-parsing helper for ProcessPoolExecutor offload.
# All page split / regex work runs on a worker process so the main asyncio
# loop can keep dispatching LLM/embedding calls concurrently. Output is a
# plain dict — no graph-rag state, no tokenizer / numpy dependencies.
_PAGE_RE = re.compile(r"-+\s*Page\s*(\d+)\s*-+", re.IGNORECASE)


def parse_pages_offline(filename: str, content: str) -> dict[str, Any]:
    """Pure-CPU page parsing extracted from `extract_knowledge`.

    Splits the document text on `--- Page N ---` markers and returns title +
    ordered page list. Designed to be safely run inside
    `concurrent.futures.ProcessPoolExecutor`.
    """
    lines = content.split("\n")
    title = filename
    if lines and lines[0].startswith("Title: "):
        title = lines[0].replace("Title: ", "").strip()

    # MuSiQue preparation writes a stable paragraph identity for evaluation.
    # It is source metadata, not evidence text: remove it before splitting,
    # embedding, or prompting while retaining it for callers that need audit
    # metadata. Every benchmark adapter exposes the filename identity instead.
    paragraph_id = ""
    body_lines = lines
    if len(lines) > 1 and lines[1].startswith("Paragraph-ID: "):
        paragraph_id = lines[1].replace("Paragraph-ID: ", "").strip()
        body_lines = [lines[0], *lines[2:]]
        content = "\n".join(body_lines)

    matches = list(_PAGE_RE.finditer(content))
    pages: list[dict[str, Any]] = []
    if matches:
        for index, start_match in enumerate(matches):
            page_num = int(start_match.group(1))
            content_start = start_match.end()
            content_end = matches[index + 1].start() if index < len(matches) - 1 else len(content)
            page_text = content[content_start:content_end].strip()
            if page_text:
                pages.append({"num": page_num, "content": page_text})

    if not pages:
        # The prepared corpora may omit page markers. Such input is one page;
        # this is a supported input form, not a model-generated conversion.
        start_idx = 0
        if body_lines and body_lines[0].startswith("Title: "):
            start_idx = 1
        body = "\n".join(body_lines[start_idx:]).strip()
        if body:
            pages = [{"num": 1, "content": body}]

    return {"filename": filename, "title": title, "paragraph_id": paragraph_id, "pages": pages}


def split_fixed_sentence_windows(
    text: str,
    chunk_sentences: int | None = None,
) -> list[str]:
    """Split one page into Prehop's exact fixed sentence windows."""
    chunk_sentences = RAGConfig.CHUNK_SENTENCES if chunk_sentences is None else chunk_sentences
    if chunk_sentences < 1:
        raise ValueError("chunk_sentences must be positive")
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    chunks = [
        " ".join(sentences[offset : offset + chunk_sentences]) for offset in range(0, len(sentences), chunk_sentences)
    ]
    return chunks


class ChunkingMixin:
    def _save_debug(self, doc_name: str, step: str, data: Any):
        safe_doc_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(doc_name or "document")).strip("._-")
        doc_dir = os.path.join(self.debug_output_dir, safe_doc_name or "document")
        os.makedirs(doc_dir, exist_ok=True)
        filepath = os.path.join(doc_dir, f"{step}.json")
        tmp_path = f"{filepath}.{os.getpid()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, filepath)
        logger.info("[DEBUG] Saved %s to %s", step, filepath)

    async def extract_knowledge(
        self,
        content: str,
        source: str = "",
        prepared_pages: "dict[str, Any] | None" = None,
    ) -> dict[str, Any]:
        # The on-disk cache skips Q-/Q+ generation for unchanged source text.
        # when the same source content was already chunked under the same
        # ablation flags. Cache key = sha256(content) + ablation signature.
        cached = _chunk_cache_load(self.corpus_tag, source, content)
        if cached is not None:
            cached_title = cached.get("title", "Unknown")
            chunk_count = len(cached.get("chunks") or [])
            logger.info(
                "[%s] chunk cache HIT (corpus=%s, chunks=%d) — skipping LLM regen",
                cached_title,
                self.corpus_tag,
                chunk_count,
            )
            if self.save_intermediate:
                self._save_debug(source or cached_title, "final_chunks", cached.get("chunks") or [])
            return cached

        parsed = prepared_pages if prepared_pages is not None else parse_pages_offline(source, content)
        title = str(parsed.get("title") or source)
        pages = list(parsed.get("pages") or [])
        logger.info("[%s] Parsed %d pages%s", title, len(pages), " (offloaded)" if prepared_pages else "")
        if not pages:
            raise ValueError(f"No indexable text found in source={source!r}")

        async def process_page(page: dict[str, Any]) -> list[dict[str, Any]]:
            page_num = page["num"]
            page_content = page["content"]
            if not page_content:
                return []

            chunk_texts = split_fixed_sentence_windows(page_content)

            if not chunk_texts:
                return []

            # Files already run concurrently in cli/index.py. Process chunks
            # within one file in source order so a document cannot enqueue an
            # unbounded second layer of generation tasks. The endpoint-wide
            # limiter remains the final request-capacity guard.
            q_results = []
            for chunk_text in chunk_texts:
                q_results.append(await self.extract_hoprag_queries(chunk_text, title))

            return [
                {
                    "page": page_num,
                    "text": chunk_text,
                    "title": title,
                    "q_minus": q_data.get("q_minus", []),
                    "q_plus": q_data.get("q_plus", []),
                }
                for chunk_text, q_data in zip(chunk_texts, q_results)
            ]

        logger.info("[%s] Processing %d pages in source order", title, len(pages))
        per_page_results = []
        for page in pages:
            per_page_results.append(await process_page(page))

        final_chunks: list[dict] = []
        global_sent_id = 0
        for idx, page_chunks in enumerate(per_page_results):
            if page_chunks:
                logger.info(
                    "[%s] Page %d/%d done (%d chunks)",
                    title,
                    idx + 1,
                    len(pages),
                    len(page_chunks),
                )
            for page_chunk in page_chunks:
                page_chunk["sent_id"] = global_sent_id
                final_chunks.append(page_chunk)
                global_sent_id += 1

        if self.save_intermediate:
            self._save_debug(
                source or title,
                "final_chunks",
                [
                    {
                        "sent_id": chunk["sent_id"],
                        "page": chunk["page"],
                        # Keep complete text in debug output for graph comparison.
                        "text": chunk["text"],
                        "q_minus": chunk["q_minus"],
                        "q_plus": chunk["q_plus"],
                    }
                    for chunk in final_chunks
                ],
            )

        knowledge = {"title": title, "chunks": final_chunks}
        _chunk_cache_save(self.corpus_tag, source, content, knowledge)
        return knowledge
