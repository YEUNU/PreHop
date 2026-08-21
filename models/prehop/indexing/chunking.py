"""Fixed-size chunking (paper §3.1.2 replacement — core-only rewrite).

Each page is split into sentences and grouped into fixed-size windows of
`RAGConfig.CHUNK_SENTENCES` sentences (a trailing window shorter than
`RAGConfig.MIN_CHUNK_SENTENCES` is merged into the preceding chunk instead of
being emitted on its own). No embedding-similarity decisions and no
cross-page grouping — chunk boundaries never cross a page. The non-OCR
table-to-text fallback also lives here because it operates inside the
per-page sentence iteration.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Any

from core.config import RAGConfig
from utils.prompts import (
    HOPRAG_FORMAT_INSTRUCTION,
    HOPRAG_PROMPT,
    TABLE_TO_TEXT_PROMPT,
)


logger = logging.getLogger(__name__)


def _make_semantic_chunk_id(source, title, sent_id):
    content_sig = f"{source}-{title}-{sent_id}"
    return hashlib.md5(content_sig.encode()).hexdigest()


# --- chunk cache (skip LLM regeneration on rerun) -----------------------------
#
# After a successful `extract_knowledge` we persist the resulting chunks
# (Q-/Q+, chunk_summary, text/page/sent_id metadata) to
# `data/index_cache/<corpus_tag>/<source>__<sha8>.json`. Rerunning indexing
# on the same file under the same paper-relevant ablation flags loads the
# cache and returns the prior knowledge dict — embeddings are still
# regenerated downstream, since they're cheap (vLLM batch) and the embedding
# model can change independently of LLM-generated text.

_CHUNK_CACHE_VERSION = "v4"  # v4: fixed-size chunking replaces adaptive chunking + rolling summary (core-only rewrite)


def _chunk_cache_root() -> str:
    return os.environ.get("RAG_CHUNK_CACHE_DIR", os.path.join("data", "index_cache"))


def _chunk_cache_enabled() -> bool:
    return os.environ.get("RAG_CHUNK_CACHE", "on").strip().lower() not in {"off", "false", "0", "no"}


def _prompt_sig() -> str:
    """Hash of every prompt template whose text ends up baked into a cached
    chunk (Q-/Q+/summary via HOPRAG_PROMPT, chunk text via
    TABLE_TO_TEXT_PROMPT). Editing a prompt changes this hash, so old cache
    entries are never reused under a changed prompt — they just sit
    untouched on disk under their old key (nothing is deleted) while a fresh
    run writes new entries under the new key."""
    combined = HOPRAG_PROMPT + HOPRAG_FORMAT_INSTRUCTION + TABLE_TO_TEXT_PROMPT
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:8]


def _ablation_signature() -> str:
    """Cache key fragment that invalidates when the chunking-relevant
    ablation flags, chunk-size setting, or prompt text change (different
    ablation/prompt = different chunk shape or content)."""
    return (
        f"table={int(RAGConfig.ABLATION_TABLE_TO_TEXT)}"
        f"-qm={int(RAGConfig.ABLATION_Q_MINUS)}"
        f"-qp={int(RAGConfig.ABLATION_Q_PLUS)}"
        f"-cs={RAGConfig.CHUNK_SENTENCES}"
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
    tmp_path = path + ".tmp"
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
    title = "Unknown"
    if lines and lines[0].startswith("Document: "):
        title = lines[0].replace("Document: ", "").strip()
    elif lines and lines[0].startswith("Title: "):
        title = lines[0].replace("Title: ", "").strip()

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
        # Fallback: no `--- Page N ---` markers → emit whole document body
        # under page 1 (chunking layer applies its own sentence split later).
        start_idx = 0
        if lines and (lines[0].startswith("Title: ") or lines[0].startswith("Document: ")):
            start_idx = 1
        body = "\n".join(lines[start_idx:]).strip()
        if body:
            pages = [{"num": 1, "content": body}]

    return {"filename": filename, "title": title, "pages": pages}


class ChunkingMixin:
    def _save_debug(self, doc_name: str, step: str, data: Any):
        doc_dir = os.path.join(self.debug_output_dir, doc_name.replace(" ", "_").replace("/", "_"))
        os.makedirs(doc_dir, exist_ok=True)
        filepath = os.path.join(doc_dir, f"{step}.json")
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, default=str)
        logger.info("[DEBUG] Saved %s to %s", step, filepath)

    async def _table_to_text(
        self,
        table_lines: list[str],
        title: str = "",
        page: int = 0,
    ) -> list[str]:
        """Sentence-by-sentence rendering of a markdown-pipe table.

        Used inside the per-page sentence iteration when input still contains
        raw `|`-delimited table fragments.

        Year/period environment is sourced ONLY from the table's own column
        headers. Earlier runs leaked a "2024" hallucination into chunks of
        2023 filings whenever the source dropped the year sub-header — the
        LLM had no anchor and defaulted to its training-time current year.
        Two guards: (a) detect header/data column-count mismatch up-front and
        bypass the LLM entirely; (b) prompt rule 7/8 forbid inferring a
        period from the document title.
        """
        if not table_lines:
            return []

        if not RAGConfig.ABLATION_TABLE_TO_TEXT:
            logger.info("Ablation: Skipping table-to-text conversion.")
            return table_lines

        # Pre-flight structural check. Markdown tables have the form
        # `| h1 | h2 | h3 |` with `|---|---|---|` separator. We tolerate the
        # row-label column being implicit (data row may have one extra
        # column for the row label), but anything beyond that is a sign that
        # the header was truncated and the columns can no longer be aligned
        # reliably.
        #
        # Inputs may be multi-line blobs (raw_sentences upstream is split on
        # `.!?\s`, so a single "sentence" can contain a heading + table +
        # tail prose stitched by `\n`). Expand newlines first so the
        # column-count heuristic operates on real table rows, not on the
        # entire blob — otherwise `line.split("|")` collapses dozens of
        # rows into one fake row of dozens of cells and we falsely flag
        # well-formed tables as broken.
        rows: list[list[str]] = []
        for raw in table_lines:
            for line in raw.split("\n"):
                if "|" not in line:
                    continue
                cells = [c.strip() for c in line.split("|")]
                cells = [c for c in cells if c != ""]
                if not cells:
                    continue
                if all(set(c) <= {"-", ":"} for c in cells):
                    continue  # markdown separator row
                rows.append(cells)

        if len(rows) >= 2:
            header_cols = len(rows[0])
            data_cols_max = max(len(r) for r in rows[1:])
            if header_cols + 1 < data_cols_max:
                # Two-level header tables (Case B in tests) place a year/quarter
                # sub-header in row 1, which our flat parser counts as a data
                # row and falsely flags as a mismatch. Only treat it as broken
                # when row 1 does NOT carry period tokens.
                #
                # Detection is lenient: cells may be wrapped in markdown
                # emphasis (``**Q1 2023**``), may include comparison
                # markers ("vs. Q1 2022"), or may be dates ("December 31,
                # 2022"). Anything that contains a recognizable year/quarter
                # token in <=30 chars counts as a period anchor.
                period_token_re = re.compile(
                    r"(?:19|20)\d{2}|[Qq][1-4]\b|\b[1-4][Qq]\b|FY\s?\d{2,4}|H[12]\b",
                    re.IGNORECASE,
                )

                def _looks_like_period(cell: str) -> bool:
                    stripped = cell.strip().strip("*_ ").strip()
                    if not stripped or len(stripped) > 30:
                        return False
                    return bool(period_token_re.search(stripped))

                second_row = rows[1]
                tail = second_row[1:] if len(second_row) > 1 else []
                period_hits = sum(1 for c in tail if _looks_like_period(c))
                # Require the tail to be majority-period to count as a
                # sub-header (defends against a data row that just happens to
                # contain a stray year reference).
                if not (tail and period_hits * 2 >= len(tail)):
                    logger.warning(
                        "[%s p%d] Broken table header: %d header cols vs %d data cols, "
                        "no period sub-header detected. Skipping LLM conversion to "
                        "avoid period hallucination; keeping raw markdown lines.",
                        title or "?", page, header_cols, data_cols_max,
                    )
                    return table_lines

        table_text = "\n".join(table_lines)
        context_block = (
            f"DOCUMENT: {title} (page {page})\n" if title else ""
        )
        prompt = TABLE_TO_TEXT_PROMPT + f"\n{context_block}TABLE:\n{table_text}"
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm.generate_response(messages, apply_default_sampling=False)
        converted = [sentence.strip() for sentence in response.split("\n") if sentence.strip()]
        if not converted:
            raise ValueError("Empty table-to-text conversion result")
        # Rule 8 escape hatch: LLM signals it can't safely convert.
        if any("<table-structure-unclear>" in line for line in converted):
            logger.info(
                "[%s p%d] LLM flagged table structure unclear; keeping raw lines.",
                title or "?", page,
            )
            return table_lines
        return converted

    async def extract_knowledge(
        self,
        content: str,
        source: str = "",
        prepared_pages: "dict[str, Any] | None" = None,
    ) -> dict[str, Any]:
        # On-disk chunk cache: skips every LLM call (Q-/Q+ + chunk_summary)
        # when the same source content was already chunked under the same
        # ablation flags. Cache key = sha256(content) + ablation signature.
        # Embeddings are *not* cached — they're cheap via vLLM batching and
        # the embed model can change independently.
        cached = _chunk_cache_load(self.corpus_tag, source, content)
        if cached is not None:
            cached_title = cached.get("title", "Unknown")
            chunk_count = len(cached.get("chunks") or [])
            logger.info(
                "[%s] chunk cache HIT (corpus=%s, chunks=%d) — skipping LLM regen",
                cached_title, self.corpus_tag, chunk_count,
            )
            return cached

        # Optional fast-path: caller already ran `parse_pages_offline` in a
        # ProcessPoolExecutor worker. Skip the regex/string splits here and
        # reuse the precomputed (title, pages) tuple. Falls back to in-process
        # parsing when no prepared payload is provided (preserves prior API).
        if prepared_pages is not None:
            title = prepared_pages.get("title", "Unknown")
            pages = list(prepared_pages.get("pages") or [])
            logger.info("[%s] Content head (500 chars): %r", title, content[:500])
            logger.info("[%s] Parsed %d pages (offloaded)", title, len(pages))
        else:
            lines = content.split("\n")
            title = "Unknown"
            if lines and lines[0].startswith("Document: "):
                title = lines[0].replace("Document: ", "").strip()
            elif lines and lines[0].startswith("Title: "):
                title = lines[0].replace("Title: ", "").strip()

            logger.info("[%s] Content head (500 chars): %r", title, content[:500])

            page_pattern = re.compile(r"-+\s*Page\s*(\d+)\s*-+", re.IGNORECASE)
            matches = list(page_pattern.finditer(content))

            pages = []
            if matches:
                for index, start_match in enumerate(matches):
                    page_num = int(start_match.group(1))
                    content_start = start_match.end()
                    if index < len(matches) - 1:
                        content_end = matches[index + 1].start()
                    else:
                        content_end = len(content)

                    page_text = content[content_start:content_end].strip()
                    if page_text:
                        pages.append({"num": page_num, "content": page_text})

            logger.info("[%s] Parsed %d pages from content.", title, len(pages))

        if not pages:
            logger.info("[%s] No page markers found, using standard chunking fallback", title)
            lines = content.split("\n")
            start_idx = 0
            if lines and (lines[0].startswith("Title: ") or lines[0].startswith("Document: ")):
                start_idx = 1
            content_body = "\n".join(lines[start_idx:])
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content_body) if s.strip()]
            pages = [{"num": 1, "content": " ".join(sentences)}]
            if not sentences:
                return {"title": title, "chunks": []}

        chunk_sem = asyncio.Semaphore(RAGConfig.MAX_CONCURRENT_LLM_CALLS)
        chunk_sentences = RAGConfig.CHUNK_SENTENCES
        min_chunk_sentences = RAGConfig.MIN_CHUNK_SENTENCES

        async def process_page(page: dict[str, Any]) -> list[dict[str, Any]]:
            page_num = page["num"]
            page_content = page["content"]
            if not page_content:
                return []

            raw_sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", page_content) if sentence.strip()]
            if not raw_sentences:
                return []

            processed_sentences: list[str] = []
            table_buffer: list[str] = []
            for line in raw_sentences:
                if "|" in line:
                    table_buffer.append(line)
                else:
                    if table_buffer:
                        processed_sentences.extend(
                            await self._table_to_text(table_buffer, title=title, page=page_num)
                        )
                        table_buffer = []
                    processed_sentences.append(line)
            if table_buffer:
                processed_sentences.extend(
                    await self._table_to_text(table_buffer, title=title, page=page_num)
                )

            if not processed_sentences:
                return []

            # Fixed-size sentence windows. A short trailing window (fewer
            # than min_chunk_sentences) merges into the previous chunk
            # instead of being emitted on its own.
            chunk_texts: list[str] = []
            current_group: list[str] = []
            for sentence in processed_sentences:
                current_group.append(sentence)
                if len(current_group) >= chunk_sentences:
                    chunk_texts.append(" ".join(current_group))
                    current_group = []
            if current_group:
                if chunk_texts and len(current_group) < min_chunk_sentences:
                    chunk_texts[-1] = chunk_texts[-1] + " " + " ".join(current_group)
                else:
                    chunk_texts.append(" ".join(current_group))

            if not chunk_texts:
                return []

            async def hoprag_for_chunk(chunk_text: str):
                async with chunk_sem:
                    return await self.extract_hoprag_queries(chunk_text, title)

            q_results = await asyncio.gather(*[hoprag_for_chunk(t) for t in chunk_texts])

            return [
                {
                    "page": page_num,
                    "text": chunk_text,
                    "title": title,
                    "q_minus": q_data.get("q_minus", []),
                    "q_plus": q_data.get("q_plus", []),
                    "summary": q_data.get("summary", ""),
                }
                for chunk_text, q_data in zip(chunk_texts, q_results)
            ]

        logger.info("[%s] Fan-out: processing %d pages in parallel", title, len(pages))
        per_page_results = await asyncio.gather(*[process_page(page) for page in pages])

        final_chunks: list[dict] = []
        global_sent_id = 0
        for idx, page_chunks in enumerate(per_page_results):
            if page_chunks:
                logger.info(
                    "[%s] Page %d/%d done (%d chunks)",
                    title, idx + 1, len(pages), len(page_chunks),
                )
            for page_chunk in page_chunks:
                page_chunk["sent_id"] = global_sent_id
                final_chunks.append(page_chunk)
                global_sent_id += 1

        self._save_debug(title, "final_chunks", [
            {
                "sent_id": chunk["sent_id"],
                "page": chunk["page"],
                "text": chunk["text"][:200] + "...",
                "q_minus": chunk["q_minus"],
                "q_plus": chunk["q_plus"],
                "summary": chunk["summary"],
            }
            for chunk in final_chunks
        ])

        knowledge = {"title": title, "chunks": final_chunks}
        _chunk_cache_save(self.corpus_tag, source, content, knowledge)
        return knowledge
