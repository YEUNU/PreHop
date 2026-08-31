"""Build a compact corpus-only continuation index from grounded Q- answers.

A direct question's answer can become the named subject of a later relation.
The linked question schema marks complete, source-verifiable entity answers as
continuation anchors. This pass records each normalized anchor once and joins
it to exact cross-document mentions without consulting benchmark questions or
labels. Sharing anchor nodes avoids materializing the Cartesian product of
every source question and every target mention for common entities.
"""

import hashlib
import logging
import re
import unicodedata
from collections import defaultdict
from typing import Any

from core.config import RAGConfig

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_ANCHOR_POLICIES = {"named_only", "all_grounded"}


def normalize_anchor_tokens(value: str) -> tuple[str, ...]:
    """Return the Unicode-normalized token identity used for exact joins."""
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return tuple(_TOKEN_RE.findall(normalized))


def _anchor_id(tokens: tuple[str, ...]) -> str:
    identity = "\x1f".join(tokens).encode("utf-8")
    return f"anchor:{hashlib.sha256(identity).hexdigest()}"


def _selected_anchor(record: dict[str, Any], anchor_policy: str) -> str:
    """Return the complete grounded answer selected by an index policy."""
    if anchor_policy not in _ANCHOR_POLICIES:
        raise ValueError(f"Unknown continuation anchor policy: {anchor_policy!r}")
    field = "answer" if anchor_policy == "all_grounded" else "continuation_anchor"
    return " ".join(str(record.get(field) or "").split())


def _anchor_mentions_by_chunk(
    anchors: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    anchor_policy: str,
) -> dict[str, set[tuple[str, ...]]]:
    """Find exact contiguous anchor mentions with one corpus scan."""
    trie: dict[str, Any] = {}
    for record in anchors:
        tokens = normalize_anchor_tokens(_selected_anchor(record, anchor_policy))
        if not tokens:
            continue
        node = trie
        for token in tokens:
            node = node.setdefault(token, {})
        node.setdefault(None, set()).add(tokens)

    mentions: dict[str, set[tuple[str, ...]]] = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "").strip()
        if not chunk_id:
            continue
        tokens = normalize_anchor_tokens(f"{chunk.get('title', '')} {chunk.get('text', '')}")
        found: set[tuple[str, ...]] = set()
        for offset, token in enumerate(tokens):
            node = trie.get(token)
            if node is None:
                continue
            cursor = offset + 1
            if None in node:
                found.update(node[None])
            while cursor < len(tokens):
                node = node.get(tokens[cursor])
                if node is None:
                    break
                if None in node:
                    found.update(node[None])
                cursor += 1
        if found:
            mentions[chunk_id] = found
    return mentions


def build_continuation_index(
    anchors: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    anchor_policy: str = "named_only",
) -> dict[str, list[dict[str, Any]]]:
    """Build shared anchors, question links, and exact-mention links."""
    mentions_by_chunk = _anchor_mentions_by_chunk(
        anchors,
        chunks,
        anchor_policy=anchor_policy,
    )
    chunk_by_id = {str(chunk.get("id") or "").strip(): chunk for chunk in chunks if str(chunk.get("id") or "").strip()}

    records_by_anchor: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    display_by_anchor: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for record in anchors:
        answer = _selected_anchor(record, anchor_policy)
        tokens = normalize_anchor_tokens(answer)
        source = str(record.get("source") or "").strip()
        question_id = str(record.get("question_id") or "").strip()
        if not tokens or not source or not question_id or not answer:
            continue
        records_by_anchor[tokens].append({"source": source, "question_id": question_id})
        display_by_anchor[tokens].add(answer)

    mention_chunks_by_anchor: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for chunk_id, mentioned_anchors in mentions_by_chunk.items():
        for tokens in mentioned_anchors:
            mention_chunks_by_anchor[tokens].add(chunk_id)

    anchor_nodes: list[dict[str, Any]] = []
    question_links: set[tuple[str, str]] = set()
    mention_links: set[tuple[str, str]] = set()
    for tokens in sorted(records_by_anchor):
        records = records_by_anchor[tokens]
        target_chunk_ids = mention_chunks_by_anchor.get(tokens, set())
        # An anchor is useful only when at least one exact mention belongs to a
        # different source document. This is a structural cross-document rule,
        # not a score or frequency threshold.
        foreign_target_ids = {
            target_id
            for target_id in target_chunk_ids
            if any(str(chunk_by_id[target_id].get("source") or "").strip() != record["source"] for record in records)
        }
        if not foreign_target_ids:
            continue

        anchor_id = _anchor_id(tokens)
        display = min(display_by_anchor[tokens], key=lambda value: (value.casefold(), value))
        anchor_nodes.append(
            {
                "id": anchor_id,
                "text": display,
                "normalized_text": " ".join(tokens),
            }
        )
        for record in records:
            question_links.add((record["question_id"], anchor_id))
        for target_id in foreign_target_ids:
            mention_links.add((anchor_id, target_id))

    return {
        "anchors": anchor_nodes,
        "question_links": [
            {"question_id": question_id, "anchor_id": anchor_id} for question_id, anchor_id in sorted(question_links)
        ],
        "mention_links": [
            {"anchor_id": anchor_id, "chunk_id": chunk_id} for anchor_id, chunk_id in sorted(mention_links)
        ],
    }


class AnswerLinkMixin:
    async def build_answer_links(self) -> int:
        """Materialize a shared exact-mention index for linked Q- answers."""
        anchor_rows = await self.retry_query(
            f"""
            MATCH (src:{self.chunk_label})-[:HAS_Q_MINUS]->(q:{self.q_minus_label})
            RETURN src.source AS source, q.id AS question_id,
                   q.answer AS answer,
                   q.continuation_anchor AS continuation_anchor
            ORDER BY q.id
            """
        )
        if not anchor_rows:
            logger.info("No grounded continuation anchors were stored; skipping answer links.")
            return 0

        chunk_rows = await self.retry_query(
            f"""
            MATCH (c:{self.chunk_label})
            RETURN c.id AS id, c.source AS source, c.title AS title, c.text AS text
            ORDER BY c.id
            """
        )
        index = build_continuation_index(
            anchor_rows,
            chunk_rows,
            anchor_policy=RAGConfig.CONTINUATION_ANCHOR_POLICY,
        )
        wave_size = 512

        for offset in range(0, len(index["anchors"]), wave_size):
            await self.retry_query(
                f"""
                UNWIND $anchors AS item
                MERGE (anchor:{self.answer_anchor_label} {{id: item.id}})
                SET anchor.text = item.text,
                    anchor.normalized_text = item.normalized_text
                RETURN count(anchor) AS anchors_written
                """,
                {"anchors": index["anchors"][offset : offset + wave_size]},
            )
        for offset in range(0, len(index["question_links"]), wave_size):
            await self.retry_query(
                f"""
                UNWIND $links AS link
                MATCH (question:{self.q_minus_label} {{id: link.question_id}})
                MATCH (anchor:{self.answer_anchor_label} {{id: link.anchor_id}})
                MERGE (question)-[:ANSWER_ANCHOR]->(anchor)
                RETURN count(anchor) AS links_written
                """,
                {"links": index["question_links"][offset : offset + wave_size]},
            )
        for offset in range(0, len(index["mention_links"]), wave_size):
            await self.retry_query(
                f"""
                UNWIND $links AS link
                MATCH (anchor:{self.answer_anchor_label} {{id: link.anchor_id}})
                MATCH (chunk:{self.chunk_label} {{id: link.chunk_id}})
                MERGE (anchor)-[:MENTIONED_IN]->(chunk)
                RETURN count(chunk) AS links_written
                """,
                {"links": index["mention_links"][offset : offset + wave_size]},
            )
        logger.info(
            "Built %d shared answer anchors, %d question links, and %d exact-mention links with policy=%s.",
            len(index["anchors"]),
            len(index["question_links"]),
            len(index["mention_links"]),
            RAGConfig.CONTINUATION_ANCHOR_POLICY,
        )
        return len(index["mention_links"])
