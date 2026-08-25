"""Shared retrieval text, context, and reciprocal-rank-fusion helpers."""

import re
import unicodedata
from typing import Any


class TextUtilsMixin:
    @staticmethod
    def _normalize_entity_term(value: str) -> str:
        """Normalize entity/query tokens for robust graph matching."""
        if not value:
            return ""
        normalized = unicodedata.normalize("NFKC", str(value)).lower()
        normalized = re.sub(r"[_\-]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _sanitize_fulltext_query(value: str) -> str:
        """Sanitize free-form text for Neo4j Lucene fulltext query parser."""
        if not value:
            return ""
        normalized = unicodedata.normalize("NFKC", str(value))
        normalized = re.sub(r"[+\-!(){}\[\]^\"~*?:\\/|&]", " ", normalized)
        normalized = re.sub(r"[`]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return ""
        return normalized[:512]

    @staticmethod
    def _build_context_from_nodes(nodes: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            [
                f"[[{node['title']}, Page {node.get('page', 0)}, Chunk {node['sent_id']}]]\n{node['text']}"
                for node in nodes
            ]
        )

    def _rrf_accumulate(
        self,
        merged: dict[str, dict[str, Any]],
        nodes: list[dict[str, Any]],
        score_key: str,
        default_keys: tuple[str, ...] = (),
    ) -> None:
        """Reciprocal-rank-fusion accumulation, shared by hybrid.py's
        vector/fulltext channel fusion and retrieve.py's stage1/stage2
        candidate merging: score += 1 / (rank + 1) into
        `merged`, keyed by node identity. `default_keys` seeds every score
        field a caller's later accumulation passes might target (e.g.
        retrieve.py's direct/dependency role scores) so a
        first-seen-in-a-later-pass node still has all fields present;
        callers with a single fixed score key can omit it.
        """
        for rank, node in enumerate(nodes):
            node_id = self._node_identity(node)
            if node_id not in merged:
                item = dict(node)
                for key in default_keys or (score_key,):
                    item.setdefault(key, 0.0)
                merged[node_id] = item
            merged[node_id][score_key] += 1.0 / (rank + 1)

    @staticmethod
    def _node_identity(node: dict[str, Any]) -> str:
        node_id = str(node.get("id", "") or "").strip()
        if node_id:
            return node_id
        return f"{node.get('title', '')}:{node.get('source', '')}:{node.get('page', 0)}:{node.get('sent_id', -1)}"

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for raw in values:
            text = str(raw or "").strip()
            if not text:
                continue
            normalized = re.sub(r"\s+", " ", text.lower()).strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(text)
        return unique
