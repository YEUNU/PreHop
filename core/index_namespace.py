"""Resolve the storage namespace independently from the benchmark dataset tag."""

from __future__ import annotations

import os
import re


def index_namespace(corpus_tag: str) -> str:
    """Return a Neo4j-safe namespace for index labels and snapshot metadata.

    ``corpus_tag`` remains the public dataset identity used in result paths.
    Paper runs can set ``RAG_INDEX_NAMESPACE`` to build a fresh set of labels
    in the same Neo4j database without deleting or mutating an active index.
    """
    raw = os.environ.get("RAG_INDEX_NAMESPACE", "").strip() or str(corpus_tag or "default")
    token = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        raise ValueError("RAG_INDEX_NAMESPACE must contain at least one letter or digit")
    return token
