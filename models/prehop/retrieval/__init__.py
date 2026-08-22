"""Retrieval layer (paper §3.2.3 Execution).

Three index types per chunk are consulted (Body, Q-, Q+); two stages combine
them via reciprocal rank fusion; cosine similarity orders candidates; pre-built
NEXT/HOP_ANSWER edges drive graph traversal; retrieval uses the original benchmark
query without query-time LLM rewriting.

Modules:
- text_utils.py — normalization and context formatting
- hybrid.py — RRF over {body, q_minus, q_plus} channels
- scoring.py — external-bi-encoder cosine ordering, without a score gate
- traversal.py — deterministic graph_search over offline NEXT/HOP_ANSWER edges
- retrieve.py — two-stage Q-/Q+ retrieve entry point
"""

from .hybrid import HybridSearchMixin
from .retrieve import RetrieveMixin
from .scoring import SimilarityScoringMixin
from .text_utils import TextUtilsMixin
from .traversal import TraversalMixin


class RetrievalPipeline(
    TextUtilsMixin,
    HybridSearchMixin,
    SimilarityScoringMixin,
    TraversalMixin,
    RetrieveMixin,
):
    """Composite mixin exposing the full query-time retrieval pipeline."""


__all__ = [
    "HybridSearchMixin",
    "RetrievalPipeline",
    "RetrieveMixin",
    "SimilarityScoringMixin",
    "TextUtilsMixin",
    "TraversalMixin",
]
