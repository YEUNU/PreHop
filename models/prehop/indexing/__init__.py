"""Offline indexing pipeline.

Layer order:
- Fixed-size chunking — chunking.py
- Predictive Knowledge Mapping (Q-/Q+) — knowledge_mapping.py
- Rank-Based HOP Edge Pre-Construction — hop_edges.py
- Neo4j storage (chunks + NEXT edges + index lifecycle) — graph_writer.py
"""
from .chunking import ChunkingMixin
from .graph_writer import GraphWriterMixin
from .hop_edges import HopEdgeMixin
from .knowledge_mapping import KnowledgeMappingMixin


class IndexingPipeline(
    ChunkingMixin,
    KnowledgeMappingMixin,
    HopEdgeMixin,
    GraphWriterMixin,
):
    """Composite mixin exposing the full offline indexing pipeline."""


__all__ = [
    "ChunkingMixin",
    "KnowledgeMappingMixin",
    "HopEdgeMixin",
    "GraphWriterMixin",
    "IndexingPipeline",
]
