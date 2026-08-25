"""Offline indexing pipeline.

Layer order:
- Fixed-size chunking — chunking.py
- Q-/Q+ generation — knowledge_mapping.py
- Strict sparse-text embedding — embedding.py
- Rank-Based HOP Edge Pre-Construction — hop_edges.py
- Neo4j storage (chunks + NEXT edges + index lifecycle) — graph_writer.py
"""

from .chunking import ChunkingMixin
from .embedding import SparseEmbeddingMixin
from .graph_writer import GraphWriterMixin
from .hop_edges import HopEdgeMixin
from .knowledge_mapping import KnowledgeMappingMixin


class IndexingPipeline(
    ChunkingMixin,
    KnowledgeMappingMixin,
    SparseEmbeddingMixin,
    HopEdgeMixin,
    GraphWriterMixin,
):
    """Composite mixin exposing the full offline indexing pipeline."""


__all__ = [
    "ChunkingMixin",
    "GraphWriterMixin",
    "HopEdgeMixin",
    "IndexingPipeline",
    "KnowledgeMappingMixin",
    "SparseEmbeddingMixin",
]
