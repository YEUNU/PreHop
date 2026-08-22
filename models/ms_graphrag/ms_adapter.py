"""
[MS GraphRAG] adapter using official graphrag.api Python interface.

Uses graphrag.api.local_search / global_search (graphrag==3.0.1) which performs
the full KG-grounded search: entity embedding retrieval → entity/relationship/
community context + text_units → LLM answer.

Parquet + lancedb artifacts are read from data/ms_graphrag_output/<corpus_tag>/
as built by official_indexer.py. No re-indexing needed.
"""

import logging
import re
from typing import Any

import pandas as pd

from models.ms_graphrag.official_indexer import (
    build_config,
    input_dir_for,
    output_dir_for,
)

logger = logging.getLogger(__name__)


class MSGraphRAGAdapter:
    def __init__(self, model_id: str = "default", corpus_tag: str = "default"):
        _ = model_id  # kept for the benchmark adapter's shared constructor
        self.corpus_tag = corpus_tag
        self.output_dir = output_dir_for(corpus_tag)

        # GraphRagConfig pointing to external inference + LanceDB at output_dir
        self._config = build_config(corpus_tag, input_dir_for(corpus_tag))

        # Lazy-loaded parquet DataFrames
        self._entities: pd.DataFrame | None = None
        self._communities: pd.DataFrame | None = None
        self._community_reports: pd.DataFrame | None = None
        self._text_units: pd.DataFrame | None = None
        self._relationships: pd.DataFrame | None = None
        self._documents: pd.DataFrame | None = None
        self._doc_id_to_title: dict[str, str] | None = None
        self._short_id_to_doc_id: dict[str, str] | None = None

    # ------------------------------------------------------------------ parquet I/O

    def _read_parquet(self, name: str) -> pd.DataFrame:
        path = self.output_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"MS GraphRAG index artifact missing: {path}")
        return pd.read_parquet(path)

    def _ensure_loaded(self) -> None:
        if self._entities is None:
            self._entities = self._read_parquet("entities")
        if self._communities is None:
            self._communities = self._read_parquet("communities")
        if self._community_reports is None:
            self._community_reports = self._read_parquet("community_reports")
        if self._text_units is None:
            self._text_units = self._read_parquet("text_units")
            # Build short_id → document_id lookup for source extraction
            if not self._text_units.empty and "human_readable_id" in self._text_units.columns:
                self._short_id_to_doc_id = {
                    str(row["human_readable_id"]): str(row.get("document_id", "") or "")
                    for _, row in self._text_units.iterrows()
                }
            else:
                self._short_id_to_doc_id = {}
        if self._relationships is None:
            self._relationships = self._read_parquet("relationships")
        if self._documents is None:
            self._documents = self._read_parquet("documents")
            if not self._documents.empty and "id" in self._documents.columns:
                self._doc_id_to_title = {
                    str(row["id"]): str(row.get("title", "") or "") for _, row in self._documents.iterrows()
                }
            else:
                self._doc_id_to_title = {}

    # ------------------------------------------------------------------ source extraction

    def _extract_sources(self, context_data: Any) -> list[dict[str, Any]]:
        """Extract source nodes from local_search context_records for doc_match metrics.

        context_records["sources"] is a DataFrame with columns [id, text, ...] where
        id == text_unit.short_id (== str(human_readable_id)).
        """
        sources = []
        if not isinstance(context_data, dict):
            raise TypeError(f"MS GraphRAG context_data must be a dict, got {type(context_data).__name__}")
        src_df = context_data.get("sources")
        if src_df is None:
            return sources
        if not hasattr(src_df, "iterrows") or not hasattr(src_df, "empty"):
            raise TypeError("MS GraphRAG context_data['sources'] is not a DataFrame-like object")
        if src_df.empty:
            return sources
        short_id_map = self._short_id_to_doc_id or {}
        doc_map = self._doc_id_to_title or {}
        for _, row in src_df.iterrows():
            unit_id = str(row.get("id", "") or "")
            doc_id = short_id_map.get(unit_id, "")
            title = doc_map.get(doc_id, doc_id)
            title = re.sub(r"\.(pdf|txt|md|json)$", "", title, flags=re.IGNORECASE)
            sources.append(
                {
                    "doc": title,
                    "page": 0,
                    "text": str(row.get("text", "") or ""),
                    "sent_id": 0,
                }
            )
        return sources

    # ------------------------------------------------------------------ search APIs

    async def local_search(self, query: str) -> tuple[str, list, list]:
        import graphrag.api as gapi

        self._ensure_loaded()

        response, context_data = await gapi.local_search(
            config=self._config,
            entities=self._entities,
            communities=self._communities,
            community_reports=self._community_reports,
            text_units=self._text_units,
            relationships=self._relationships,
            covariates=None,
            community_level=2,
            response_type="single concise answer",
            query=query,
        )

        answer = str(response or "").strip()
        if not answer:
            raise ValueError("MS GraphRAG local search returned an empty answer")
        sources = self._extract_sources(context_data)
        trace = [{"step": "ms_local_search_api"}]
        return answer, sources, trace

    async def global_search(self, query: str) -> tuple[str, list, list]:
        import graphrag.api as gapi

        self._ensure_loaded()

        response, context_data = await gapi.global_search(
            config=self._config,
            entities=self._entities,
            communities=self._communities,
            community_reports=self._community_reports,
            community_level=2,
            dynamic_community_selection=False,
            response_type="single concise answer",
            query=query,
        )

        answer = str(response or "").strip()
        if not answer:
            raise ValueError("MS GraphRAG global search returned an empty answer")
        sources = self._extract_sources(context_data)
        trace = [{"step": "ms_global_search_api"}]
        return answer, sources, trace

    # ------------------------------------------------------------------ workflow

    async def run_workflow(self, query: str, history: list[dict] | None = None) -> tuple[str, list, list]:
        _ = history
        abstract_keywords = [
            "overall",
            "summary",
            "main themes",
            "in general",
            "relationship between",
            "high-level",
            "broadly",
            "across documents",
        ]
        is_global = any(kw in query.lower() for kw in abstract_keywords)
        if is_global:
            logger.info("MS GraphRAG API GlobalSearch path")
            return await self.global_search(query)
        logger.info("MS GraphRAG API LocalSearch path")
        return await self.local_search(query)
