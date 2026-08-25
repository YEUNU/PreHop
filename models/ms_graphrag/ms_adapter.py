"""
[MS GraphRAG] adapter using official graphrag.api Python interface.

Uses graphrag.api.local_search / global_search (graphrag==3.0.1) which performs
the full KG-grounded search: entity embedding retrieval → entity/relationship/
community context + text_units → LLM answer.

Parquet + lancedb artifacts are read from data/ms_graphrag_output/<corpus_tag>/
as built by official_indexer.py. No re-indexing needed.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from models.ms_graphrag.official_indexer import (
    build_config,
    input_dir_for,
    output_dir_for,
    snapshot_metadata_path,
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

    def verify_active_snapshot(self, expected_source_ids: list[str], corpus_manifest: dict | None) -> dict:
        """Read-only proof that the live parquet snapshot matches its marker.

        This is intentionally independent of MS GraphRAG's official search
        API. It reads only ``documents.parquet`` and an unconnected sidecar
        JSON before any query, so it cannot alter official retrieval behavior.
        """
        metadata_file = snapshot_metadata_path(self.corpus_tag)
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"MS GraphRAG active snapshot metadata is unreadable: {metadata_file}") from exc
        if not isinstance(metadata, dict) or metadata.get("status") != "complete":
            raise RuntimeError("MS GraphRAG active snapshot is not marked complete")
        if corpus_manifest is not None:
            if metadata.get("corpus_manifest_fingerprint") != corpus_manifest.get("fingerprint"):
                raise RuntimeError("MS GraphRAG active snapshot fingerprint does not match corpus manifest")
            if metadata.get("corpus_manifest_paragraph_count") != corpus_manifest.get("paragraph_count"):
                raise RuntimeError("MS GraphRAG active snapshot paragraph count does not match corpus manifest")
        documents = self._read_parquet("documents")
        if "title" not in documents.columns:
            raise RuntimeError("MS GraphRAG documents.parquet lacks title column")
        actual_ids = sorted(
            {
                Path(str(title)).stem
                for title in documents["title"].tolist()
                if isinstance(title, str) and title.strip()
            }
        )
        expected = sorted(expected_source_ids)
        if actual_ids != expected:
            raise RuntimeError(
                "MS GraphRAG active documents.parquet does not match prepared corpus "
                f"(expected={len(expected)}, actual={len(actual_ids)})"
            )
        import hashlib

        source_digest = hashlib.sha256("\n".join(actual_ids).encode("utf-8")).hexdigest()
        if metadata.get("source_count") != len(actual_ids) or metadata.get("source_set_sha256") != source_digest:
            raise RuntimeError("MS GraphRAG active metadata does not match its documents.parquet snapshot")
        return metadata

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

    @staticmethod
    def _display_title(source_text: str, source_filename: str) -> str:
        """Return the human-facing document title without losing provenance.

        GraphRAG 3.0.1 assigns an opaque content hash to ``documents.id`` and
        retains the staged input filename in ``documents.title``.  Prepared
        corpora put the original title in their first ``Title:`` header, so
        use that for citations while retaining the filename separately as the
        stable retrieval identity.
        """
        first_line = str(source_text or "").splitlines()[0].strip() if source_text else ""
        if first_line.startswith("Title: "):
            return first_line.removeprefix("Title: ").strip()
        return re.sub(r"\.(pdf|txt|md|json)$", "", str(source_filename or ""), flags=re.IGNORECASE)

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
            # In GraphRAG 3.0.1, document_id is an opaque content hash; the
            # staged filename lives in documents.title.  Never use the opaque
            # id as the source identity, or MuSiQue paragraph provenance
            # becomes dependent on the diagnostic ``doc`` fallback.
            source_filename = str(doc_map.get(doc_id, "") or "")
            source_text = str(row.get("text", "") or "")
            title = self._display_title(source_text, source_filename)
            sources.append(
                {
                    "doc": title,
                    "source": source_filename,
                    "source_filename": source_filename,
                    "document_id": doc_id,
                    "page": 0,
                    "text": source_text,
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
