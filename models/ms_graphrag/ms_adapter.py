"""
[MS GraphRAG] adapter using official graphrag.api Python interface.

Uses graphrag.api.local_search / global_search (the uv-locked GraphRAG release) which performs
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
    _source_titles_sha256,
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
        self._source_id_to_display_title: dict[str, str] | None = None

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
        if metadata.get("strategy") != "ms_graphrag" or metadata.get("corpus_tag") != self.corpus_tag:
            raise RuntimeError("MS GraphRAG active snapshot metadata identifies a different target")
        if metadata.get("snapshot_version") != 2:
            raise RuntimeError("MS GraphRAG active snapshot metadata has an unsupported version")
        if corpus_manifest is not None:
            if metadata.get("corpus_manifest_fingerprint") != corpus_manifest.get("fingerprint"):
                raise RuntimeError("MS GraphRAG active snapshot fingerprint does not match corpus manifest")
            if metadata.get("corpus_manifest_paragraph_count") != corpus_manifest.get("paragraph_count"):
                raise RuntimeError("MS GraphRAG active snapshot paragraph count does not match corpus manifest")
        documents = self._read_parquet("documents")
        if "title" not in documents.columns:
            raise RuntimeError("MS GraphRAG documents.parquet lacks title column")
        titles = documents["title"].tolist()
        if any(not isinstance(title, str) or not title.strip() for title in titles):
            raise RuntimeError("MS GraphRAG documents.parquet contains an empty or non-string title")
        actual_ids = [Path(title).stem for title in titles]
        if len(actual_ids) != len(set(actual_ids)):
            raise RuntimeError("MS GraphRAG documents.parquet contains duplicate source identities")
        actual_ids.sort()
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
        source_titles = metadata.get("source_titles")
        if (
            not isinstance(source_titles, dict)
            or sorted(source_titles) != expected
            or any(not isinstance(title, str) or not title.strip() for title in source_titles.values())
            or metadata.get("source_titles_sha256") != _source_titles_sha256(source_titles)
        ):
            raise RuntimeError("MS GraphRAG active source-title metadata is invalid")
        self._source_id_to_display_title = source_titles
        return metadata

    # ------------------------------------------------------------------ parquet I/O

    def _read_parquet(self, name: str) -> pd.DataFrame:
        path = self.output_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"MS GraphRAG index artifact missing: {path}")
        return pd.read_parquet(path)

    def _ensure_loaded(self) -> None:
        if self._source_id_to_display_title is None:
            metadata_file = snapshot_metadata_path(self.corpus_tag)
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"MS GraphRAG source-title metadata is unreadable: {metadata_file}") from exc
            source_titles = metadata.get("source_titles") if isinstance(metadata, dict) else None
            if (
                not isinstance(source_titles, dict)
                or any(not isinstance(key, str) or not key for key in source_titles)
                or any(not isinstance(title, str) or not title.strip() for title in source_titles.values())
                or metadata.get("source_titles_sha256") != _source_titles_sha256(source_titles)
            ):
                raise RuntimeError("MS GraphRAG source-title metadata is invalid")
            self._source_id_to_display_title = source_titles
        if self._entities is None:
            self._entities = self._read_parquet("entities")
        if self._communities is None:
            self._communities = self._read_parquet("communities")
        if self._community_reports is None:
            self._community_reports = self._read_parquet("community_reports")
        if self._text_units is None:
            self._text_units = self._read_parquet("text_units")
        if self._relationships is None:
            self._relationships = self._read_parquet("relationships")
        if self._documents is None:
            self._documents = self._read_parquet("documents")
        self._short_id_to_doc_id, self._doc_id_to_title = self._build_source_maps(
            self._text_units,
            self._documents,
        )

    @staticmethod
    def _build_source_maps(
        text_units: pd.DataFrame,
        documents: pd.DataFrame,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Build the exact provenance mapping used by GraphRAG's query API.

        ``read_text_units`` assigns ``TextUnit.short_id`` from the DataFrame
        index after ``reset_index``; it does not use ``human_readable_id``.
        Mapping that other column can silently attach a retrieved text unit to
        the wrong document when parquet rows are reordered or carry a custom
        index.
        """
        missing_text_columns = {"document_id"} - set(text_units.columns)
        missing_document_columns = {"id", "title"} - set(documents.columns)
        if missing_text_columns:
            raise RuntimeError(
                f"MS GraphRAG text_units.parquet lacks required columns: {sorted(missing_text_columns)}"
            )
        if missing_document_columns:
            raise RuntimeError(
                f"MS GraphRAG documents.parquet lacks required columns: {sorted(missing_document_columns)}"
            )
        if not text_units.index.is_unique:
            raise RuntimeError("MS GraphRAG text_units.parquet has duplicate row indices")

        doc_id_to_title: dict[str, str] = {}
        source_filenames: set[str] = set()
        for _, row in documents.iterrows():
            doc_id = str(row.get("id", "") or "").strip()
            source_filename = str(row.get("title", "") or "").strip()
            if not doc_id or not source_filename:
                raise RuntimeError("MS GraphRAG documents.parquet contains empty provenance fields")
            if doc_id in doc_id_to_title:
                raise RuntimeError(f"MS GraphRAG documents.parquet has duplicate document id: {doc_id!r}")
            if source_filename in source_filenames:
                raise RuntimeError(
                    f"MS GraphRAG documents.parquet has duplicate source filename: {source_filename!r}"
                )
            doc_id_to_title[doc_id] = source_filename
            source_filenames.add(source_filename)

        short_id_to_doc_id: dict[str, str] = {}
        for row_index, row in text_units.iterrows():
            short_id = str(row_index)
            doc_id = str(row.get("document_id", "") or "").strip()
            if not doc_id:
                raise RuntimeError(f"MS GraphRAG text unit {short_id!r} has no document id")
            if doc_id not in doc_id_to_title:
                raise RuntimeError(
                    f"MS GraphRAG text unit {short_id!r} references unknown document id {doc_id!r}"
                )
            short_id_to_doc_id[short_id] = doc_id
        return short_id_to_doc_id, doc_id_to_title

    # ------------------------------------------------------------------ source extraction

    def _display_title(self, source_text: str, source_filename: str) -> str:
        """Return the human-facing document title without losing provenance.

        GraphRAG assigns an opaque content hash to ``documents.id`` and
        retains the staged input filename in ``documents.title``.  Prepared
        corpora put the original title in their first ``Title:`` header, so
        use that for citations while retaining the filename separately as the
        stable retrieval identity.
        """
        first_line = str(source_text or "").splitlines()[0].strip() if source_text else ""
        if first_line.startswith("Title: "):
            return first_line.removeprefix("Title: ").strip()
        source_id = Path(str(source_filename or "")).stem
        mapped_title = (getattr(self, "_source_id_to_display_title", None) or {}).get(source_id, "")
        if mapped_title:
            return mapped_title
        return re.sub(r"\.(pdf|txt|md|json)$", "", str(source_filename or ""), flags=re.IGNORECASE)

    def _extract_sources(self, context_data: Any) -> list[dict[str, Any]]:
        """Extract source nodes from local_search context_records for doc_match metrics.

        context_records["sources"] is a DataFrame with columns [id, text, ...] where
        id is the official API's text-unit short ID: the row index assigned by
        ``reset_index``, not ``human_readable_id``.
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
            if not unit_id or unit_id not in short_id_map:
                raise RuntimeError(f"MS GraphRAG returned an unknown source id: {unit_id!r}")
            doc_id = short_id_map[unit_id]
            # In GraphRAG, document_id is an opaque content hash; the
            # staged filename lives in documents.title.  Never use the opaque
            # id as the source identity, or MuSiQue paragraph provenance
            # becomes dependent on the diagnostic ``doc`` fallback.
            if doc_id not in doc_map:
                raise RuntimeError(
                    f"MS GraphRAG source {unit_id!r} references unknown document id {doc_id!r}"
                )
            source_filename = str(doc_map[doc_id] or "").strip()
            source_text = str(row.get("text", "") or "")
            if not source_filename or not source_text.strip():
                raise RuntimeError(f"MS GraphRAG source {unit_id!r} has incomplete provenance")
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
        # This benchmark asks entity-grounded questions over source passages,
        # which is the official LocalSearch use case. Do not route between
        # official APIs with adapter-owned keywords: that silently changes the
        # baseline based on surface phrasing and introduces an undocumented
        # query classifier unrelated to the published method.
        logger.info("MS GraphRAG API LocalSearch path")
        return await self.local_search(query)
