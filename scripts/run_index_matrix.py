#!/usr/bin/env python3
"""Run and measure the complete dataset × indexing-strategy matrix.

The runner clears Neo4j at most once, executes distinct corpus/strategy targets
through a bounded parallel queue, samples host/vLLM pressure, and writes
paper-ready JSON/CSV/Markdown artifacts under ``artifacts/indexing/<run-id>``.
It lowers the queue width for not-yet-started targets after sustained memory or
vLLM-waiting pressure; already-running targets are allowed to finish cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import itertools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.io import _write_json

load_dotenv(ROOT / ".env")

DATASETS = {
    "multihoprag": ROOT / "data/multihoprag_corpus",
    "hotpotqa": ROOT / "data/hotpotqa_corpus",
    "musique": ROOT / "data/musique_corpus",
}
EXPECTED_FILE_COUNTS = {"multihoprag": 609, "hotpotqa": 66_581, "musique": 17_629}
STRATEGIES = ("prehop", "naive", "hoprag", "ms_graphrag")
GENERATION_HEAVY_STRATEGIES = frozenset({"prehop", "hoprag", "ms_graphrag"})


@dataclass(frozen=True)
class Target:
    dataset: str
    strategy: str
    corpus_path: Path

    @property
    def key(self) -> str:
        return f"{self.dataset}__{self.strategy}"


def _next_compatible_target_index(
    pending: list[Target],
    active: list[Target],
    max_generation_parallel: int,
) -> int | None:
    """Choose work without overlapping too many generation-heavy targets."""
    active_generation = sum(target.strategy in GENERATION_HEAVY_STRATEGIES for target in active)
    for index, target in enumerate(pending):
        if target.strategy not in GENERATION_HEAVY_STRATEGIES or active_generation < max_generation_parallel:
            return index
    return None


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return token or "default"


def _normalized_endpoint(value: str) -> str:
    return str(value or "").strip().removesuffix("/v1").rstrip("/")


def _fit_inference_capacity(max_parallel: int, max_generation_parallel: int = 1) -> dict[str, Any]:
    """Fit aggregate child-process pressure to the configured inference capacity.

    A shared OpenAI-compatible gateway does not imply shared accelerator
    capacity: generation and embedding model names may route to independent
    servers. ``RAG_INFERENCE_CAPACITY_MODE`` makes that topology explicit.
    """
    max_seqs = int(os.environ.get("VLLM_MAX_NUM_SEQS", "128"))
    generation_max_seqs = int(os.environ.get("VLLM_GENERATION_MAX_NUM_SEQS", str(max_seqs)))
    embedding_max_seqs = int(os.environ.get("VLLM_EMBED_MAX_NUM_SEQS", str(max_seqs)))
    generation = int(os.environ.get("MAX_CONCURRENT_LLM_CALLS", "30"))
    embed_batch = int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE", "32"))
    embed_concurrency = int(os.environ.get("RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS", "2"))
    if (
        min(
            max_parallel,
            max_generation_parallel,
            max_seqs,
            generation_max_seqs,
            embedding_max_seqs,
            generation,
            embed_batch,
            embed_concurrency,
        )
        < 1
    ):
        raise ValueError("Inference capacity and concurrency values must all be positive")
    if max_generation_parallel > max_parallel:
        raise ValueError("Generation-heavy parallelism cannot exceed total parallelism")

    original = {
        "generation_concurrency_per_target": generation,
        "embedding_batch_size": embed_batch,
        "embedding_concurrency_per_target": embed_concurrency,
    }
    generation_url = _normalized_endpoint(os.environ.get("VLLM_URL", ""))
    embedding_url = _normalized_endpoint(os.environ.get("VLLM_EMBED_URL", ""))
    capacity_mode = os.environ.get("RAG_INFERENCE_CAPACITY_MODE", "auto").strip().lower()
    if capacity_mode not in {"auto", "shared", "separate"}:
        raise ValueError("RAG_INFERENCE_CAPACITY_MODE must be auto, shared, or separate")
    same_gateway = bool(generation_url and generation_url == embedding_url)
    shared_capacity = same_gateway if capacity_mode == "auto" else capacity_mode == "shared"

    if shared_capacity:
        embedding_pressure = max_parallel * embed_batch * embed_concurrency
        while embed_concurrency > 1 and embedding_pressure >= max_seqs:
            embed_concurrency -= 1
            embedding_pressure = max_parallel * embed_batch * embed_concurrency
        generation_budget = (max_seqs - embedding_pressure) // max_generation_parallel
        generation = min(generation, generation_budget)
        if generation < 1:
            raise ValueError(
                "Shared inference endpoint cannot fit one generation request plus the embedding batch: "
                f"max_num_seqs={max_seqs}, max_parallel={max_parallel}, "
                f"max_generation_parallel={max_generation_parallel}, embed_batch={embed_batch}"
            )
    else:
        generation = min(generation, generation_max_seqs // max_generation_parallel)
        embed_concurrency = min(
            embed_concurrency,
            embedding_max_seqs // (max_parallel * embed_batch),
        )
        if generation < 1 or embed_concurrency < 1:
            raise ValueError(
                "Separate inference endpoint capacity is too small for the requested matrix width: "
                f"generation_max_num_seqs={generation_max_seqs}, "
                f"embedding_max_num_seqs={embedding_max_seqs}, "
                f"max_parallel={max_parallel}, embed_batch={embed_batch}"
            )

    os.environ["MAX_CONCURRENT_LLM_CALLS"] = str(generation)
    os.environ["RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS"] = str(embed_concurrency)
    generation_pressure = max_generation_parallel * generation
    embedding_pressure = max_parallel * embed_batch * embed_concurrency
    effective_total = (
        generation_pressure + embedding_pressure if shared_capacity else max(generation_pressure, embedding_pressure)
    )
    return {
        "server_max_num_seqs": max_seqs,
        "generation_server_max_num_seqs": generation_max_seqs,
        "embedding_server_max_num_seqs": embedding_max_seqs,
        "max_parallel": max_parallel,
        "max_generation_parallel": max_generation_parallel,
        "capacity_mode": capacity_mode,
        "generation_embedding_share_gateway": same_gateway,
        "generation_embedding_share_endpoint": shared_capacity,
        "requested": original,
        "effective": {
            "generation_concurrency_per_target": generation,
            "embedding_batch_size": embed_batch,
            "embedding_concurrency_per_target": embed_concurrency,
            "generation_capacity_upper_bound": generation_pressure,
            "embedding_capacity_upper_bound": embedding_pressure,
            "aggregate_capacity_upper_bound": effective_total,
        },
        "adjusted": generation != original["generation_concurrency_per_target"]
        or embed_concurrency != original["embedding_concurrency_per_target"],
    }


def _git_revision() -> dict[str, Any]:
    def _run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        revision = _run("rev-parse", "HEAD")
        dirty = bool(_run("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unavailable", "dirty": None}
    return {"revision": revision, "dirty": dirty}


def _source_tree_digest() -> dict[str, Any]:
    """Hash the exact tracked/untracked source snapshot used by this run.

    A commit id plus ``dirty=true`` cannot reproduce an experiment. Git's
    own file list gives us every non-ignored tracked or untracked file while
    excluding generated corpora/artifacts, which are measured separately.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"sha256": "unavailable", "file_count": None}

    paths = sorted(path for path in result.stdout.split(b"\0") if path)
    digest = hashlib.sha256()
    count = 0
    for raw_path in paths:
        relative = os.fsdecode(raw_path)
        path = ROOT / relative
        if not path.is_file():
            continue
        content = path.read_bytes()
        digest.update(len(raw_path).to_bytes(8, "big"))
        digest.update(raw_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        count += 1
    return {"sha256": digest.hexdigest(), "file_count": count}


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _tree_size(path: Path, *, exclude: set[str] | None = None) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    excluded = exclude or set()
    for root, dirs, files in os.walk(path):
        dirs[:] = [name for name in dirs if name not in excluded]
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                continue
    return total


def _artifact_sizes(target: Target) -> dict[str, int]:
    if target.strategy == "prehop":
        root = ROOT / "data/index_cache" / "v1" / target.dataset
    elif target.strategy == "hoprag":
        root = ROOT / "data/hoprag_output" / target.dataset
    elif target.strategy == "ms_graphrag":
        root = ROOT / "data/ms_graphrag_output" / target.dataset
    else:
        root = ROOT / "data/naive_output" / target.dataset
    return {
        "artifact_total_bytes": _tree_size(root),
        "artifact_usable_bytes": _tree_size(root, exclude={"_input", "_cache"}),
        "artifact_cache_bytes": _tree_size(root / "_cache"),
        "artifact_staging_bytes": _tree_size(root / "_input"),
    }


def _one_row(session, query: str) -> dict[str, Any]:
    record = session.run(query).single()
    return record.data() if record is not None else {}


def _normalize_title(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _gold_hop_alignment(session, target: Target, chunk_label: str, document_label: str) -> dict[str, Any]:
    """Measure whether offline HOP topology connects known evidence documents."""
    query_path = ROOT / "data" / f"{target.dataset}_queries.json"
    if not query_path.is_file():
        raise FileNotFoundError(f"Full query file required for gold HOP alignment: {query_path}")
    queries = json.loads(query_path.read_text(encoding="utf-8"))
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"Gold HOP query file is empty or invalid: {query_path}")

    observed_titles = {
        _normalize_title(record["title"])
        for record in session.run(f"MATCH (d:{document_label}) RETURN DISTINCT d.title AS title")
        if _normalize_title(record["title"])
    }
    hop_pairs = {
        tuple(
            sorted(
                (
                    _normalize_title(record["source_title"]),
                    _normalize_title(record["target_title"]),
                )
            )
        )
        for record in session.run(
            f"""
            MATCH (src:{chunk_label})-[:HOP_ANSWER]->(tgt:{chunk_label})
            RETURN DISTINCT src.title AS source_title, tgt.title AS target_title
            """
        )
        if _normalize_title(record["source_title"])
        and _normalize_title(record["target_title"])
        and _normalize_title(record["source_title"]) != _normalize_title(record["target_title"])
    }

    resolved_mentions = 0
    total_mentions = 0
    eligible_queries = 0
    connected_queries = 0
    gold_pairs: set[tuple[str, str]] = set()
    for item in queries:
        docs = list(
            dict.fromkeys(
                normalized
                for normalized in (_normalize_title(value) for value in (item.get("evidence_docs") or []))
                if normalized
            )
        )
        total_mentions += len(docs)
        resolved_mentions += sum(doc in observed_titles for doc in docs)
        resolved_docs = [doc for doc in docs if doc in observed_titles]
        # A partial corpus can make a query look connected by ignoring the
        # missing evidence document. Only fully resolved gold sets are valid
        # for query-level topology coverage.
        if len(resolved_docs) != len(docs):
            continue
        query_pairs = {tuple(sorted(pair)) for pair in itertools.combinations(resolved_docs, 2) if pair[0] != pair[1]}
        if not query_pairs:
            continue
        eligible_queries += 1
        gold_pairs.update(query_pairs)
        if query_pairs & hop_pairs:
            connected_queries += 1

    connected_pairs = gold_pairs & hop_pairs
    return {
        "gold_query_count": len(queries),
        "gold_hop_eligible_query_count": eligible_queries,
        "gold_hop_connected_query_count": connected_queries,
        "gold_query_hop_coverage": connected_queries / eligible_queries if eligible_queries else 0.0,
        "gold_evidence_document_mentions": total_mentions,
        "gold_evidence_document_resolved_mentions": resolved_mentions,
        "gold_evidence_document_resolution_rate": resolved_mentions / total_mentions if total_mentions else 0.0,
        "gold_document_pair_count": len(gold_pairs),
        "gold_document_pair_hop_connected_count": len(connected_pairs),
        "gold_document_pair_hop_coverage": len(connected_pairs) / len(gold_pairs) if gold_pairs else 0.0,
        "distinct_hop_document_pair_count": len(hop_pairs),
        "distinct_hop_pair_gold_alignment_rate": len(connected_pairs) / len(hop_pairs) if hop_pairs else 0.0,
    }


def _graph_stats(target: Target) -> dict[str, Any]:
    """Read strategy-scoped integrity and logical-size statistics."""
    if target.strategy == "ms_graphrag":
        import pyarrow.parquet as pq

        out = ROOT / "data/ms_graphrag_output" / target.dataset
        counts = {}
        for name in (
            "documents",
            "text_units",
            "entities",
            "relationships",
            "communities",
            "community_reports",
        ):
            path = out / f"{name}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Missing MS GraphRAG measurement artifact: {path}")
            counts[f"{name}_count"] = pq.ParquetFile(path).metadata.num_rows
        return counts

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", ""),
    )
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    safe = _safe_token(target.dataset)
    driver = GraphDatabase.driver(uri, auth=auth)
    try:
        with driver.session(database=database) as session:
            if target.strategy == "prehop":
                chunk = f"PR_{safe}_Chunk"
                document = f"PR_{safe}_Document"
                q_minus = f"PR_{safe}_QMinus"
                q_plus = f"PR_{safe}_QPlus"
                stats = _one_row(
                    session,
                    f"""
                    MATCH (c:{chunk})
                    RETURN count(c) AS chunk_count,
                           count(DISTINCT c.id) AS unique_chunk_ids,
                           count(c.embedding) AS body_embedding_count,
                           count(CASE WHEN trim(coalesce(c.text, '')) = '' THEN 1 END) AS empty_text_count,
                           count(CASE WHEN trim(coalesce(c.chunk_summary, '')) = '' THEN 1 END)
                               AS empty_summary_count,
                           count(CASE WHEN size(split(trim(coalesce(c.chunk_summary, '')), ' ')) > 35
                                      THEN 1 END) AS summary_over_35_words,
                           collect(DISTINCT size(c.embedding)) AS embedding_dimensions,
                           sum(size(coalesce(c.title, '')) + size(coalesce(c.text, '')) +
                               size(coalesce(c.chunk_summary, '')) +
                               4 * coalesce(size(c.embedding), 0)) AS chunk_payload_bytes
                    """,
                )
                stats.update(
                    _one_row(
                        session,
                        f"""
                        MATCH (q:{q_minus})
                        RETURN count(q) AS q_minus_count,
                               count(DISTINCT q.id) AS unique_q_minus_ids,
                               count(q.embedding) AS q_minus_embedding_count,
                               count(CASE WHEN trim(coalesce(q.text, '')) = '' THEN 1 END) AS empty_q_minus_count,
                               count(CASE WHEN size(split(trim(coalesce(q.text, '')), ' ')) > 22
                                          THEN 1 END) AS q_minus_over_22_words,
                               collect(DISTINCT size(q.embedding)) AS q_minus_embedding_dimensions,
                               sum(size(coalesce(q.title, '')) + size(coalesce(q.text, '')) +
                                   4 * coalesce(size(q.embedding), 0)) AS q_minus_payload_bytes
                        """,
                    )
                )
                stats.update(
                    _one_row(
                        session,
                        f"""
                        MATCH (q:{q_plus})
                        RETURN count(CASE WHEN EXISTS {{
                            MATCH (q)-[:ANSWERED_BY|SUPPORTED_BY]->()
                        }} THEN 1 END) AS linked_q_plus_count
                        """,
                    )
                )
                stats.update(
                    _one_row(
                        session,
                        f"""
                        MATCH (q:{q_plus})
                        RETURN count(q) AS q_plus_count,
                               count(DISTINCT q.id) AS unique_q_plus_ids,
                               count(q.embedding) AS q_plus_embedding_count,
                               count(q.query_embedding) AS q_plus_query_embedding_count,
                               count(CASE WHEN trim(coalesce(q.text, '')) = '' THEN 1 END) AS empty_q_plus_count,
                               count(CASE WHEN size(split(trim(coalesce(q.text, '')), ' ')) > 22
                                          THEN 1 END) AS q_plus_over_22_words,
                               collect(DISTINCT size(q.embedding)) AS q_plus_embedding_dimensions,
                               collect(DISTINCT size(q.query_embedding)) AS q_plus_query_embedding_dimensions,
                               sum(size(coalesce(q.title, '')) + size(coalesce(q.text, '')) +
                                   4 * coalesce(size(q.embedding), 0) +
                                   4 * coalesce(size(q.query_embedding), 0)) AS q_plus_payload_bytes
                        """,
                    )
                )
                stats.update(_one_row(session, f"MATCH (d:{document}) RETURN count(d) AS document_count"))
                stats.update(
                    _one_row(
                        session,
                        f"""
                        MATCH (d:{document})
                        OPTIONAL MATCH (d)-[:CONTAINS]->(c:{chunk})
                        WITH d, count(c) AS chunks
                        RETURN sum(CASE WHEN chunks > 0 THEN chunks - 1 ELSE 0 END) AS expected_next_edge_count
                        """,
                    )
                )
                stats.update(
                    _one_row(
                        session,
                        f"""
                        MATCH (d:{document})-[:CONTAINS]->(c:{chunk})
                        RETURN count(CASE WHEN d.filename <> c.source THEN 1 END)
                                   AS contains_source_mismatch_count
                        """,
                    )
                )
                stats.update(
                    _one_row(
                        session,
                        f"""
                        MATCH (c:{chunk})
                        RETURN count(CASE WHEN NOT EXISTS {{
                            MATCH (:{document})-[:CONTAINS]->(c)
                        }} THEN 1 END) AS orphan_chunk_count
                        """,
                    )
                )
                stats.update(
                    _one_row(
                        session,
                        f"""
                        MATCH (src:{chunk})-[r:NEXT]->(tgt:{chunk})
                        RETURN count(CASE WHEN src.source <> tgt.source THEN 1 END)
                                   AS next_cross_document_count,
                               count(CASE WHEN tgt.sent_id <> src.sent_id + 1 THEN 1 END)
                                   AS next_nonconsecutive_count
                        """,
                    )
                )
                stats.update(
                    _one_row(
                        session,
                        f"""
                        MATCH (c:{chunk})
                        OPTIONAL MATCH (c)-[out:NEXT]->()
                        WITH c, count(out) AS out_degree
                        OPTIONAL MATCH ()-[incoming:NEXT]->(c)
                        WITH c, out_degree, count(incoming) AS in_degree
                        RETURN count(CASE WHEN out_degree > 1 THEN 1 END)
                                   AS next_outdegree_violation_count,
                               count(CASE WHEN in_degree > 1 THEN 1 END)
                                   AS next_indegree_violation_count
                        """,
                    )
                )
                for rel, key in (
                    ("CONTAINS", "contains_edge_count"),
                    ("NEXT", "next_edge_count"),
                    ("HOP_ANSWER", "hop_edge_count"),
                ):
                    stats.update(
                        _one_row(
                            session,
                            f"MATCH (a:{chunk})-[r:{rel}]->(b:{chunk}) RETURN count(r) AS {key}"
                            if rel != "CONTAINS"
                            else f"MATCH (:{document})-[r:CONTAINS]->(:{chunk}) RETURN count(r) AS {key}",
                        )
                    )
                total = stats.get("chunk_count", 0) or 0
                q_minus_covered = _one_row(
                    session,
                    f"MATCH (c:{chunk})-[:HAS_Q_MINUS]->(:{q_minus}) RETURN count(DISTINCT c) AS count",
                ).get("count", 0)
                q_plus_covered = _one_row(
                    session,
                    f"MATCH (c:{chunk})-[:HAS_Q_PLUS]->(:{q_plus}) RETURN count(DISTINCT c) AS count",
                ).get("count", 0)
                stats["q_minus_coverage"] = q_minus_covered / total if total else 0.0
                stats["q_plus_coverage"] = q_plus_covered / total if total else 0.0
                stats["duplicate_chunk_id_count"] = total - (stats.get("unique_chunk_ids", 0) or 0)
                stats["duplicate_q_minus_id_count"] = (stats.get("q_minus_count", 0) or 0) - (
                    stats.get("unique_q_minus_ids", 0) or 0
                )
                stats["duplicate_q_plus_id_count"] = (stats.get("q_plus_count", 0) or 0) - (
                    stats.get("unique_q_plus_ids", 0) or 0
                )
                stats["avg_hop_out_degree"] = (stats.get("hop_edge_count", 0) or 0) / total if total else 0.0
                stats["logical_payload_bytes_estimate"] = sum(
                    stats.get(key, 0) or 0
                    for key in ("chunk_payload_bytes", "q_minus_payload_bytes", "q_plus_payload_bytes")
                )
                for rel, key, target_label in (
                    ("ANSWERED_BY", "answered_by_edge_count", q_minus),
                    ("SAME_NEED", "same_need_edge_count", q_plus),
                    ("SUPPORTED_BY", "supported_by_edge_count", chunk),
                ):
                    stats.update(
                        _one_row(
                            session,
                            f"MATCH (:{q_plus})-[r:{rel}]->(:{target_label}) RETURN count(r) AS {key}",
                        )
                    )
                stats.update(
                    _one_row(
                        session,
                        f"""
                        MATCH (q:{q_minus})
                        RETURN count(CASE WHEN NOT EXISTS {{
                            MATCH (:{chunk})-[:HAS_Q_MINUS]->(q)
                        }} THEN 1 END) AS orphan_q_minus_count
                        """,
                    )
                )
                stats.update(
                    _one_row(
                        session,
                        f"""
                        MATCH (owner:{chunk})-[:HAS_Q_MINUS]->(q:{q_minus})
                        RETURN count(CASE WHEN owner.source <> q.source OR owner.title <> q.title
                                          THEN 1 END) AS q_minus_owner_mismatch_count
                        """,
                    )
                )
                stats.update(
                    _one_row(
                        session,
                        f"""
                        MATCH (q:{q_plus})
                        RETURN count(CASE WHEN NOT EXISTS {{
                            MATCH (:{chunk})-[:HAS_Q_PLUS]->(q)
                        }} THEN 1 END) AS orphan_q_plus_count
                        """,
                    )
                )
                stats.update(
                    _one_row(
                        session,
                        f"""
                        MATCH (owner:{chunk})-[:HAS_Q_PLUS]->(q:{q_plus})
                        RETURN count(CASE WHEN owner.source <> q.source OR owner.title <> q.title
                                          THEN 1 END) AS q_plus_owner_mismatch_count
                        """,
                    )
                )
                stats.update(
                    _one_row(
                        session,
                        f"""
                        MATCH (src:{chunk})-[r:HOP_ANSWER]->(tgt:{chunk})
                        RETURN count(CASE WHEN src.source = tgt.source THEN 1 END) AS same_document_hop_count,
                               count(CASE WHEN size(coalesce(r.direct_channels, [])) = 0 THEN 1 END)
                                   AS hop_without_direct_signal_count,
                               count(CASE WHEN ('q_minus' IN coalesce(r.direct_channels, []) AND
                                                r.q_minus_score IS NULL) OR
                                               ('body' IN coalesce(r.direct_channels, []) AND
                                                r.body_score IS NULL)
                                          THEN 1 END) AS hop_direct_score_mismatch_count,
                               count(CASE WHEN size(coalesce(r.source_question_ids, [])) = 0 OR
                                               size(coalesce(r.source_question_texts, [])) = 0
                                          THEN 1 END) AS hop_without_bridge_provenance_count
                        """,
                    )
                )
                stats.update(
                    _one_row(
                        session,
                        f"""
                        MATCH (src:{chunk})-[r:HOP_ANSWER]->(:{chunk})
                        WITH src, count(r) AS degree
                        RETURN count(CASE WHEN degree > {int(os.environ.get("RAG_HOP_LINK_LIMIT", "5"))}
                                          THEN 1 END) AS hop_outdegree_violation_count
                        """,
                    )
                )
                samples = session.run(
                    f"""
                    MATCH (c:{chunk})
                    OPTIONAL MATCH (c)-[:HAS_Q_MINUS]->(qm:{q_minus})
                    WITH c, collect(qm.text) AS q_minus_texts
                    OPTIONAL MATCH (c)-[:HAS_Q_PLUS]->(qp:{q_plus})
                    RETURN c.source AS source, c.title AS title, c.page AS page,
                           c.sent_id AS sent_id, c.text AS text,
                           q_minus_texts AS q_minus,
                           collect(qp.text) AS q_plus,
                           c.chunk_summary AS chunk_summary
                    ORDER BY c.source, c.sent_id
                    LIMIT 8
                    """
                )
                stats["inspection_samples"] = [record.data() for record in samples]
                hop_samples = session.run(
                    f"""
                    MATCH (src:{chunk})-[r:HOP_ANSWER]->(tgt:{chunk})
                    RETURN src.source AS source, src.text AS source_text,
                           tgt.source AS target_source, tgt.text AS target_text,
                           r.score AS score, r.direct_channels AS direct_channels,
                           r.same_need_score AS same_need_score
                    ORDER BY r.score DESC
                    LIMIT 8
                    """
                )
                stats["hop_inspection_samples"] = [record.data() for record in hop_samples]
                stats.update(_gold_hop_alignment(session, target, chunk, document))
                return stats

            if target.strategy == "naive":
                chunk = f"NA_{safe}_Chunk"
                stats = _one_row(
                    session,
                    f"""
                    MATCH (c:{chunk})
                    RETURN count(c) AS chunk_count,
                           count(DISTINCT c.source) AS document_count,
                           count(DISTINCT c.id) AS unique_chunk_ids,
                           count(c.embedding) AS body_embedding_count,
                           count(CASE WHEN trim(coalesce(c.text, '')) = '' THEN 1 END) AS empty_text_count,
                           collect(DISTINCT size(c.embedding)) AS embedding_dimensions,
                           sum(size(coalesce(c.title, '')) + size(coalesce(c.text, '')) +
                               4 * coalesce(size(c.embedding), 0))
                               AS logical_payload_bytes_estimate
                    """,
                )
                stats["duplicate_chunk_id_count"] = (stats.get("chunk_count", 0) or 0) - (
                    stats.get("unique_chunk_ids", 0) or 0
                )
                return stats

            label = f"HO_{safe}"
            rel = f"HO_{safe}_p2a"
            stats = _one_row(
                session,
                f"""
                MATCH (n:{label})
                RETURN count(n) AS node_count,
                       count(DISTINCT n.source) AS document_count,
                       count(n.embed) AS body_embedding_count,
                       count(CASE WHEN trim(coalesce(n.text, '')) = '' THEN 1 END) AS empty_text_count,
                       collect(DISTINCT size(n.embed)) AS embedding_dimensions,
                       sum(size(coalesce(n.text, '')) + 4 * coalesce(size(n.embed), 0))
                           AS logical_payload_bytes_estimate
                """,
            )
            stats.update(_one_row(session, f"MATCH (:{label})-[r:{rel}]->(:{label}) RETURN count(r) AS edge_count"))
            node_count = stats.get("node_count", 0) or 0
            stats["avg_out_degree"] = (stats.get("edge_count", 0) or 0) / node_count if node_count else 0.0
            return stats
    finally:
        driver.close()


def _validate_target_stats(target: Target, stats: dict[str, Any], input_file_count: int) -> None:
    """Fail paper measurements on incomplete, duplicated, empty, or wrong-dimension indices."""
    expected_dim = int(os.environ.get("NEO4J_VECTOR_DIMENSIONS", "1024"))
    dimensions = sorted(int(value) for value in (stats.get("embedding_dimensions") or []) if value is not None)
    if dimensions and dimensions != [expected_dim]:
        raise ValueError(f"embedding dimensions are {dimensions}, expected [{expected_dim}]")
    if (stats.get("empty_text_count", 0) or 0) != 0:
        raise ValueError(f"empty indexed texts: {stats['empty_text_count']}")

    if target.strategy == "prehop":
        if stats.get("document_count") != input_file_count:
            raise ValueError(f"document count {stats.get('document_count')} != input files {input_file_count}")
        if not (stats.get("chunk_count", 0) or 0):
            raise ValueError("Prehop index contains no chunks")
        if stats.get("contains_edge_count") != stats.get("chunk_count"):
            raise ValueError("Prehop CONTAINS edge count does not match chunk count")
        for key in ("orphan_chunk_count", "duplicate_chunk_id_count"):
            if (stats.get(key, 0) or 0) != 0:
                raise ValueError(f"Prehop integrity violation: {key}={stats[key]}")
        if (stats.get("body_embedding_count", 0) or 0) != stats.get("chunk_count"):
            raise ValueError("Prehop body embedding coverage is incomplete")
        for count_key, embedding_key in (
            ("q_minus_count", "q_minus_embedding_count"),
            ("q_plus_count", "q_plus_embedding_count"),
            ("q_plus_count", "q_plus_query_embedding_count"),
        ):
            if (stats.get(embedding_key, 0) or 0) != (stats.get(count_key, 0) or 0):
                raise ValueError(f"Prehop question embedding coverage is incomplete: {embedding_key}")
        for dimension_key in (
            "q_minus_embedding_dimensions",
            "q_plus_embedding_dimensions",
            "q_plus_query_embedding_dimensions",
        ):
            question_dimensions = sorted(int(value) for value in (stats.get(dimension_key) or []) if value is not None)
            if question_dimensions and question_dimensions != [expected_dim]:
                raise ValueError(f"Prehop {dimension_key}={question_dimensions}, expected [{expected_dim}]")
        for key in (
            "empty_q_minus_count",
            "empty_q_plus_count",
            "duplicate_q_minus_id_count",
            "duplicate_q_plus_id_count",
            "orphan_q_minus_count",
            "orphan_q_plus_count",
            "empty_summary_count",
            "contains_source_mismatch_count",
            "q_minus_owner_mismatch_count",
            "q_plus_owner_mismatch_count",
            "next_cross_document_count",
            "next_nonconsecutive_count",
            "next_outdegree_violation_count",
            "next_indegree_violation_count",
            "same_document_hop_count",
            "hop_without_direct_signal_count",
            "hop_direct_score_mismatch_count",
            "hop_without_bridge_provenance_count",
            "hop_outdegree_violation_count",
        ):
            if (stats.get(key, 0) or 0) != 0:
                raise ValueError(f"Prehop integrity violation: {key}={stats[key]}")
        if stats.get("next_edge_count") != stats.get("expected_next_edge_count"):
            raise ValueError("Prehop NEXT edge count does not match the ordered per-document chunk topology")
        if input_file_count > 1 and (stats.get("q_plus_count", 0) or 0) and not (stats.get("hop_edge_count", 0) or 0):
            raise ValueError("Prehop generated Q+ questions but no cross-document HOP_ANSWER edges")
    elif target.strategy == "naive":
        if stats.get("document_count") != input_file_count:
            raise ValueError(f"Naive document count {stats.get('document_count')} != input files {input_file_count}")
        if not (stats.get("chunk_count", 0) or 0):
            raise ValueError("Naive index contains no chunks")
        if (stats.get("duplicate_chunk_id_count", 0) or 0) != 0:
            raise ValueError("Naive index contains duplicate chunk ids")
        if (stats.get("body_embedding_count", 0) or 0) != stats.get("chunk_count"):
            raise ValueError("Naive body embedding coverage is incomplete")
    elif target.strategy == "hoprag":
        if stats.get("document_count") != input_file_count:
            raise ValueError(f"HopRAG document count {stats.get('document_count')} != input files {input_file_count}")
        if not (stats.get("node_count", 0) or 0):
            raise ValueError("HopRAG index contains no nodes")
        if (stats.get("body_embedding_count", 0) or 0) != stats.get("node_count"):
            raise ValueError("HopRAG embedding coverage is incomplete")
    else:
        if stats.get("documents_count") != input_file_count:
            raise ValueError(
                f"MS GraphRAG document count {stats.get('documents_count')} != input files {input_file_count}"
            )
        if not (stats.get("text_units_count", 0) or 0):
            raise ValueError("MS GraphRAG index contains no text units")


def _host_memory() -> tuple[int, int, float]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    used_ratio = (total - available) / total if total else 0.0
    return total, available, used_ratio


def _endpoint_pressure(configured: str) -> dict[str, int]:
    if not configured:
        return {"running": 0, "waiting": 0}
    base = configured.removesuffix("/v1").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=3) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError):
        return {"running": 0, "waiting": 0}

    def _metric(name: str) -> int:
        values = re.findall(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$", body, re.MULTILINE)
        return int(sum(float(value) for value in values))

    return {
        "running": _metric("vllm:num_requests_running"),
        "waiting": _metric("vllm:num_requests_waiting"),
    }


def _vllm_pressure() -> dict[str, int]:
    """Read generation and embedding servers independently.

    The two model classes intentionally live on different external servers;
    observing only generation could miss an embedding queue bottleneck and
    keep launching targets after the embedding server is saturated.
    """
    generation = _endpoint_pressure(os.environ.get("VLLM_URL", "").strip())
    embedding_url = os.environ.get("VLLM_EMBED_URL", "").strip()
    if embedding_url == os.environ.get("VLLM_URL", "").strip():
        embedding = dict(generation)
        total_running = generation["running"]
        total_waiting = generation["waiting"]
    else:
        embedding = _endpoint_pressure(embedding_url)
        total_running = generation["running"] + embedding["running"]
        total_waiting = generation["waiting"] + embedding["waiting"]
    return {
        "generation_running": generation["running"],
        "generation_waiting": generation["waiting"],
        "embedding_running": embedding["running"],
        "embedding_waiting": embedding["waiting"],
        "running": total_running,
        "waiting": total_waiting,
    }


async def _resource_sampler(run_dir: Path, state: dict[str, Any], stop: asyncio.Event, interval: float) -> None:
    pressure_streak = 0
    max_waiting = max(1, int(os.environ.get("RAG_MATRIX_MAX_VLLM_WAITING", "32")))
    while not stop.is_set():
        total, available, host_ratio = await asyncio.to_thread(_host_memory)
        vllm = await asyncio.to_thread(_vllm_pressure)
        sample = {
            "timestamp": time.time(),
            "active_targets": sorted(state["active"]),
            "parallel_limit": state["parallel_limit"],
            "host_memory_total_bytes": total,
            "host_memory_available_bytes": available,
            "host_memory_used_ratio": host_ratio,
            "vllm_running": vllm["running"],
            "vllm_waiting": vllm["waiting"],
            "generation_vllm_running": vllm["generation_running"],
            "generation_vllm_waiting": vllm["generation_waiting"],
            "embedding_vllm_running": vllm["embedding_running"],
            "embedding_vllm_waiting": vllm["embedding_waiting"],
        }
        state["samples"].append(sample)
        pressured = host_ratio >= 0.90 or vllm["waiting"] > max_waiting
        pressure_streak = pressure_streak + 1 if pressured else 0
        if pressure_streak >= 3 and state["parallel_limit"] > 1:
            old = state["parallel_limit"]
            state["parallel_limit"] -= 1
            event = {
                "timestamp": time.time(),
                "event": "parallelism_reduced",
                "old": old,
                "new": state["parallel_limit"],
                "reason": {
                    "host_memory_used_ratio": host_ratio,
                    "vllm_waiting": vllm["waiting"],
                },
            }
            state["events"].append(event)
            print(f"[matrix] pressure confirmed; parallelism {old} -> {state['parallel_limit']}", flush=True)
            pressure_streak = 0
        await asyncio.to_thread(_append_jsonl, run_dir / "resource_samples.jsonl", sample)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            continue


def _parse_time_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    out: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ": " not in line:
            continue
        key, value = line.strip().split(": ", 1)
        if key == "Maximum resident set size (kbytes)":
            out["max_rss_bytes"] = int(value) * 1024
        elif key == "User time (seconds)":
            out["user_cpu_seconds"] = float(value)
        elif key == "System time (seconds)":
            out["system_cpu_seconds"] = float(value)
        elif key == "Percent of CPU this job got":
            out["cpu_percent_time_average"] = float(value.rstrip("%"))
    return out


async def _run_target(
    target: Target,
    run_dir: Path,
    matrix_id: str,
    save_intermediate: bool,
    attempt: int,
    concurrency_divisor: int,
) -> dict[str, Any]:
    attempt_suffix = f"__attempt_{attempt}" if attempt > 1 else ""
    child_run_id = f"{matrix_id}_{target.key}{attempt_suffix}"
    log_path = run_dir / "logs" / f"{target.key}{attempt_suffix}.log"
    time_path = run_dir / "time" / f"{target.key}{attempt_suffix}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    time_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(time_path),
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "main.py"),
        "--mode",
        "index",
        "--strategy",
        target.strategy,
        "--dataset",
        str(target.corpus_path),
        "--corpus-tag",
        target.dataset,
    ]
    if save_intermediate and target.strategy == "prehop":
        command.append("--save-intermediate")
    env = os.environ.copy()
    env["RAG_RUN_ID"] = child_run_id
    env["RAG_CHUNK_CACHE"] = "off"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    base_file_workers = max(1, int(os.environ.get("RAG_MAX_PARALLEL_FILES", "16")))
    base_hop_doc_workers = max(1, int(os.environ.get("RAG_HOP_DOC_WORKERS", "10")))
    base_ms_requests = max(1, int(os.environ.get("RAG_MS_CONCURRENT_REQUESTS", "32")))
    base_llm_calls = max(1, int(os.environ.get("MAX_CONCURRENT_LLM_CALLS", "30")))
    env["RAG_MAX_PARALLEL_FILES"] = str(max(1, base_file_workers // concurrency_divisor))
    env.setdefault("RAG_FILE_SCHEDULE_BATCH", "32")
    env.setdefault("RAG_PARSE_WORKERS", "8")
    env["RAG_HOP_DOC_WORKERS"] = str(max(1, base_hop_doc_workers // concurrency_divisor))
    env.setdefault("RAG_HOP_MAX_THREADS", "4")
    env["RAG_MS_CONCURRENT_REQUESTS"] = str(max(1, base_ms_requests // concurrency_divisor))
    env["MAX_CONCURRENT_LLM_CALLS"] = str(max(1, base_llm_calls // concurrency_divisor))
    before = await asyncio.to_thread(_artifact_sizes, target)
    started = time.time()
    print(f"[matrix] start {target.key} attempt={attempt}", flush=True)
    log_handle = await asyncio.to_thread(log_path.open, "wb")
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        return_code = await process.wait()
    except asyncio.CancelledError:
        if process is not None and process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
        raise
    finally:
        await asyncio.to_thread(log_handle.close)
    finished = time.time()
    after = await asyncio.to_thread(_artifact_sizes, target)
    result: dict[str, Any] = {
        "target": target.key,
        "dataset": target.dataset,
        "strategy": target.strategy,
        "corpus_path": str(target.corpus_path),
        "input_file_count": len(list(target.corpus_path.glob("*.txt"))) + len(list(target.corpus_path.glob("*.md"))),
        "input_bytes": await asyncio.to_thread(_tree_size, target.corpus_path),
        "run_id": child_run_id,
        "attempt": attempt,
        "concurrency_divisor": concurrency_divisor,
        "effective_concurrency": {
            "prehop_file_workers": int(env["RAG_MAX_PARALLEL_FILES"]),
            "hoprag_doc_workers": int(env["RAG_HOP_DOC_WORKERS"]),
            "hoprag_chunk_threads": int(env["RAG_HOP_MAX_THREADS"]),
            "ms_requests": int(env["RAG_MS_CONCURRENT_REQUESTS"]),
            "max_concurrent_llm_calls": int(env["MAX_CONCURRENT_LLM_CALLS"]),
        },
        "started_at": started,
        "finished_at": finished,
        "elapsed_seconds": finished - started,
        "return_code": return_code,
        "status": "complete" if return_code == 0 else "failed",
        "log_path": str(log_path),
        "time_path": str(time_path),
        "artifact_before": before,
        "artifact_after": after,
        "artifact_growth_bytes": after["artifact_total_bytes"] - before["artifact_total_bytes"],
        **(await asyncio.to_thread(_parse_time_file, time_path)),
    }
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    result["hoprag_skipped_question_groups"] = log_text.count("using the official empty-list skip")
    failure_path = ROOT / "data/index_failures" / f"{target.strategy}_{target.dataset}_{child_run_id}.json"
    if failure_path.is_file():
        failure_payload = json.loads(failure_path.read_text(encoding="utf-8"))
        result["failure_count"] = int(failure_payload.get("failed", 0))
    else:
        result["failure_count"] = 0
    runtime_stats_path = ROOT / "data/index_stats" / f"{target.strategy}_{target.dataset}_{child_run_id}.json"
    if runtime_stats_path.is_file():
        runtime_payload = json.loads(runtime_stats_path.read_text(encoding="utf-8"))
        result["runtime_stage_timing_seconds"] = runtime_payload.get("timing_seconds", {})
    if return_code == 0:
        try:
            result["index_stats"] = await asyncio.to_thread(_graph_stats, target)
            _validate_target_stats(target, result["index_stats"], result["input_file_count"])
        except Exception as exc:  # noqa: BLE001 - measurement failure invalidates this target result
            result["status"] = "measurement_failed"
            result["measurement_error"] = f"{type(exc).__name__}: {exc}"
    result["files_per_second"] = (
        result["input_file_count"] / result["elapsed_seconds"] if result["elapsed_seconds"] else 0.0
    )
    print(
        f"[matrix] finish {target.key} attempt={attempt}: {result['status']} ({result['elapsed_seconds']:.1f}s)",
        flush=True,
    )
    return result


def _add_target_pressure_peaks(results: list[dict[str, Any]], samples: list[dict[str, Any]]) -> None:
    for result in results:
        relevant = [sample for sample in samples if result["target"] in sample.get("active_targets", [])]
        result["peak_vllm_waiting"] = max((sample.get("vllm_waiting", 0) for sample in relevant), default=0)
        result["peak_host_memory_used_ratio"] = max(
            (sample.get("host_memory_used_ratio", 0.0) for sample in relevant), default=0.0
        )


def _primary_counts(result: dict[str, Any]) -> tuple[int, int]:
    stats = result.get("index_stats", {})
    nodes = stats.get("chunk_count", stats.get("node_count", stats.get("text_units_count", 0))) or 0
    edges = stats.get("hop_edge_count", stats.get("edge_count", stats.get("relationships_count", 0))) or 0
    return int(nodes), int(edges)


def _write_tables(run_dir: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "strategy",
        "status",
        "input_file_count",
        "elapsed_seconds",
        "files_per_second",
        "max_rss_bytes",
        "peak_host_memory_used_ratio",
        "peak_vllm_waiting",
        "failure_count",
        "hoprag_skipped_question_groups",
        "node_or_chunk_count",
        "edge_count",
        "artifact_usable_bytes",
        "artifact_cache_bytes",
        "logical_payload_bytes_estimate",
        "q_minus_coverage",
        "q_plus_coverage",
        "gold_query_hop_coverage",
        "gold_document_pair_hop_coverage",
    ]
    rows = []
    for result in results:
        nodes, edges = _primary_counts(result)
        stats = result.get("index_stats", {})
        row = {
            **{key: result.get(key, "") for key in fields},
            "node_or_chunk_count": nodes,
            "edge_count": edges,
            "artifact_usable_bytes": result.get("artifact_after", {}).get("artifact_usable_bytes", 0),
            "artifact_cache_bytes": result.get("artifact_after", {}).get("artifact_cache_bytes", 0),
            "logical_payload_bytes_estimate": stats.get("logical_payload_bytes_estimate", 0),
            "q_minus_coverage": stats.get("q_minus_coverage", ""),
            "q_plus_coverage": stats.get("q_plus_coverage", ""),
            "gold_query_hop_coverage": stats.get("gold_query_hop_coverage", ""),
            "gold_document_pair_hop_coverage": stats.get("gold_document_pair_hop_coverage", ""),
        }
        rows.append(row)
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(run_dir / "summary.csv", csv_buffer.getvalue())

    lines = [
        "| Dataset | Strategy | Status | Files | Time (min) | Nodes/chunks | Edges | Gold-query HOP (%) | Max RSS (GiB) | Usable artifacts (MiB) | Logical graph (MiB) | Failures |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result, row in zip(results, rows):
        lines.append(
            "| {dataset} | {strategy} | {status} | {files} | {minutes:.2f} | {nodes:,} | {edges:,} | {gold_hop} | "
            "{rss:.2f} | {usable:.2f} | {logical:.2f} | {failures} |".format(
                dataset=result["dataset"],
                strategy=result["strategy"],
                status=result["status"],
                files=result["input_file_count"],
                minutes=result["elapsed_seconds"] / 60,
                nodes=row["node_or_chunk_count"],
                edges=row["edge_count"],
                gold_hop=(
                    f"{100 * row['gold_query_hop_coverage']:.2f}"
                    if isinstance(row["gold_query_hop_coverage"], (int, float))
                    else "—"
                ),
                rss=(result.get("max_rss_bytes", 0) or 0) / 2**30,
                usable=(row["artifact_usable_bytes"] or 0) / 2**20,
                logical=(row["logical_payload_bytes_estimate"] or 0) / 2**20,
                failures=result.get("failure_count", 0),
            )
        )
    lines.extend(
        [
            "",
            "`Logical graph` is a reproducible payload estimate (UTF-8-like text characters plus 4 bytes per float vector element), not Neo4j store-file size.",
            "`Usable artifacts` excludes staging inputs and rebuild caches; caches are reported separately in `summary.csv`.",
        ]
    )
    _atomic_write_text(run_dir / "paper_table.md", "\n".join(lines) + "\n")


async def _clear_graph(run_dir: Path) -> None:
    command = [str(ROOT / ".venv/bin/python"), str(ROOT / "main.py"), "--mode", "clear_graph"]
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await process.communicate()
    _atomic_write_text(run_dir / "clear_graph.log", output.decode("utf-8", errors="replace"))
    if process.returncode != 0:
        raise RuntimeError(f"Neo4j clear failed; see {run_dir / 'clear_graph.log'}")


def _clear_cold_artifacts(targets: list[Target], run_dir: Path) -> None:
    """Remove generated state that could make a selected target warm or resumed."""
    removed: list[str] = []
    for dataset in {target.dataset for target in targets}:
        for path in (
            ROOT / "data/index_cache/v1" / dataset,
            ROOT / "data/hoprag_output" / dataset,
            ROOT / "data/ms_graphrag_output" / dataset,
            ROOT / "data/naive_output" / dataset,
        ):
            if path.exists():
                shutil.rmtree(path)
                removed.append(str(path))
    for path in (
        ROOT / "data/debug",
        ROOT / "data/index_failures",
        ROOT / "data/index_stats",
        ROOT / "data/index_locks",
        ROOT / "logs",
    ):
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    _write_json(run_dir / "cold_artifact_cleanup.json", {"removed": removed})


async def _verify_external_inference(run_dir: Path) -> None:
    """Fail before graph deletion if the configured remote models disagree."""
    from core.config import RAGConfig
    from core.vllm_client import VLLMClient

    RAGConfig.validate()
    client = VLLMClient()
    try:
        embedding = await client.get_embedding("prehop indexing configuration probe")
        if len(embedding) != RAGConfig.EMBEDDING_DIMENSIONS:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"endpoint returned {len(embedding)}, configured {RAGConfig.EMBEDDING_DIMENSIONS}"
            )
        response = await client.generate_response(
            [{"role": "user", "content": "Reply exactly OK"}],
            temperature=0.0,
            max_tokens=8,
        )
        if not str(response or "").strip():
            raise ValueError("Generation endpoint returned an empty preflight response")
        _write_json(
            run_dir / "inference_preflight.json",
            {
                "generation_url": RAGConfig.VLLM_URL,
                "embedding_url": RAGConfig.VLLM_EMBED_URL,
                "generation_model": RAGConfig.DEFAULT_MODEL,
                "embedding_model": RAGConfig.EMBEDDING_MODEL,
                "embedding_dimensions_configured": RAGConfig.EMBEDDING_DIMENSIONS,
                "embedding_dimensions_observed": len(embedding),
                "generation_nonempty": True,
            },
        )
    finally:
        await VLLMClient.global_close()


async def _main(args: argparse.Namespace) -> int:
    matrix_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = (ROOT / "artifacts/indexing" / _safe_token(matrix_id)).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    selected_datasets = list(DATASETS) if args.datasets == ["all"] else args.datasets
    selected_strategies = list(STRATEGIES) if args.strategies == ["all"] else args.strategies
    capacity_plan = _fit_inference_capacity(args.max_parallel, args.max_generation_parallel)
    targets = [
        Target(dataset, strategy, DATASETS[dataset])
        for dataset in selected_datasets
        for strategy in selected_strategies
    ]
    missing = [str(target.corpus_path) for target in targets if not target.corpus_path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing corpus directories: {missing}")
    for dataset in selected_datasets:
        observed = len(list(DATASETS[dataset].glob("*.txt"))) + len(list(DATASETS[dataset].glob("*.md")))
        expected = EXPECTED_FILE_COUNTS[dataset]
        if observed != expected:
            raise ValueError(f"Corpus {dataset} has {observed} files; expected full dataset count {expected}")
    manifest = {
        "run_id": matrix_id,
        "created_at": time.time(),
        "cold_graph": bool(args.clear_graph),
        "cold_artifacts": bool(args.clear_graph),
        "initial_parallelism": args.max_parallel,
        "max_generation_parallelism": args.max_generation_parallel,
        "target_attempts": args.target_attempts,
        "datasets": selected_datasets,
        "strategies": selected_strategies,
        "code": {**_git_revision(), "source_tree": _source_tree_digest()},
        "dataset_inputs": {
            dataset: {
                "file_count": len(list(DATASETS[dataset].glob("*.txt"))) + len(list(DATASETS[dataset].glob("*.md"))),
                "bytes": _tree_size(DATASETS[dataset]),
            }
            for dataset in selected_datasets
        },
        "method_settings": {
            "prehop_naive_chunk_sentences": int(os.environ.get("RAG_CHUNK_SENTENCES", "6")),
            "prehop_naive_min_chunk_sentences": int(os.environ.get("RAG_MIN_CHUNK_SENTENCES", "2")),
            "prehop_naive_top_k": int(os.environ.get("RAG_DEFAULT_TOP_K", "12")),
            "hoprag_official_top_k": 20,
            "ms_graphrag_context_budget": "official package configuration",
            "hop_link_limit": int(os.environ.get("RAG_HOP_LINK_LIMIT", "5")),
            "hop_candidate_limit": int(os.environ.get("RAG_HOP_CANDIDATE_LIMIT", "15")),
            "hop_ann_pool": int(os.environ.get("RAG_HOP_ANN_POOL", "50")),
            "hop_gather_wave": int(os.environ.get("RAG_HOP_GATHER_WAVE", "64")),
            "hop_channel_concurrency": int(os.environ.get("RAG_HOP_CHANNEL_CONCURRENCY", "2")),
            "hop_same_need_weight": float(os.environ.get("RAG_HOP_SAME_NEED_WEIGHT", "0.5")),
            "q_minus_enabled": os.environ.get("RAG_ABLATION_Q_MINUS", "True").lower() == "true",
            "q_plus_enabled": os.environ.get("RAG_ABLATION_Q_PLUS", "True").lower() == "true",
            "prehop_file_workers": int(os.environ.get("RAG_MAX_PARALLEL_FILES", "16")),
            "hoprag_doc_workers": int(os.environ.get("RAG_HOP_DOC_WORKERS", "10")),
            "hoprag_chunk_threads_per_document": int(os.environ.get("RAG_HOP_MAX_THREADS", "4")),
            "hoprag_question_validation_retries": int(os.environ.get("RAG_HOP_QUESTION_RETRIES", "3")),
            "ms_concurrent_requests": int(os.environ.get("RAG_MS_CONCURRENT_REQUESTS", "32")),
        },
        "inference_capacity_plan": capacity_plan,
        "inference": {
            "generation_url": os.environ.get("VLLM_URL", ""),
            "embedding_url": os.environ.get("VLLM_EMBED_URL", ""),
            "generation_model": os.environ.get("VLLM_SERVED_MODEL_NAME", "generation-model"),
            "embedding_model": os.environ.get("VLLM_SERVED_EMBED_MODEL_NAME", "embedding-model"),
            "embedding_dimensions": int(os.environ.get("NEO4J_VECTOR_DIMENSIONS", "1024")),
            "server_max_num_seqs": int(os.environ.get("VLLM_MAX_NUM_SEQS", "128")),
            "client_max_concurrent_llm_calls_per_target": int(os.environ.get("MAX_CONCURRENT_LLM_CALLS", "30")),
            "embedding_batch_size": int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE", "32")),
            "max_concurrent_embedding_requests_per_target": int(
                os.environ.get("RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS", "2")
            ),
            "compute_location": "external (accelerator telemetry unavailable to this runner)",
        },
        "targets": [target.key for target in targets],
        "environment_overrides": {
            key: os.environ.get(key)
            for key in (
                "RAG_MAX_PARALLEL_FILES",
                "RAG_FILE_SCHEDULE_BATCH",
                "RAG_PARSE_WORKERS",
                "RAG_HOP_DOC_WORKERS",
                "RAG_HOP_MAX_THREADS",
                "RAG_HOP_QUESTION_RETRIES",
                "RAG_MS_CONCURRENT_REQUESTS",
                "RAG_INFERENCE_CAPACITY_MODE",
                "VLLM_MAX_NUM_SEQS",
                "VLLM_GENERATION_MAX_NUM_SEQS",
                "VLLM_EMBED_MAX_NUM_SEQS",
                "MAX_CONCURRENT_LLM_CALLS",
            )
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    if shutil.which("/usr/bin/time") is None:
        raise FileNotFoundError("GNU /usr/bin/time is required for per-target RSS/CPU measurement")
    await _verify_external_inference(run_dir)
    if args.clear_graph:
        await _clear_graph(run_dir)
        await asyncio.to_thread(_clear_cold_artifacts, targets, run_dir)

    state: dict[str, Any] = {
        "parallel_limit": args.max_parallel,
        "active": set(),
        "samples": [],
        "events": [],
    }
    stop = asyncio.Event()
    sampler = asyncio.create_task(_resource_sampler(run_dir, state, stop, args.sample_seconds))
    pending = list(targets)
    running: dict[asyncio.Task, Target] = {}
    attempts_by_target: dict[str, int] = {target.key: 0 for target in targets}
    concurrency_divisors: dict[str, int] = {target.key: 1 for target in targets}
    attempt_history: dict[str, list[dict[str, Any]]] = {target.key: [] for target in targets}
    results: list[dict[str, Any]] = []
    try:
        while pending or running:
            while pending and len(running) < state["parallel_limit"]:
                pending_index = _next_compatible_target_index(
                    pending,
                    list(running.values()),
                    args.max_generation_parallel,
                )
                if pending_index is None:
                    break
                target = pending.pop(pending_index)
                attempts_by_target[target.key] += 1
                attempt = attempts_by_target[target.key]
                state["active"].add(target.key)
                task = asyncio.create_task(
                    _run_target(
                        target,
                        run_dir,
                        matrix_id,
                        args.save_prehop_intermediate,
                        attempt,
                        concurrency_divisors[target.key],
                    )
                )
                running[task] = target
            if not running:
                continue
            done, _ = await asyncio.wait(running, timeout=5, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                target = running.pop(task)
                state["active"].discard(target.key)
                result = await task
                attempt_history[target.key].append(result)
                if result["status"] == "failed":
                    log_text = Path(result["log_path"]).read_text(encoding="utf-8", errors="replace").lower()
                    resource_markers = ("out of memory", "cuda oom", "killed", "429", "rate limit")
                    if any(marker in log_text for marker in resource_markers):
                        old = state["parallel_limit"]
                        state["parallel_limit"] = max(1, old - 1)
                        concurrency_divisors[target.key] *= 2
                        state["events"].append(
                            {
                                "timestamp": time.time(),
                                "event": "parallelism_reduced_after_resource_failure",
                                "old": old,
                                "new": state["parallel_limit"],
                                "target": target.key,
                                "attempt": result["attempt"],
                                "next_concurrency_divisor": concurrency_divisors[target.key],
                            }
                        )
                if result["status"] != "complete" and result["attempt"] < args.target_attempts:
                    state["events"].append(
                        {
                            "timestamp": time.time(),
                            "event": "target_retry_scheduled",
                            "target": target.key,
                            "failed_attempt": result["attempt"],
                            "status": result["status"],
                        }
                    )
                    await asyncio.to_thread(_append_jsonl, run_dir / "failed_attempts.jsonl", result)
                    pending.append(target)
                    continue
                history = attempt_history[target.key]
                if len(history) > 1:
                    first = history[0]
                    result["measurement_attempts"] = len(history)
                    result["elapsed_seconds"] = sum(item.get("elapsed_seconds", 0.0) for item in history)
                    result["user_cpu_seconds"] = sum(item.get("user_cpu_seconds", 0.0) for item in history)
                    result["system_cpu_seconds"] = sum(item.get("system_cpu_seconds", 0.0) for item in history)
                    result["max_rss_bytes"] = max(item.get("max_rss_bytes", 0) or 0 for item in history)
                    result["started_at"] = first["started_at"]
                    result["artifact_before"] = first["artifact_before"]
                    result["artifact_growth_bytes"] = (
                        result["artifact_after"]["artifact_total_bytes"]
                        - first["artifact_before"]["artifact_total_bytes"]
                    )
                    result["files_per_second"] = (
                        result["input_file_count"] / result["elapsed_seconds"] if result["elapsed_seconds"] else 0.0
                    )
                else:
                    result["measurement_attempts"] = 1
                if result["strategy"] == "prehop" and result.get("index_stats"):
                    _write_json(
                        run_dir / f"prehop_inspection__{result['dataset']}.json",
                        {
                            "dataset": result["dataset"],
                            "inspection_samples": result["index_stats"].get("inspection_samples", []),
                            "hop_inspection_samples": result["index_stats"].get("hop_inspection_samples", []),
                        },
                    )
                results.append(result)
                results.sort(
                    key=lambda item: (
                        selected_datasets.index(item["dataset"]),
                        selected_strategies.index(item["strategy"]),
                    )
                )
                _write_json(run_dir / "results.json", results)
    finally:
        stop.set()
        await sampler

    _add_target_pressure_peaks(results, state["samples"])
    _write_json(run_dir / "results.json", results)
    _write_json(run_dir / "parallelism_events.json", state["events"])
    _write_tables(run_dir, results)
    failed = [result for result in results if result["status"] != "complete"]
    print(f"[matrix] artifacts: {run_dir}")
    print(f"[matrix] complete={len(results) - len(failed)} failed={len(failed)}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["all", *DATASETS], default=["all"])
    parser.add_argument("--strategies", nargs="+", choices=["all", *STRATEGIES], default=["all"])
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument(
        "--max-generation-parallel",
        type=int,
        default=1,
        help="Maximum generation-heavy targets in flight; embedding-only Naive work can fill remaining slots.",
    )
    parser.add_argument("--sample-seconds", type=float, default=5.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--clear-graph", action="store_true")
    parser.add_argument("--save-prehop-intermediate", action="store_true")
    parser.add_argument("--target-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_parallel < 1:
        parser.error("--max-parallel must be >= 1")
    if args.max_generation_parallel < 1 or args.max_generation_parallel > args.max_parallel:
        parser.error("--max-generation-parallel must be between 1 and --max-parallel")
    if args.sample_seconds <= 0:
        parser.error("--sample-seconds must be > 0")
    if args.target_attempts < 1:
        parser.error("--target-attempts must be >= 1")
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
