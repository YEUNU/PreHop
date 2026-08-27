import asyncio
import fcntl
import hashlib
import json
import logging
import multiprocessing as _mp
import os
import re
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from core.config import RAGConfig
from core.neo4j_service import Neo4jService
from models.naive.naive_rag import NaiveRAG
from models.prehop.graphrag import GraphRAG
from models.prehop.indexing.chunking import parse_pages_offline
from utils.io import _write_json
from utils.provenance import code_provenance

# Spawn-based context for the parsing worker pool. Using the default `fork`
# context corrupts the parent process's httpx/openai async clients (vLLM
# requests stop being dispatched after the pool shuts down), which manifests
# as 100% CPU on the main thread but 0 reqs at the vLLM serve endpoint.
_PARSE_MP_CTX = _mp.get_context("spawn")


logger = logging.getLogger("Prehop")
_CORPUS_MANIFEST_FILENAME = "corpus_manifest.json"
_SNAPSHOT_LABEL = "RAGIndexSnapshot"
_SNAPSHOT_VERSION = 1


async def _reap_bounded_tasks(
    pending: dict[asyncio.Task[None], str],
    *,
    wait_for_one: bool,
) -> list[tuple[str, Exception]]:
    """Remove completed tasks from a bounded scheduling window."""
    if not pending:
        return []
    done, _remaining = await asyncio.wait(
        pending,
        return_when=asyncio.FIRST_COMPLETED if wait_for_one else asyncio.ALL_COMPLETED,
    )
    errors = []
    for task in done:
        filename = pending.pop(task)
        try:
            await task
        except Exception as exc:  # noqa: BLE001 - preserve the file identity for aggregate reporting
            errors.append((filename, exc))
    return errors


async def _submit_bounded_task(
    pending: dict[asyncio.Task[None], str],
    limit: int,
    filename: str,
    coroutine_factory: Callable[[], Awaitable[None]],
) -> list[tuple[str, Exception]]:
    """Submit work after waiting only for the next available window slot."""
    errors = []
    if len(pending) >= limit:
        errors = await _reap_bounded_tasks(pending, wait_for_one=True)
    pending[asyncio.create_task(coroutine_factory())] = filename
    return errors


def _artifact_run_id() -> str:
    raw = os.environ.get("RAG_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-") or "run"


def _resolved_index_policy(strategy: str, indexing_model_id: str) -> dict:
    """Record semantic index settings separately from throughput controls."""
    if strategy == "prehop":
        resolved_generation_model = (
            RAGConfig.DEFAULT_MODEL if indexing_model_id == "default" else indexing_model_id
        )
    elif strategy in {"hoprag", "ms_graphrag"}:
        resolved_generation_model = RAGConfig.DEFAULT_MODEL
    else:
        resolved_generation_model = None
    embedding_model = (
        os.environ.get("RAG_HOP_EMBED_MODEL_NAME", RAGConfig.EMBEDDING_MODEL)
        if strategy == "hoprag"
        else RAGConfig.EMBEDDING_MODEL
    )
    policy = {
        "strategy": strategy,
        "indexing_model": resolved_generation_model,
        "generation_revision": os.environ.get("RAG_GENERATION_REVISION", "").strip() or None,
        "embedding_model": embedding_model,
        "embedding_revision": os.environ.get("RAG_EMBEDDING_REVISION", "").strip() or None,
        "embedding_query_instruction": RAGConfig.EMBEDDING_QUERY_INSTRUCTION,
        "embedding_dimensions": RAGConfig.EMBEDDING_DIMENSIONS,
        "embedding_max_input_tokens": RAGConfig.MAX_EMBEDDING_LENGTH,
        "fulltext_analyzer": RAGConfig.FULLTEXT_ANALYZER,
    }
    if strategy in {"prehop", "naive"}:
        policy["chunk_sentences"] = RAGConfig.CHUNK_SENTENCES
    if strategy == "prehop":
        policy.update(
            {
                "questions_per_direction": RAGConfig.QUESTIONS_PER_DIRECTION,
                "question_schema": RAGConfig.QUESTION_SCHEMA,
                "q_minus_enabled": RAGConfig.ABLATION_Q_MINUS,
                "q_plus_enabled": RAGConfig.ABLATION_Q_PLUS,
                "precompute_reciprocal_hops": RAGConfig.PRECOMPUTE_RECIPROCAL_HOPS,
                "hop_construction": (
                    "qplus_to_qminus_owner" if RAGConfig.ABLATION_Q_MINUS else "qplus_to_body_ablation"
                ),
            }
        )
    return policy


def _load_corpus_manifest(dataset_path: str | Path) -> dict | None:
    """Read the optional immutable corpus identity without changing old corpora."""
    manifest_path = Path(dataset_path) / _CORPUS_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid corpus manifest: {manifest_path}: {exc}") from exc
    fingerprint = payload.get("fingerprint")
    paragraph_count = payload.get("paragraph_count")
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise ValueError(f"Corpus manifest has no fingerprint: {manifest_path}")
    if not isinstance(paragraph_count, int) or paragraph_count < 0:
        raise ValueError(f"Corpus manifest has invalid paragraph_count: {manifest_path}")
    return {"fingerprint": fingerprint, "paragraph_count": paragraph_count}


def _source_ids_from_filenames(filenames: list[str]) -> list[str]:
    """Return the exact staged-document identity set used by every indexer.

    The corpus filename (without extension) is the durable source identity.
    Reject ambiguous ``foo.txt``/``foo.md`` pairs instead of silently merging
    them in a graph index whose source property is extension-independent.
    """
    source_ids = sorted(Path(filename).stem for filename in filenames)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Corpus has duplicate .txt/.md filename stems; source identity would be ambiguous")
    return source_ids


def _source_set_sha256(source_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(source_ids)).encode("utf-8")).hexdigest()


def _validate_staged_snapshot(files: list[str], corpus_manifest: dict | None) -> list[str]:
    source_ids = _source_ids_from_filenames(files)
    if corpus_manifest is not None and corpus_manifest["paragraph_count"] != len(source_ids):
        raise ValueError(
            "Corpus manifest paragraph_count does not match staged .txt/.md files: "
            f"{corpus_manifest['paragraph_count']} != {len(source_ids)}"
        )
    return source_ids


async def _set_neo4j_snapshot_state(
    engine,
    strategy: str,
    corpus_tag: str,
    corpus_manifest: dict | None,
    status: str,
) -> None:
    """Invalidate or publish metadata outside all retrieval labels.

    The node is deliberately unconnected and uses a fixed generic label, so
    it cannot participate in vector/full-text retrieval or alter an upstream
    strategy's scoring, ranking, top-k, or traversal.
    """
    await engine.neo4j.execute_query(
        f"""
        MERGE (m:{_SNAPSHOT_LABEL} {{strategy: $strategy, corpus_tag: $corpus_tag}})
        SET m.status = $status,
            m.snapshot_version = $_version,
            m.corpus_manifest_fingerprint = $fingerprint,
            m.corpus_manifest_paragraph_count = $paragraph_count,
            m.updated_at_epoch = $updated_at
        """,
        {
            "strategy": strategy,
            "corpus_tag": corpus_tag,
            "status": status,
            "_version": _SNAPSHOT_VERSION,
            "fingerprint": (corpus_manifest or {}).get("fingerprint"),
            "paragraph_count": (corpus_manifest or {}).get("paragraph_count"),
            "updated_at": time.time(),
        },
    )


async def _verify_and_publish_neo4j_snapshot(
    engine,
    strategy: str,
    corpus_tag: str,
    source_ids: list[str],
    corpus_manifest: dict | None,
) -> dict[str, object]:
    """Verify the active graph itself, then and only then mark it complete."""
    rows = await engine.neo4j.execute_query(
        f"""
        MATCH (c:{engine.chunk_label})
        WHERE coalesce(c.source, '') <> ''
        RETURN DISTINCT c.source AS source
        """
    )
    actual_ids = sorted({Path(str(row.get("source") or "")).stem for row in rows})
    expected = sorted(source_ids)
    if actual_ids != expected:
        missing = sorted(set(expected) - set(actual_ids))
        unexpected = sorted(set(actual_ids) - set(expected))
        raise RuntimeError(
            "Active Neo4j source snapshot does not match staged corpus: "
            f"expected={len(expected)} actual={len(actual_ids)} "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    source_digest = _source_set_sha256(actual_ids)
    await engine.neo4j.execute_query(
        f"""
        MERGE (m:{_SNAPSHOT_LABEL} {{strategy: $strategy, corpus_tag: $corpus_tag}})
        SET m.status = 'complete',
            m.snapshot_version = $_version,
            m.corpus_manifest_fingerprint = $fingerprint,
            m.corpus_manifest_paragraph_count = $paragraph_count,
            m.source_count = $source_count,
            m.source_set_sha256 = $source_digest,
            m.completed_at_epoch = $completed_at,
            m.updated_at_epoch = $completed_at
        """,
        {
            "strategy": strategy,
            "corpus_tag": corpus_tag,
            "_version": _SNAPSHOT_VERSION,
            "fingerprint": (corpus_manifest or {}).get("fingerprint"),
            "paragraph_count": (corpus_manifest or {}).get("paragraph_count"),
            "source_count": len(actual_ids),
            "source_digest": source_digest,
            "completed_at": time.time(),
        },
    )
    return {"status": "complete", "source_count": len(actual_ids), "source_set_sha256": source_digest}


def _write_runtime_stage_stats(
    strategy: str,
    corpus_tag: str,
    dataset_path: str,
    timing_seconds: dict[str, float],
    status: str,
    corpus_manifest: dict | None = None,
    indexing_model_id: str = "default",
    index_capacity: dict | None = None,
) -> None:
    """Persist timing even for official adapters that do not use our graph stats."""
    stats_dir = Path("data/index_stats")
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / f"{strategy}_{corpus_tag}_{_artifact_run_id()}.json"
    _write_json(
        stats_path,
        {
            "run_id": _artifact_run_id(),
            "index_code_provenance": code_provenance(),
            "index_policy": _resolved_index_policy(strategy, indexing_model_id),
            "strategy": strategy,
            "corpus_tag": corpus_tag,
            "dataset_path": dataset_path,
            "timing_seconds": dict(timing_seconds),
            "index_capacity": index_capacity,
            "status": status,
            "corpus_manifest_fingerprint": (corpus_manifest or {}).get("fingerprint"),
            "corpus_manifest_paragraph_count": (corpus_manifest or {}).get("paragraph_count"),
        },
    )


async def _collect_index_capacity(
    strategy: str,
    corpus_tag: str,
    neo4j: Neo4jService | None = None,
) -> dict[str, object]:
    """Measure strategy storage with the fixed comparison-table definitions."""
    safe_corpus = re.sub(r"[^A-Za-z0-9_]", "_", corpus_tag)
    if strategy == "ms_graphrag":
        root = Path("data/ms_graphrag_output") / corpus_tag
        if not root.is_dir():
            raise FileNotFoundError(f"MS GraphRAG retrieval artifact directory not found: {root}")
        excluded = {"_cache", "_logs", "_input"}
        total_bytes = sum(
            path.stat().st_size
            for path in root.rglob("*")
            if path.is_file() and not any(part in excluded for part in path.relative_to(root).parts)
        )
        return {
            "measurement": "physical_retrieval_artifact_size",
            "bytes": total_bytes,
            "gib": total_bytes / 1024**3,
            "excluded_directories": sorted(excluded),
            "definition_version": 1,
        }

    service = neo4j or Neo4jService()
    if strategy == "prehop":
        labels = [
            f"PR_{safe_corpus}_Chunk",
            f"PR_{safe_corpus}_Document",
            f"PR_{safe_corpus}_QMinus",
            f"PR_{safe_corpus}_QPlus",
        ]
        label_literal = json.dumps(labels)
        queries = [
            f"""
            MATCH (n)
            WHERE any(label IN labels(n) WHERE label IN {label_literal})
            RETURN sum(coalesce(size(n.embedding), 0) + coalesce(size(n.query_embedding), 0)) AS floats,
                   sum(coalesce(size(n.text), 0) + coalesce(size(n.id), 0) +
                       coalesce(size(n.source), 0) + coalesce(size(n.title), 0) +
                       coalesce(size(n.filename), 0)) AS chars,
                   0 AS list_items, count(n) AS records
            """,
            f"""
            MATCH (a)-[r]->()
            WHERE any(label IN labels(a) WHERE label IN {label_literal})
            RETURN 0 AS floats,
                   sum(coalesce(size(r.question), 0) + coalesce(size(r.type), 0)) AS chars,
                   sum(coalesce(size(r.source_question_texts), 0) +
                       coalesce(size(r.direct_channels), 0) +
                       coalesce(size(r.source_question_ids), 0) +
                       coalesce(size(r.reciprocal_source_question_ids), 0)) AS list_items,
                   count(r) AS records
            """,
        ]
    elif strategy == "naive":
        label = f"NA_{safe_corpus}_Chunk"
        queries = [
            f"""
            MATCH (n:{label})
            RETURN sum(coalesce(size(n.embedding), 0)) AS floats,
                   sum(coalesce(size(n.text), 0) + coalesce(size(n.id), 0) +
                       coalesce(size(n.source), 0) + coalesce(size(n.title), 0)) AS chars,
                   0 AS list_items, count(n) AS records
            """
        ]
    elif strategy == "hoprag":
        label = f"HO_{safe_corpus}"
        queries = [
            f"""
            MATCH (n:{label})
            RETURN sum(coalesce(size(n.embed), 0)) AS floats,
                   sum(coalesce(size(n.text), 0) + coalesce(size(n.source), 0) +
                       coalesce(size(n.title), 0)) +
                       sum(reduce(total = 0, item IN coalesce(n.keywords, []) |
                           total + size(item))) AS chars,
                   0 AS list_items, count(n) AS records
            """,
            f"""
            MATCH (:{label})-[r]->(:{label})
            RETURN sum(coalesce(size(r.embed), 0)) AS floats,
                   sum(coalesce(size(r.question), 0)) +
                       sum(reduce(total = 0, item IN coalesce(r.keywords, []) |
                           total + size(item))) AS chars,
                   0 AS list_items, count(r) AS records
            """,
        ]
    else:
        raise ValueError(f"Unsupported capacity measurement strategy: {strategy}")

    rows = []
    for query in queries:
        result = await service.execute_query(query)
        if len(result) != 1:
            raise RuntimeError(f"Capacity query returned {len(result)} rows for strategy={strategy}")
        rows.append(result[0])
    vector_elements = sum(int(row.get("floats") or 0) for row in rows)
    text_characters = sum(int(row.get("chars") or 0) for row in rows)
    list_items = sum(int(row.get("list_items") or 0) for row in rows)
    records = sum(int(row.get("records") or 0) for row in rows)
    total_bytes = vector_elements * 8 + text_characters + list_items * 8 + records * 8
    return {
        "measurement": "estimated_logical_property_payload",
        "bytes": total_bytes,
        "gib": total_bytes / 1024**3,
        "vector_elements": vector_elements,
        "text_characters": text_characters,
        "list_items": list_items,
        "records": records,
        "definition_version": 1,
    }


async def _collect_prehop_integrity(engine) -> dict[str, object]:
    chunk = engine.chunk_label
    doc = engine.doc_label
    q_minus = engine.q_minus_label
    q_plus = engine.q_plus_label

    representation_rows = await engine.neo4j.execute_query(f"""
        CALL () {{
            MATCH (c:{chunk})
            RETURN count(CASE WHEN c.embedding IS NULL THEN 1 END) AS missing_embeddings,
                   count(CASE WHEN NOT (:{doc})-[:CONTAINS]->(c) THEN 1 END) AS orphan_nodes
            UNION ALL
            MATCH (q:{q_minus})
            RETURN count(CASE WHEN q.embedding IS NULL THEN 1 END) AS missing_embeddings,
                   count(CASE WHEN NOT (:{chunk})-[:HAS_Q_MINUS]->(q) THEN 1 END) AS orphan_nodes
            UNION ALL
            MATCH (q:{q_plus})
            RETURN count(CASE WHEN q.embedding IS NULL OR q.query_embedding IS NULL THEN 1 END) AS missing_embeddings,
                   count(CASE WHEN NOT (:{chunk})-[:HAS_Q_PLUS]->(q) THEN 1 END) AS orphan_nodes
        }}
        RETURN sum(missing_embeddings) AS missing_embeddings,
               sum(orphan_nodes) AS orphan_nodes
    """)
    representation = representation_rows[0] if representation_rows else {}

    question_rows = await engine.neo4j.execute_query(f"""
        CALL () {{
            MATCH (c:{chunk})-[:HAS_Q_MINUS]->(q:{q_minus})
            WITH c, toLower(trim(q.text)) AS text, count(*) AS copies
            RETURN count(CASE WHEN text = '' THEN 1 END) AS empty_questions,
                   count(CASE WHEN text =~ '.*(provided text|given text|this chunk|the passage).*' THEN 1 END) AS source_relative,
                   count(CASE WHEN copies > 1 THEN 1 END) AS duplicate_groups
            UNION ALL
            MATCH (c:{chunk})-[:HAS_Q_PLUS]->(q:{q_plus})
            WITH c, toLower(trim(q.text)) AS text, count(*) AS copies
            RETURN count(CASE WHEN text = '' THEN 1 END) AS empty_questions,
                   count(CASE WHEN text =~ '.*(provided text|given text|this chunk|the passage).*' THEN 1 END) AS source_relative,
                   count(CASE WHEN copies > 1 THEN 1 END) AS duplicate_groups
        }}
        RETURN sum(empty_questions) AS empty_questions,
               sum(source_relative) AS source_relative_questions,
               sum(duplicate_groups) AS duplicate_question_groups
    """)
    questions = question_rows[0] if question_rows else {}
    cross_channel_rows = await engine.neo4j.execute_query(f"""
        MATCH (c:{chunk})-[:HAS_Q_MINUS]->(qm:{q_minus}),
              (c)-[:HAS_Q_PLUS]->(qp:{q_plus})
        WHERE toLower(trim(qm.text)) = toLower(trim(qp.text))
        RETURN count(*) AS identical_cross_channel_questions
    """)
    cross_channel = cross_channel_rows[0] if cross_channel_rows else {}
    grounding: dict[str, object] = {}
    if RAGConfig.QUESTION_SCHEMA == "grounded_v1":
        grounding_rows = await engine.neo4j.execute_query(f"""
            CALL () {{
                MATCH (q:{q_minus})
                RETURN count(CASE WHEN q.question_schema <> 'grounded_v1'
                                       OR trim(coalesce(q.grounding_quote, '')) = ''
                                       OR size(coalesce(q.anchor_entities, [])) = 0
                                       OR trim(coalesce(q.answer, '')) = ''
                                  THEN 1 END) AS invalid_grounding
                UNION ALL
                MATCH (q:{q_plus})
                RETURN count(CASE WHEN q.question_schema <> 'grounded_v1'
                                       OR trim(coalesce(q.grounding_quote, '')) = ''
                                       OR size(coalesce(q.anchor_entities, [])) = 0
                                       OR trim(coalesce(q.missing_information, '')) = ''
                                  THEN 1 END) AS invalid_grounding
            }}
            RETURN sum(invalid_grounding) AS invalid_grounding
        """)
        grounding = grounding_rows[0] if grounding_rows else {}
    q_plus_count_rows = await engine.neo4j.execute_query(
        f"MATCH (q:{q_plus}) RETURN count(q) AS count"
    )
    q_plus_count = int((q_plus_count_rows[0] if q_plus_count_rows else {}).get("count", 0) or 0)

    next_rows = await engine.neo4j.execute_query(f"""
        CALL () {{
            MATCH (d:{doc})-[:CONTAINS]->(c:{chunk})
            WITH d, count(c) AS chunks
            RETURN sum(CASE WHEN chunks > 0 THEN chunks - 1 ELSE 0 END) AS expected,
                   0 AS actual, 0 AS invalid, 0 AS missing
            UNION ALL
            MATCH (source:{chunk})-[r:NEXT]->(target:{chunk})
            RETURN 0 AS expected, count(r) AS actual,
                   count(CASE WHEN source.source <> target.source OR
                                   target.sent_id <> source.sent_id + 1 THEN 1 END) AS invalid,
                   0 AS missing
            UNION ALL
            MATCH (source:{chunk})
            MATCH (target:{chunk} {{source: source.source, sent_id: source.sent_id + 1}})
            WHERE NOT (source)-[:NEXT]->(target)
            RETURN 0 AS expected, 0 AS actual, 0 AS invalid, count(*) AS missing
        }}
        RETURN sum(expected) AS expected, sum(actual) AS actual,
               sum(invalid) AS invalid, sum(missing) AS missing
    """)
    next_topology = next_rows[0] if next_rows else {}

    hop_rows = await engine.neo4j.execute_query(f"""
        MATCH (source:{chunk})-[h:HOP_ANSWER]->(target:{chunk})
        RETURN source.id AS source_id, source.source AS source_document,
               target.id AS target_id, target.source AS target_document,
               h.direct_channels AS direct_channels,
               h.source_question_ids AS source_question_ids,
               h.source_question_texts AS source_question_texts,
               h.type AS edge_type
    """)
    expected_channels = {"q_minus"} if RAGConfig.ABLATION_Q_MINUS else {"body"}
    expected_edge_type = "qplus_to_qminus_owner" if RAGConfig.ABLATION_Q_MINUS else "qplus_to_body_ablation"
    invalid_hop_edges = 0
    for row in hop_rows:
        invalid_hop_edges += int(
            row.get("source_document") == row.get("target_document")
            or set(row.get("direct_channels") or []) != expected_channels
            or not row.get("source_question_ids")
            or not row.get("source_question_texts")
            or row.get("edge_type") != expected_edge_type
        )

    provenance_rows = await engine.neo4j.execute_query(f"""
        MATCH (source:{chunk})-[h:HOP_ANSWER]->(target:{chunk})
        UNWIND coalesce(h.source_question_ids, []) AS question_id
        OPTIONAL MATCH (source)-[:HAS_Q_PLUS]->(q:{q_plus} {{id: question_id}})
        RETURN count(CASE WHEN q IS NULL THEN 1 END) AS missing_source_questions,
               count(CASE WHEN q IS NOT NULL AND NOT EXISTS {{
                   MATCH (q)-[:ANSWERED_BY]->(:{q_minus})<-[:HAS_Q_MINUS]-(target)
               }} THEN 1 END) AS missing_answered_by,
               count(CASE WHEN q IS NOT NULL AND NOT EXISTS {{
                   MATCH (q)-[:SUPPORTED_BY]->(target)
               }} THEN 1 END) AS missing_supported_by
    """)
    provenance = provenance_rows[0] if provenance_rows else {}
    provenance_mismatches = (
        int(provenance.get("missing_answered_by", 0) or 0)
        if RAGConfig.ABLATION_Q_MINUS
        else int(provenance.get("missing_supported_by", 0) or 0)
    )
    degree_rows = await engine.neo4j.execute_query(f"""
        MATCH (source:{chunk})-[h:HOP_ANSWER]->()
        WITH source, count(h) AS degree
        RETURN max(degree) AS max_out_degree
    """)
    degree = degree_rows[0] if degree_rows else {}
    index_rows = await engine.neo4j.execute_query(
        f"SHOW INDEXES YIELD name, state WHERE name STARTS WITH 'prehop_{engine._safe_corpus}' RETURN name, state"
    )

    checks = {
        "complete_embeddings": int(representation.get("missing_embeddings", 0) or 0) == 0,
        "complete_ownership": int(representation.get("orphan_nodes", 0) or 0) == 0,
        "valid_questions": int(questions.get("empty_questions", 0) or 0) == 0
        and int(questions.get("source_relative_questions", 0) or 0) == 0,
        "unique_questions_per_channel": int(questions.get("duplicate_question_groups", 0) or 0) == 0,
        "distinct_question_roles": int(cross_channel.get("identical_cross_channel_questions", 0) or 0) == 0,
        "complete_question_grounding": int(grounding.get("invalid_grounding", 0) or 0) == 0,
        "exact_next_topology": int(next_topology.get("expected", 0) or 0)
        == int(next_topology.get("actual", 0) or 0)
        and int(next_topology.get("invalid", 0) or 0) == 0
        and int(next_topology.get("missing", 0) or 0) == 0,
        "valid_hop_edges": invalid_hop_edges == 0,
        "hop_graph_available": not RAGConfig.ABLATION_Q_PLUS or q_plus_count == 0 or bool(hop_rows),
        "consistent_hop_provenance": int(provenance.get("missing_source_questions", 0) or 0) == 0
        and provenance_mismatches == 0,
        "bounded_hop_out_degree": int(degree.get("max_out_degree", 0) or 0)
        <= RAGConfig.QUESTIONS_PER_DIRECTION,
        "search_indexes_online": bool(index_rows) and all(row.get("state") == "ONLINE" for row in index_rows),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "diagnostics": {
            "representation": representation,
            "questions": questions,
            "grounding": grounding,
            "cross_channel": cross_channel,
            "next_topology": next_topology,
            "hop_edges": len(hop_rows),
            "invalid_hop_edges": invalid_hop_edges,
            "provenance": provenance,
            "max_hop_out_degree": int(degree.get("max_out_degree", 0) or 0),
        },
    }


async def _collect_graph_stats(engine, strategy: str) -> dict | None:
    """Query the live graph for structural statistics after indexing.

    Queries the graph directly (not counters threaded through indexing)
    so this is always consistent with what actually landed in Neo4j, the
    same way CLAUDE.md's "Neo4j data layout" integrity probes work. Only
    meaningful for `prehop` (Q-/Q+/HOP structure); other strategies don't
    have this graph shape.
    """
    if strategy != "prehop" or not hasattr(engine, "chunk_label"):
        return None
    chunk_label = engine.chunk_label
    doc_label = engine.doc_label
    q_minus_label = engine.q_minus_label
    q_plus_label = engine.q_plus_label

    chunk_rows = await engine.neo4j.execute_query(f"""
        MATCH (c:{chunk_label})
        RETURN count(c) AS total_chunks
    """)
    chunk_stats = chunk_rows[0] if chunk_rows else {}

    doc_rows = await engine.neo4j.execute_query(f"MATCH (d:{doc_label}) RETURN count(d) AS total_docs")
    doc_stats = doc_rows[0] if doc_rows else {}

    question_rows = await engine.neo4j.execute_query(f"""
        MATCH (c:{chunk_label})
        OPTIONAL MATCH (c)-[:HAS_Q_MINUS]->(qm:{q_minus_label})
        WITH count(DISTINCT c) AS total_chunks,
             count(DISTINCT qm) AS q_minus_questions,
             count(DISTINCT CASE WHEN qm IS NOT NULL THEN c END) AS q_minus_chunks
        MATCH (c2:{chunk_label})
        OPTIONAL MATCH (c2)-[:HAS_Q_PLUS]->(qp:{q_plus_label})
        RETURN total_chunks, q_minus_questions, q_minus_chunks,
               count(DISTINCT qp) AS q_plus_questions,
               count(DISTINCT CASE WHEN qp IS NOT NULL THEN c2 END) AS q_plus_chunks
    """)
    question_stats = question_rows[0] if question_rows else {}

    edge_rows = await engine.neo4j.execute_query(f"""
        OPTIONAL MATCH (:{chunk_label})-[hop:HOP_ANSWER]->(:{chunk_label})
        WITH count(hop) AS total_hop_edges
        OPTIONAL MATCH (:{q_plus_label})-[answer:ANSWERED_BY]->(:{q_minus_label})
        WITH total_hop_edges, count(answer) AS answered_by_edges
        OPTIONAL MATCH (:{q_plus_label})-[body:SUPPORTED_BY]->(:{chunk_label})
        RETURN total_hop_edges, answered_by_edges, count(body) AS supported_by_edges
    """)
    edge_stats = edge_rows[0] if edge_rows else {}
    direction_rows = await engine.neo4j.execute_query(f"""
        MATCH (q:{q_plus_label})
        RETURN count(q) AS total_q_plus,
               count(CASE WHEN EXISTS {{
                   MATCH (q)-[:ANSWERED_BY|SUPPORTED_BY]->()
               }} THEN 1 END) AS linked_q_plus
    """)
    direction_stats = direction_rows[0] if direction_rows else {}
    hop_bridge_rows = await engine.neo4j.execute_query(f"""
        MATCH (:{chunk_label})-[hop:HOP_ANSWER]->(:{chunk_label})
        RETURN count(CASE WHEN size(coalesce(hop.source_question_ids, [])) = 0 OR
                               size(coalesce(hop.source_question_texts, [])) = 0
                          THEN 1 END) AS hop_without_bridge_provenance
    """)
    hop_bridge_stats = hop_bridge_rows[0] if hop_bridge_rows else {}

    total_chunks = chunk_stats.get("total_chunks", 0) or 0
    total_docs = doc_stats.get("total_docs", 0) or 0
    q_minus_chunks = question_stats.get("q_minus_chunks", 0) or 0
    q_plus_chunks = question_stats.get("q_plus_chunks", 0) or 0
    q_minus_questions = question_stats.get("q_minus_questions", 0) or 0
    q_plus_questions = question_stats.get("q_plus_questions", 0) or 0
    total_hop_edges = edge_stats.get("total_hop_edges", 0) or 0
    integrity = await _collect_prehop_integrity(engine)

    return {
        "index_quality": integrity,
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "avg_chunks_per_doc": (total_chunks / total_docs) if total_docs else 0.0,
        "q_minus_coverage": (q_minus_chunks / total_chunks) if total_chunks else 0.0,
        "q_plus_coverage": (q_plus_chunks / total_chunks) if total_chunks else 0.0,
        "q_minus_questions": q_minus_questions,
        "q_plus_questions": q_plus_questions,
        "avg_q_minus_per_covered_chunk": (q_minus_questions / q_minus_chunks) if q_minus_chunks else 0.0,
        "avg_q_plus_per_covered_chunk": (q_plus_questions / q_plus_chunks) if q_plus_chunks else 0.0,
        "total_hop_edges": total_hop_edges,
        "answered_by_edges": edge_stats.get("answered_by_edges", 0) or 0,
        "supported_by_edges": edge_stats.get("supported_by_edges", 0) or 0,
        "linked_q_plus_questions": direction_stats.get("linked_q_plus", 0) or 0,
        "q_plus_direction_coverage": (
            (direction_stats.get("linked_q_plus", 0) or 0) / (direction_stats.get("total_q_plus", 0) or 1)
        ),
        "hop_without_bridge_provenance": (hop_bridge_stats.get("hop_without_bridge_provenance", 0) or 0),
        # Out-degree among HOP-eligible (Q+-surviving) source chunks only —
        # matches the "wrote N HOP edges over M Q+ chunks" indexing log line.
        "avg_hop_out_degree_per_eligible_chunk": (total_hop_edges / q_plus_chunks) if q_plus_chunks else 0.0,
        # Out-degree across the whole chunk population (most chunks have none
        # since only Q+-surviving chunks can be a HOP source) — overall graph
        # density.
        "avg_hop_out_degree_per_chunk": (total_hop_edges / total_chunks) if total_chunks else 0.0,
    }


async def rebuild_hop_edges(corpus_tag: str, strategy: str = "prehop") -> dict | None:
    """Rebuild HOP and provenance edges without changing chunks or questions."""
    if strategy != "prehop":
        raise ValueError(f"rebuild_hop_edges only supports strategy=prehop (got {strategy})")
    engine = GraphRAG(strategy=strategy, corpus_tag=corpus_tag)
    await engine.clear_hop_edges()
    await engine.build_all_hop_edges()
    stats = await _collect_graph_stats(engine, strategy)
    if stats is not None and not bool(stats.get("index_quality", {}).get("pass")):
        raise RuntimeError(f"Prehop index quality checks failed: {stats['index_quality']['checks']}")
    logger.info("HOP rebuild complete for corpus_tag=%s: %s", corpus_tag, stats)
    return stats


@asynccontextmanager
async def _index_run_lock(strategy: str, corpus_tag: str):
    lock_dir = Path("data/index_locks")
    lock_dir.mkdir(parents=True, exist_ok=True)
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{strategy}_{corpus_tag}")
    lock_path = lock_dir / f"{safe_key}.lock"
    handle = await asyncio.to_thread(lock_path.open, "a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Indexing is already running for strategy={strategy}, corpus={corpus_tag}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} run_id={_artifact_run_id()}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


async def run_indexing(
    dataset_path: str,
    strategy: str,
    model_id: str,
    corpus_tag: str | None = None,
    save_intermediate: bool = False,
):
    """Serialize duplicate strategy/corpus runs while allowing all distinct targets in parallel."""
    async with _index_run_lock(strategy, corpus_tag or "default"):
        return await _run_indexing_unlocked(
            dataset_path,
            strategy,
            model_id,
            corpus_tag,
            save_intermediate,
        )


async def _run_indexing_unlocked(
    dataset_path: str,
    strategy: str,
    model_id: str,
    corpus_tag: str | None = None,
    save_intermediate: bool = False,
):
    """Index files using selected strategy with parallel processing."""
    started_at = time.perf_counter()
    stage_timing: dict[str, float] = {}
    logger.info(
        "Indexing strategy: %s | Dataset: %s | Corpus: %s",
        strategy,
        dataset_path,
        corpus_tag or "default",
    )
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    corpus_manifest = _load_corpus_manifest(dataset_path)

    files = sorted(file for file in os.listdir(dataset_path) if file.endswith((".txt", ".md")))
    if not files:
        raise ValueError(f"Dataset contains no supported .txt/.md files: {dataset_path}")
    source_ids = _validate_staged_snapshot(files, corpus_manifest)

    if strategy == "ms_graphrag":
        # Official MS GraphRAG pipeline (extract_graph + Leiden + community
        # reports), routed through the configured external endpoint. Outputs parquet
        # under data/ms_graphrag_output/<corpus_tag>/. Skips our chunking/
        # HOP/summary stages — MS does its own.
        from models.ms_graphrag.official_indexer import run_official_index as run_ms_index

        official_started = time.perf_counter()
        timing = {}
        official_status = "complete"
        index_capacity = None
        try:
            adapter_timing = await run_ms_index(
                dataset_path=dataset_path,
                corpus_tag=corpus_tag or "default",
                corpus_manifest=corpus_manifest,
            )
            timing.update(adapter_timing or {})
            timing["official_pipeline_seconds"] = time.perf_counter() - official_started
            timing["total_elapsed_seconds"] = time.perf_counter() - started_at
            index_capacity = await _collect_index_capacity(strategy, corpus_tag or "default")
        except BaseException:
            official_status = "failed"
            raise
        finally:
            timing.setdefault("official_pipeline_seconds", time.perf_counter() - official_started)
            timing.setdefault("total_elapsed_seconds", time.perf_counter() - started_at)
            _write_runtime_stage_stats(
                strategy,
                corpus_tag or "default",
                dataset_path,
                timing,
                official_status,
                corpus_manifest,
                model_id,
                index_capacity,
            )
        return

    if strategy == "hoprag":
        # Official HopRAG indexing (QABuilder.create_nodes + grouped
        # create_edge + create_index). Generation and embeddings use the
        # configured external OpenAI-compatible endpoints. Writes directly to
        # Neo4j under HO_<corpus_tag>_* labels.
        from models.hoprag.official_indexer import run_official_index as run_hop_index

        official_started = time.perf_counter()
        timing = {}
        official_status = "complete"
        index_capacity = None
        try:
            adapter_timing = await run_hop_index(
                dataset_path=dataset_path,
                corpus_tag=corpus_tag or "default",
                corpus_manifest=corpus_manifest,
            )
            timing.update(adapter_timing or {})
            timing["official_pipeline_seconds"] = time.perf_counter() - official_started
            timing["total_elapsed_seconds"] = time.perf_counter() - started_at
            index_capacity = await _collect_index_capacity(strategy, corpus_tag or "default")
        except BaseException:
            official_status = "failed"
            raise
        finally:
            timing.setdefault("official_pipeline_seconds", time.perf_counter() - official_started)
            timing.setdefault("total_elapsed_seconds", time.perf_counter() - started_at)
            _write_runtime_stage_stats(
                strategy,
                corpus_tag or "default",
                dataset_path,
                timing,
                official_status,
                corpus_manifest,
                model_id,
                index_capacity,
            )
        return

    if strategy == "prehop":
        engine = GraphRAG(
            strategy=strategy,
            indexing_model_id=model_id,
            corpus_tag=corpus_tag,
            save_intermediate=save_intermediate,
        )
        is_graph = True
    elif strategy == "naive":
        engine = NaiveRAG(strategy=strategy, corpus_tag=corpus_tag)
        is_graph = False
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Keep the benchmark gate closed until the rebuilt graph passes its direct
    # source-set check.
    await _set_neo4j_snapshot_state(
        engine,
        strategy,
        corpus_tag or "default",
        corpus_manifest,
        "in_progress",
    )

    # Reconcile the graph to the current directory; per-document writes
    # atomically replace changed or shortened files.
    if hasattr(engine, "reconcile_dataset_files"):
        await engine.reconcile_dataset_files(files)

    # Cap how many files can sit in the post-parse chunking+LLM pipeline
    # simultaneously. Each file can fan out one generation task per chunk;
    # VLLMClient applies one process-wide-per-event-loop request semaphore,
    # while this outer gate bounds resident document/chunk state.
    file_concurrency = max(1, int(os.environ.get("RAG_MAX_PARALLEL_FILES", "16")))
    file_semaphore = asyncio.Semaphore(file_concurrency)
    progress = {"started": 0, "completed": 0, "lock": asyncio.Lock()}
    failed_files = []
    stats = {"succeeded": 0}
    progress_step = max(1, int(os.environ.get("RAG_PROGRESS_LOG_STEP", "1")))

    async def _log_progress(stage: str, filename: str):
        async with progress["lock"]:
            done = progress["completed"]
            started = progress["started"]
            failed = len(failed_files)
            total = len(files)
            remaining = total - done
            if stage == "start":
                if started % progress_step == 0 or started == total:
                    logger.info(
                        "Indexing progress | started=%d/%d completed=%d failed=%d remaining=%d | now: %s",
                        started,
                        total,
                        done,
                        failed,
                        remaining,
                        filename,
                    )
            else:
                if done % progress_step == 0 or done == total:
                    logger.info(
                        "Indexing progress | completed=%d/%d failed=%d remaining=%d | finished: %s",
                        done,
                        total,
                        failed,
                        remaining,
                        filename,
                    )

    async def process_file(filename: str, content: str, prepared_pages: dict | None = None):
        async with file_semaphore:
            async with progress["lock"]:
                progress["started"] += 1
            await _log_progress("start", filename)

            try:
                if is_graph:
                    knowledge = await engine.extract_knowledge(content, source=filename, prepared_pages=prepared_pages)
                    await engine.build_graph(knowledge, source=filename, document_filename=filename)
                async with progress["lock"]:
                    stats["succeeded"] += 1
                    progress["completed"] += 1
            except Exception as exc:  # noqa: BLE001 - isolate and report each document failure
                logger.error("Failed to index file %s: %s", filename, exc)
                async with progress["lock"]:
                    failed_files.append({"item": filename, "stage": "index", "error": str(exc)})
                    progress["completed"] += 1
            await _log_progress("done", filename)

    # Bound resident file contents and scheduled tasks independently from the
    # active file-processing limit. Completed slots are reused immediately, so
    # one long document cannot hold a full batch barrier.
    default_schedule_batch = 32 if not is_graph else file_concurrency * 2
    schedule_batch = max(
        file_concurrency,
        int(os.environ.get("RAG_FILE_SCHEDULE_BATCH", str(default_schedule_batch))),
    )
    parse_worker_cap = max(1, int(os.environ.get("RAG_PARSE_WORKERS", str(min(8, os.cpu_count() or 4)))))
    parse_pool = ProcessPoolExecutor(max_workers=parse_worker_cap, mp_context=_PARSE_MP_CTX) if is_graph else None
    loop = asyncio.get_running_loop()
    pending_tasks: dict[asyncio.Task[None], str] = {}

    def record_task_errors(errors: list[tuple[str, Exception]]) -> None:
        for filename, exc in errors:
            logger.error("Unhandled indexing task error in %s: %s", filename, exc)
            failed_files.append({"item": filename, "stage": "task", "error": str(exc)})

    try:
        offset = 0
        while offset < len(files):
            if is_graph and len(pending_tasks) >= schedule_batch:
                record_task_errors(await _reap_bounded_tasks(pending_tasks, wait_for_one=True))
            available_slots = schedule_batch - len(pending_tasks) if is_graph else schedule_batch
            batch_names = files[offset : offset + available_slots]
            offset += len(batch_names)
            read_results = await asyncio.gather(
                *[
                    asyncio.to_thread(
                        (Path(dataset_path) / filename).read_text,
                        encoding="utf-8",
                    )
                    for filename in batch_names
                ],
                return_exceptions=True,
            )
            file_contents: list[tuple[str, str]] = []
            for filename, result in zip(batch_names, read_results):
                if isinstance(result, (OSError, UnicodeError)):
                    logger.error("Failed to read file %s: %s", filename, result)
                    failed_files.append({"item": filename, "stage": "read", "error": str(result)})
                    async with progress["lock"]:
                        progress["completed"] += 1
                    await _log_progress("done", filename)
                    continue
                if isinstance(result, BaseException):
                    raise result
                file_contents.append((filename, result))

            if not is_graph and file_contents:
                batch_label = f"{file_contents[0][0]}..{file_contents[-1][0]}"
                async with progress["lock"]:
                    progress["started"] += len(file_contents)
                try:
                    indexed = await engine.index_documents(file_contents)
                    if indexed != len(file_contents):
                        raise RuntimeError(f"Naive batch reported {indexed} documents, expected {len(file_contents)}")
                    async with progress["lock"]:
                        stats["succeeded"] += indexed
                        progress["completed"] += indexed
                    logger.info(
                        "Indexing progress | completed=%d/%d failed=%d remaining=%d | batch: %s",
                        progress["completed"],
                        len(files),
                        len(failed_files),
                        len(files) - progress["completed"],
                        batch_label,
                    )
                except Exception as exc:  # noqa: BLE001 - one atomic batch has one failure boundary
                    logger.error("Failed to index Naive document batch %s: %s", batch_label, exc)
                    async with progress["lock"]:
                        failed_files.extend(
                            {"item": filename, "stage": "index_batch", "error": str(exc)}
                            for filename, _content in file_contents
                        )
                        progress["completed"] += len(file_contents)
                continue

            if parse_pool is not None and file_contents:
                async def prepare_file(filename: str, content: str) -> tuple[str, str, dict | None]:
                    try:
                        prepared_pages = await loop.run_in_executor(
                            parse_pool,
                            parse_pages_offline,
                            filename,
                            content,
                        )
                    except Exception as exc:  # noqa: BLE001 - main-process parsing remains available
                        logger.warning(
                            "Page parsing failed for %s; will re-parse in main process: %s",
                            filename,
                            exc,
                        )
                        prepared_pages = None
                    return filename, content, prepared_pages

                parse_tasks = {
                    asyncio.create_task(prepare_file(filename, content)) for filename, content in file_contents
                }
                try:
                    for completed_parse in asyncio.as_completed(parse_tasks):
                        filename, content, prepared_pages = await completed_parse
                        errors = await _submit_bounded_task(
                            pending_tasks,
                            schedule_batch,
                            filename,
                            lambda filename=filename, content=content, prepared_pages=prepared_pages: process_file(
                                filename,
                                content,
                                prepared_pages,
                            ),
                        )
                        record_task_errors(errors)
                finally:
                    for task in parse_tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*parse_tasks, return_exceptions=True)

        record_task_errors(await _reap_bounded_tasks(pending_tasks, wait_for_one=False))
    finally:
        if pending_tasks:
            for task in pending_tasks:
                task.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)
            pending_tasks.clear()
        if parse_pool is not None:
            await asyncio.to_thread(parse_pool.shutdown, wait=True, cancel_futures=True)
    stage_timing["document_pipeline_seconds"] = time.perf_counter() - started_at

    if is_graph:
        graph_flush_started = time.perf_counter()
        try:
            await engine.flush_graph_batch()
        except Exception as exc:  # noqa: BLE001 - aggregate graph finalization failure
            logger.error("Final graph batch flush failed: %s", exc)
            failed_files.append({"item": "__graph_flush__", "stage": "graph_flush", "error": str(exc)})
        stage_timing["graph_flush_seconds"] = time.perf_counter() - graph_flush_started

        # Build evidence edges after all chunks and embeddings are written so every
        # source chunk has the same candidate pool. Strategies that don't
        # use HOP (e.g., naive_rag) won't have this method.
        if not any(item["stage"] == "graph_flush" for item in failed_files) and hasattr(engine, "build_all_hop_edges"):
            hop_started = time.perf_counter()
            try:
                if hasattr(engine, "clear_hop_edges"):
                    # Remove the complete edge set before rebuilding so stale
                    # Q-/Q+ provenance cannot survive changed documents.
                    await engine.clear_hop_edges()
                await engine.build_all_hop_edges()
            except Exception as exc:  # noqa: BLE001 - aggregate post-index HOP failure
                logger.error("HOP edge construction failed: %s", exc)
                failed_files.append({"item": "__hop_edges__", "stage": "hop_edges", "error": str(exc)})
            stage_timing["hop_build_seconds"] = time.perf_counter() - hop_started

    graph_stats = None
    graph_stats_started = time.perf_counter()
    try:
        graph_stats = await _collect_graph_stats(engine, strategy)
    except Exception as exc:  # noqa: BLE001 - Neo4j driver exposes heterogeneous errors
        logger.error("Graph stats collection failed: %s", exc)
        failed_files.append({"item": "__graph_stats__", "stage": "graph_stats", "error": str(exc)})
    stage_timing["graph_stats_seconds"] = time.perf_counter() - graph_stats_started
    if graph_stats is not None:
        quality = graph_stats.get("index_quality")
        if quality is not None and not bool(quality.get("pass")):
            failed_files.append(
                {
                    "item": "__index_quality__",
                    "stage": "index_quality",
                    "error": f"Prehop index quality checks failed: {quality.get('checks')}",
                }
            )

    global_failure = any(
        item["stage"] in {"graph_flush", "hop_edges", "graph_stats", "index_quality"}
        for item in failed_files
    )
    finalized_successes = 0 if global_failure else stats["succeeded"]

    snapshot_metadata = None
    if not failed_files:
        try:
            snapshot_metadata = await _verify_and_publish_neo4j_snapshot(
                engine,
                strategy,
                corpus_tag or "default",
                source_ids,
                corpus_manifest,
            )
        except Exception as exc:  # noqa: BLE001 - integrity mismatch invalidates the result
            logger.error("Active index snapshot validation failed: %s", exc)
            failed_files.append({"item": "__active_snapshot__", "stage": "active_snapshot", "error": str(exc)})

    # Freeze comparable indexing time after finalization and integrity checks;
    # the reporting-only capacity query below is not indexing work.
    stage_timing["total_elapsed_seconds"] = time.perf_counter() - started_at

    index_capacity = None
    if not failed_files:
        try:
            index_capacity = await _collect_index_capacity(
                strategy,
                corpus_tag or "default",
                engine.neo4j,
            )
        except Exception as exc:  # noqa: BLE001 - missing cost evidence invalidates the run artifact
            logger.error("Index capacity measurement failed: %s", exc)
            failed_files.append({"item": "__index_capacity__", "stage": "index_capacity", "error": str(exc)})

    logger.info(
        "Indexing complete for %d files. Success: %d | Failed: %d",
        len(files),
        finalized_successes,
        len(failed_files),
    )
    if failed_files:
        preview = ", ".join([f["item"] for f in failed_files[:10]])
        logger.warning("Indexing failures (up to 10): %s", preview)

        # Persist the full failure list (not just the log preview) so a
        # partial run's errors are queryable/reviewable afterward instead of
        # only living in scrollback. One JSON per (strategy, corpus_tag, run).
        failures_dir = Path("data/index_failures")
        failures_dir.mkdir(parents=True, exist_ok=True)
        failures_path = failures_dir / f"{strategy}_{corpus_tag or 'default'}_{_artifact_run_id()}.json"
        try:
            _write_json(
                failures_path,
                {
                    "run_id": _artifact_run_id(),
                    "index_code_provenance": code_provenance(),
                    "index_policy": _resolved_index_policy(strategy, model_id),
                    "strategy": strategy,
                    "corpus_tag": corpus_tag or "default",
                    "dataset_path": dataset_path,
                    "total_files": len(files),
                    "succeeded": finalized_successes,
                    "processed_before_finalization": stats["succeeded"],
                    "failed": len(failed_files),
                    "failures": failed_files,
                },
            )
            logger.warning("Full failure list written to %s", failures_path)
        except (OSError, TypeError, ValueError) as exc:
            logger.error("Could not write failure log to %s: %s", failures_path, exc)

    # Record structured graph and corpus statistics from the live database.
    # tables (chunk/HOP-edge counts, Q-/Q+ coverage) — queried from the live
    # graph so it's always consistent with what actually landed in Neo4j.
    # Measurement failures make the run incomplete; result tables must not
    # silently report an index whose structural statistics were never read.
    if graph_stats is not None:
        graph_stats["timing_seconds"] = dict(stage_timing)
        logger.info("Graph stats: %s", graph_stats)
        stats_dir = Path("data/index_stats")
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_path = stats_dir / f"{strategy}_{corpus_tag or 'default'}_{_artifact_run_id()}.json"
        try:
            _write_json(
                stats_path,
                {
                    "run_id": _artifact_run_id(),
                    "index_code_provenance": code_provenance(),
                    "index_policy": _resolved_index_policy(strategy, model_id),
                    "strategy": strategy,
                    "corpus_tag": corpus_tag or "default",
                    "dataset_path": dataset_path,
                    "status": "complete" if not failed_files else "failed",
                    "corpus_manifest_fingerprint": (corpus_manifest or {}).get("fingerprint"),
                    "corpus_manifest_paragraph_count": (corpus_manifest or {}).get("paragraph_count"),
                    "active_snapshot": snapshot_metadata,
                    "index_capacity": index_capacity,
                    **graph_stats,
                },
            )
            logger.info("Graph stats written to %s", stats_path)
        except (OSError, TypeError, ValueError) as exc:
            logger.error("Could not write graph stats to %s: %s", stats_path, exc)
    elif strategy == "naive":
        # Naive has no Prehop structural graph statistics, but it still needs
        # a strategy-scoped corpus identity artifact for benchmark provenance.
        stats_dir = Path("data/index_stats")
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_path = stats_dir / f"{strategy}_{corpus_tag or 'default'}_{_artifact_run_id()}.json"
        try:
            _write_json(
                stats_path,
                {
                    "run_id": _artifact_run_id(),
                    "index_code_provenance": code_provenance(),
                    "index_policy": _resolved_index_policy(strategy, model_id),
                    "strategy": strategy,
                    "corpus_tag": corpus_tag or "default",
                    "dataset_path": dataset_path,
                    "status": "complete" if not failed_files else "failed",
                    "timing_seconds": dict(stage_timing),
                    "corpus_manifest_fingerprint": (corpus_manifest or {}).get("fingerprint"),
                    "corpus_manifest_paragraph_count": (corpus_manifest or {}).get("paragraph_count"),
                    "active_snapshot": snapshot_metadata,
                    "index_capacity": index_capacity,
                },
            )
            logger.info("Index provenance stats written to %s", stats_path)
        except (OSError, TypeError, ValueError) as exc:
            logger.error("Could not write index provenance stats to %s: %s", stats_path, exc)

    elapsed_seconds = stage_timing["total_elapsed_seconds"]
    logger.info(
        "Indexing timing | elapsed_seconds=%.3f files_per_second=%.3f",
        elapsed_seconds,
        len(files) / elapsed_seconds if elapsed_seconds else 0.0,
    )

    if failed_files:
        raise RuntimeError(
            f"Indexing completed with {len(failed_files)} failure(s); see data/index_failures for details"
        )
