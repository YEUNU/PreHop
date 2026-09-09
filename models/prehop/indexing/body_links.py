"""Body-to-body ANN links with a frozen, per-passage reference degree budget."""

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from core.config import RAGConfig


def node_digest(node):
    fields = {key: node.get(key) for key in ("id", "source", "title", "sent_id", "text", "embedding")}
    vector = fields["embedding"]
    if not vector or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
        raise ValueError("Reference requires finite, nonempty body embeddings")
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def make_reference(namespace, rows):
    if not re.fullmatch(r"[A-Za-z0-9_]+", namespace):
        raise ValueError("Invalid reference namespace")
    nodes = {}
    for row in rows:
        node_id = row["id"]
        degree = row["degree"]
        if not node_id or node_id in nodes or type(degree) is not int or not 0 <= degree <= 3:
            raise ValueError("Invalid reference node identity or degree")
        nodes[node_id] = {"sha256": node_digest(row), "degree": degree}
    if not nodes:
        raise ValueError("Reference graph is empty")
    return {"schema": "prehop-body-link-reference-v1", "namespace": namespace, "nodes": nodes}


def load_reference(path):
    if not path:
        raise ValueError("RAG_BODY_LINK_REFERENCE is required")
    reference = json.loads(Path(path).read_text())
    if reference.get("schema") != "prehop-body-link-reference-v1" or not reference.get("nodes"):
        raise ValueError("Invalid body-link reference manifest")
    if not re.fullmatch(r"[A-Za-z0-9_]+", reference.get("namespace", "")):
        raise ValueError("Invalid reference namespace")
    for value in reference["nodes"].values():
        if type(value.get("degree")) is not int or not 0 <= value["degree"] <= 3:
            raise ValueError("Invalid reference degree")
        if not re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", "")):
            raise ValueError("Invalid reference body digest")
    return reference


async def read_body_rows(engine, *, degrees=False):
    # Embeddings are paged to bound the driver response size.
    after = ""
    while True:
        degree_clause = (
            f"OPTIONAL MATCH (c)-[:HOP_ANSWER]->(t:{engine.chunk_label}) WITH c, count(DISTINCT t) AS degree"
            if degrees
            else "WITH c, 0 AS degree"
        )
        rows = await engine.retry_query(
            f"MATCH (c:{engine.chunk_label}) WHERE c.id > $after "
            f"WITH c ORDER BY c.id LIMIT $limit {degree_clause} "
            "RETURN c.id AS id, c.source AS source, c.title AS title, "
            "c.sent_id AS sent_id, c.text AS text, c.embedding AS embedding, degree ORDER BY id",
            {"after": after, "limit": 64},
        )
        if not rows:
            break
        for row in rows:
            yield row
        after = rows[-1]["id"]


async def build_body_links(engine):
    from core.prehop_ablation import validate_profile

    validate_profile(RAGConfig)
    reference = load_reference(RAGConfig.BODY_LINK_REFERENCE)
    if engine._safe_corpus == reference["namespace"] or not engine._safe_corpus.startswith("ablation_"):
        raise ValueError("Refusing body-link writes to the reference/primary namespace")
    # Verify the entire input before the first link write. Retain only counts,
    # not the high-dimensional embeddings, between the two streaming passes.
    counts = Counter()
    seen = set()
    async for row in read_body_rows(engine):
        expected = reference["nodes"].get(row["id"])
        if expected is None or node_digest(row) != expected["sha256"]:
            raise ValueError(f"Body snapshot differs from reference: {row['id']}")
        seen.add(row["id"])
        counts[row["source"]] += 1
    if seen != set(reference["nodes"]):
        raise ValueError("Body snapshot node set differs from reference")
    await engine.retry_query("CALL db.awaitIndexes($timeout_seconds)", {"timeout_seconds": 300})
    async for row in read_body_rows(engine):
        degree = reference["nodes"][row["id"]]["degree"]
        if not degree:
            continue
        candidates = await engine.retry_query(
            "CALL db.index.vector.queryNodes($index, $pool, $embedding) YIELD node, score "
            "WHERE node.source <> $source AND node.id <> $id "
            "RETURN node.id AS id, score ORDER BY score DESC, id LIMIT $degree",
            {
                "index": engine.body_vector_index,
                "pool": min(len(seen), counts[row["source"]] + degree),
                "embedding": row["embedding"],
                "source": row["source"],
                "id": row["id"],
                "degree": degree,
            },
        )
        if len(candidates) != degree or len({item["id"] for item in candidates}) != degree:
            raise RuntimeError(f"ANN cannot satisfy frozen out-degree for {row['id']}")
        await engine.retry_query(
            f"UNWIND $edges AS edge MATCH (src:{engine.chunk_label} {{id: $id}}), "
            f"(dst:{engine.chunk_label} {{id: edge.id}}) "
            "WHERE src.source <> dst.source MERGE (src)-[h:HOP_ANSWER]->(dst) "
            "SET h.type='body_to_body', h.direct_channels=['body'], h.score=edge.score, "
            "h.source_question_ids=[], h.source_question_texts=[]",
            {"id": row["id"], "edges": candidates},
        )
