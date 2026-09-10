"""Body-to-body ANN links with a frozen, per-passage reference degree budget."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from core.config import RAGConfig


def node_digest(node):
    fields = {key: node.get(key) for key in ("id", "source", "title", "sent_id", "text", "embedding")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def make_reference(namespace, rows):
    nodes = {}
    for row in rows:
        node_id = row["id"]
        degree = row["degree"]
        nodes[node_id] = {"sha256": node_digest(row), "degree": degree}
    return {"schema": "prehop-body-link-reference-v1", "namespace": namespace, "nodes": nodes}


def load_reference(path):
    reference = json.loads(Path(path).read_text())
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

    reference = load_reference(RAGConfig.BODY_LINK_REFERENCE)
    # Observe body identity and size; do not gate execution on the report.
    counts = Counter()
    seen = set()
    identity_mismatches = []
    degree_rows = []
    async for row in read_body_rows(engine):
        expected = reference["nodes"].get(row["id"])
        if expected is None or expected["sha256"] != node_digest(row):
            identity_mismatches.append(row["id"])
        seen.add(row["id"])
        counts[row["source"]] += 1
    await engine.retry_query("CALL db.awaitIndexes($timeout_seconds)", {"timeout_seconds": 300})
    async for row in read_body_rows(engine):
        degree = reference["nodes"][row["id"]]["degree"]
        if not degree:
            degree_rows.append({"id":row["id"],"requested":0,"actual":0})
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
        degree_rows.append({"id":row["id"],"requested":degree,"actual":len(candidates)})
        await engine.retry_query(
            f"UNWIND $edges AS edge MATCH (src:{engine.chunk_label} {{id: $id}}), "
            f"(dst:{engine.chunk_label} {{id: edge.id}}) "
            "WHERE src.source <> dst.source MERGE (src)-[h:HOP_ANSWER]->(dst) "
            "SET h.type='body_to_body', h.direct_channels=['body'], h.score=edge.score, "
            "h.source_question_ids=[], h.source_question_texts=[]",
            {"id": row["id"], "edges": candidates},
        )

    engine.body_link_diagnostics = {
        "passages":len(seen),"identity_mismatch_ids":identity_mismatches,
        "degree_matched_passages":sum(r["requested"]==r["actual"] for r in degree_rows),
        "degree_mismatch_passages":[r for r in degree_rows if r["requested"]!=r["actual"]],
        "requested_edges":sum(r["requested"] for r in degree_rows),
        "actual_edges":sum(r["actual"] for r in degree_rows),
        "interpretation":"observations only; mismatch weakens the matched-degree contrast"}
