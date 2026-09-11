"""Matched destination resolver for explicit connection-timing ablations.

Both arms hydrate the same destination metadata from Neo4j. The precomputed
arm reads experiment-scoped Neo4j HOP_TIMING edges, never historical HOP_ANSWER.
Online matching has no cross-query destination cache.
"""
import hashlib
import json
import time
from pathlib import Path

CONTRACT = "prehop-connection-timing-v2"


def metadata(path, namespace=None):
    """Read the experiment pointer; graph data lives only in Neo4j."""
    return json.loads(Path(path).read_text())


async def resolve_pairs(engine, starts):
    starts = sorted(set(starts))
    if not starts:
        return [], 0
    rows = await engine.retry_query(
        f"MATCH (c:{engine.chunk_label}) WHERE c.id IN $starts "
        f"OPTIONAL MATCH (c)-[:HAS_Q_PLUS]->(q:{engine.q_plus_label}) "
        "WITH c,q ORDER BY c.id,q.ordinal,q.id "
        "RETURN c.id AS id,c.source AS source,collect(CASE WHEN q.query_embedding IS NULL "
        "THEN null ELSE {id:q.id,text:q.text,query_embedding:q.query_embedding} END) AS questions",
        {"starts": starts})
    sources = sorted({r["source"] for r in rows})
    counts = await engine.retry_query(
        f"MATCH (c:{engine.chunk_label}) WHERE c.source IN $sources "
        f"MATCH (c)-[:HAS_Q_MINUS]->(q:{engine.q_minus_label}) "
        "RETURN c.source AS source,count(q) AS count", {"sources": sources})
    pools = {r["source"]: int(r["count"]) + 1 for r in counts}
    wave = [{**r, "questions": [q for q in r["questions"] if q],
             "ann_pools": {"q_minus": pools.get(r["source"], 1)}} for r in rows]
    edges = await engine._process_hop_wave(wave)
    pairs = [{"source_id": e["src_id"], "id": e["tgt_id"],
              "activated_question_ids": sorted(e["source_question_ids"])} for e in edges]
    return sorted(pairs, key=lambda e:(e["source_id"],e["id"])), sum(len(r["questions"]) for r in wave)


async def hydrate(engine, pairs, excluded):
    if not pairs:
        return []
    return await engine.retry_query(
        f"UNWIND $pairs AS p MATCH (n:{engine.chunk_label} {{id:p.id}}) "
        "WHERE NOT n.id IN $excluded "
        "RETURN p.source_id AS source_id,n.id AS id,n.title AS title,n.sent_id AS sent_id,"
        "n.page AS page,n.text AS text,n.source AS source,n.author AS author,n.publisher AS publisher,"
        "n.published_at AS published_at,n.category AS category,n.url AS url,n.embedding AS embedding,"
        "'hop' AS path_type,null AS bridge_embeddings,p.activated_question_ids AS activated_question_ids "
        "ORDER BY source_id,id", {"pairs": pairs, "excluded": sorted(excluded)})


async def read_pairs(engine, starts, experiment):
    rows = await engine.retry_query(
        f"MATCH (s:{engine.chunk_label}) WHERE s.id IN $starts "
        "OPTIONAL MATCH (s)-[r:HOP_TIMING {experiment:$experiment}]->(d) "
        "WITH s,r,d ORDER BY s.id,d.id "
        "RETURN s.id AS source_id, collect(CASE WHEN d IS NULL THEN null ELSE "
        "{source_id:s.id,id:d.id,activated_question_ids:r.source_question_ids} END) AS pairs",
        {"starts": sorted(set(starts)), "experiment": experiment})
    return [pair for row in rows for pair in row["pairs"] if pair]


async def expand(engine, starts, excluded, mode, store, namespace):
    # Small pointer loading is setup, not destination resolution.
    meta = metadata(store, namespace)
    begin = time.perf_counter()
    if mode == "online":
        pairs, matches = await resolve_pairs(engine, starts)
    elif mode == "precomputed":
        pairs = await read_pairs(engine, starts, meta["experiment"])
        matches = sum(meta["question_counts"].get(s, 0) for s in set(starts))
    else:
        raise ValueError("Unknown connection timing arm")
    rows = await hydrate(engine, pairs, excluded)
    return rows, {"arm": mode, "connection_ms": (time.perf_counter()-begin)*1000,
                  "match_requests": matches, "starts": sorted(set(starts)),
                  "destinations": {s: sorted({p["id"] for p in pairs if p["source_id"] == s and p["id"] not in excluded}) for s in sorted(set(starts))}}


async def build(engine, path, namespace, *, page_size=128):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    experiment = "timing_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24]
    start_time = time.perf_counter()
    await engine.retry_query("CREATE (m:RAGTimingSnapshot {experiment:$experiment,namespace:$namespace,status:'building'})",
                             {"experiment":experiment,"namespace":namespace})
    after, count, edge_count, question_counts = "", 0, 0, {}
    while True:
        rows = await engine.retry_query(f"MATCH (c:{engine.chunk_label}) WHERE c.id > $after RETURN c.id AS id ORDER BY id LIMIT $limit",
                                         {"after":after,"limit":page_size})
        if not rows:
            break
        for row in rows:
            pairs, matches = await resolve_pairs(engine, [row["id"]])
            await engine.retry_query(
                f"UNWIND $pairs AS p MATCH (s:{engine.chunk_label} {{id:p.source_id}}), (d:{engine.chunk_label} {{id:p.id}}) "
                "CREATE (s)-[r:HOP_TIMING {experiment:$experiment}]->(d) "
                "SET r.source_question_ids=p.activated_question_ids",
                {"pairs":pairs,"experiment":experiment})
            question_counts[row["id"]] = matches
            count += 1
            edge_count += len(pairs)
        after = rows[-1]["id"]
        print(f"Timing destinations prepared: {count}", flush=True)
    elapsed = time.perf_counter()-start_time
    result = {"contract":CONTRACT,"storage":"neo4j","experiment":experiment,"namespace":namespace,
              "passages":count,"edges":edge_count,"question_counts":question_counts,
              "preparation_seconds":elapsed,"connection_build_seconds":elapsed,
              "shared_nodes_and_ann":True,"historical_primary_edges_modified":False}
    await engine.retry_query("MATCH (m:RAGTimingSnapshot {experiment:$experiment}) "
        "SET m.status='complete',m.passages=$passages,m.edges=$edges,m.build_seconds=$seconds",
        {"experiment":experiment,"passages":count,"edges":edge_count,"seconds":elapsed})
    path.write_text(json.dumps(result,sort_keys=True))
    return result
