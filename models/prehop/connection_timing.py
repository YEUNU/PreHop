"""Matched destination resolver for explicit connection-timing ablations.

Both arms hydrate the same destination metadata from Neo4j. The precomputed
arm reads a separately built SQLite edge table, never historical HOP_ANSWER.
Online matching has no cross-query destination cache.
"""
import hashlib
import json
import sqlite3
import time
from pathlib import Path

CONTRACT = "prehop-connection-timing-v1"


async def graph_fingerprint(engine):
    """Bind bodies, representations, ownership and ANN configuration, paged."""
    digest = hashlib.sha256()
    after = ""
    while True:
        rows = await engine.retry_query(
            f"MATCH (c:{engine.chunk_label}) WHERE c.id > $after WITH c ORDER BY c.id LIMIT 64 "
            "CALL (c) { OPTIONAL MATCH (c)-[r:HAS_Q_PLUS|HAS_Q_MINUS]->(q) "
            "WITH r,q ORDER BY type(r),q.id RETURN collect(CASE WHEN q IS NULL THEN null ELSE "
            "{kind:type(r),properties:properties(q)} END) AS questions } "
            f"CALL (c) {{ OPTIONAL MATCH (c)-[:NEXT]->(n:{engine.chunk_label}) "
            "WITH n ORDER BY n.id RETURN collect(n.id) AS next_ids } "
            "RETURN c.id AS id,properties(c) AS body,questions,next_ids ORDER BY id", {"after":after})
        if not rows:
            break
        for row in rows:
            digest.update(json.dumps(row,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()+b"\n")
        after = rows[-1]["id"]
    indexes = await engine.retry_query("SHOW INDEXES YIELD name,type,labelsOrTypes,properties,options "
        "WHERE name IN $names RETURN name,type,labelsOrTypes,properties,options ORDER BY name",
        {"names":[engine.q_minus_vector_index,engine.q_plus_vector_index,engine.body_vector_index]})
    if len(indexes) != 3:
        raise ValueError("Frozen timing graph requires all three vector indexes")
    digest.update(json.dumps(indexes,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode())
    return digest.hexdigest()


def metadata(path, namespace):
    with sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro",uri=True) as db:
        meta = dict(db.execute("SELECT key,value FROM metadata"))
    if meta.get("contract") != CONTRACT or meta.get("namespace") != namespace or meta.get("status") != "complete":
        raise ValueError("Timing links are not a completed snapshot for this namespace")
    return meta


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
    if {r["id"] for r in rows} != set(starts):
        raise ValueError("Connection activation contains missing passage IDs")
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


def read_pairs(path, starts, namespace):
    path = Path(path).resolve()
    with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as db:
        meta = dict(db.execute("SELECT key,value FROM metadata"))
        if meta.get("contract") != CONTRACT or meta.get("namespace") != namespace or meta.get("status") != "complete":
            raise ValueError("Timing links are not a completed snapshot for this namespace")
        result, matches = [], 0
        for start in sorted(set(starts)):
            row = db.execute("SELECT pairs,matches FROM starts WHERE id=?", (start,)).fetchone()
            if row is None:
                raise ValueError("Precomputed timing snapshot lacks an activation (including zero-link starts)")
            result.extend(json.loads(row[0])); matches += row[1]
    return result, matches


async def expand(engine, starts, excluded, mode, store, namespace):
    begin = time.perf_counter()
    metadata(store, namespace)
    if mode == "online":
        pairs, matches = await resolve_pairs(engine, starts)
    elif mode == "precomputed":
        pairs, matches = read_pairs(store, starts, namespace)
    else:
        raise ValueError("Unknown connection timing arm")
    rows = await hydrate(engine, pairs, excluded)
    return rows, {"arm": mode, "connection_ms": (time.perf_counter()-begin)*1000,
                  "match_requests": matches, "starts": sorted(set(starts)),
                  "destinations": {s: sorted({p["id"] for p in pairs if p["source_id"] == s}) for s in sorted(set(starts))}}


async def build(engine, path, namespace, *, page_size=128):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Use a new timing-link snapshot")
    path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    frozen = await graph_fingerprint(engine)
    build_started = time.perf_counter()
    db = sqlite3.connect(path)
    try:
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT)")
        db.executemany("INSERT INTO metadata VALUES (?,?)", [("contract",CONTRACT),("namespace",namespace),("status","building")])
        db.execute("CREATE TABLE starts (id TEXT PRIMARY KEY,pairs TEXT,matches INTEGER)")
        after, count = "", 0
        while True:
            rows = await engine.retry_query(f"MATCH (c:{engine.chunk_label}) WHERE c.id > $after RETURN c.id AS id ORDER BY id LIMIT $limit",
                                             {"after":after,"limit":page_size})
            if not rows:
                break
            for row in rows:
                pairs, matches = await resolve_pairs(engine, [row["id"]])
                db.execute("INSERT INTO starts VALUES (?,?,?)",(row["id"],json.dumps(pairs,sort_keys=True),matches))
                count += 1
            db.commit(); after = rows[-1]["id"]
            print(f"Timing destinations prepared: {count}", flush=True)
        if not count:
            raise ValueError("Empty question graph")
        construction_seconds = time.perf_counter()-build_started
        if await graph_fingerprint(engine) != frozen:
            raise ValueError("Graph changed during timing-link preparation")
        elapsed = time.perf_counter()-start_time
        db.execute("UPDATE metadata SET value='complete' WHERE key='status'")
        db.executemany("INSERT INTO metadata VALUES (?,?)",[("preparation_seconds",str(elapsed)),("passages",str(count)),("graph_fingerprint",frozen)])
        db.commit()
        return {"contract":CONTRACT,"namespace":namespace,"passages":count,"preparation_seconds":elapsed,
                "connection_build_seconds":construction_seconds,"verification_seconds":elapsed-construction_seconds}
    finally:
        db.close()
