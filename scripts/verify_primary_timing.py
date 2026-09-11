"""Measure paired connection handling with native owner-HOP bridge hydration.

This command reuses an existing isolated timing store without modifying graph
state. Historical primary edges and both timing arms are checked separately.
"""
import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def install_checks():
    from models.prehop import connection_timing as timing
    from scripts import prehop_connection_timing as runner

    original_expand = timing.expand
    original_replay = runner.replay
    native = {}
    previous = {}
    count = 0

    async def hydrate(engine, pairs, excluded):
        if not pairs:
            return []
        rows = await engine.retry_query(
            f"UNWIND $pairs AS p MATCH (src:{engine.chunk_label} {{id:p.source_id}}), "
            f"(n:{engine.chunk_label} {{id:p.id}}) WHERE NOT n.id IN $excluded "
            "RETURN p.source_id AS source_id,n.id AS id,n.title AS title,n.sent_id AS sent_id,"
            "n.page AS page,n.text AS text,n.source AS source,n.author AS author,n.publisher AS publisher,"
            "n.published_at AS published_at,n.category AS category,n.url AS url,n.embedding AS embedding,"
            "'hop' AS path_type,p.activated_question_ids AS activated_question_ids,"
            f"[(src)-[:HAS_Q_PLUS]->(q:{engine.q_plus_label}) "
            "WHERE q.id IN p.activated_question_ids | {id:q.id,embedding:q.embedding}] AS bridge_info "
            "ORDER BY source_id,id", {"pairs": pairs, "excluded": sorted(excluded)})
        for row in rows:
            row["bridge_info"] = sorted(row["bridge_info"], key=lambda q: q["id"])
            row["bridge_embeddings"] = [q["embedding"] for q in row["bridge_info"]]
        return rows

    async def checked_expand(engine, starts, excluded, mode, store, namespace):
        nonlocal count
        rows, stats = await original_expand(engine, starts, excluded, mode, store, namespace)
        identities = {(r["source_id"], r["id"]): tuple(sorted(r["activated_question_ids"])) for r in rows}
        expected = {pair: ids for pair, ids in native.items() if pair[0] in starts and pair[1] not in excluded}
        stats["native_destination_and_question_identity_agreement"] = identities == expected
        stats["missing_bridge_nodes"] = sum(len(r["bridge_info"]) != len(set(r["activated_question_ids"])) for r in rows)
        # Exact value comparison occurs outside measured connection handling.
        bridges = {(r["source_id"], r["id"]): r["bridge_info"] for r in rows}
        destinations = {(r["source_id"], r["id"]): r["embedding"] for r in rows}
        if previous:
            stats["paired_bridge_and_destination_vector_agreement"] = (
                previous["bridges"] == bridges and previous["destinations"] == destinations)
            previous.clear()
            count += 1
            if count % 100 == 0:
                print(f"Completed paired primary connection queries: {count}", flush=True)
        else:
            previous.update(bridges=bridges, destinations=destinations)
        return rows, stats

    async def checked_replay(engine, activations, store, namespace, repetitions=1, warmups=0, groups=None, exclusions=None):
        if repetitions != 1 or warmups != 0:
            raise ValueError("Primary timing uses exactly one execution per arm and no warmups")
        starts = sorted({s for values in activations.values() for s in values})
        rows = await engine.retry_query(
            f"MATCH (s:{engine.chunk_label})-[r:HOP_ANSWER]->(d:{engine.chunk_label}) "
            "WHERE s.id IN $starts RETURN s.id AS source,d.id AS destination,"
            "r.source_question_ids AS question_ids", {"starts": starts})
        for row in rows:
            native[(row["source"], row["destination"])] = tuple(sorted(row["question_ids"] or []))
        result = await original_replay(engine, activations, store, namespace, repetitions, warmups, groups, exclusions)
        arms = [arm for row in result["details"] for arm in row["arms"].values()]
        checks = [arm["paired_bridge_and_destination_vector_agreement"] for arm in arms
                  if "paired_bridge_and_destination_vector_agreement" in arm]
        result["bridge_hydration"] = "native owner-HOP source-question embeddings, both arms"
        result["paired_bridge_and_destination_vector_agreement"] = len(checks) == len(activations) and all(checks)
        result["native_destination_and_question_identity_agreement"] = all(
            arm["native_destination_and_question_identity_agreement"] for arm in arms)
        result["missing_bridge_nodes"] = sum(arm["missing_bridge_nodes"] for arm in arms)
        result["verification_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["comparison_interpretation"] = (
            "Primary-activation matched connection timing; not measured end-to-end latency"
            if result["native_destination_and_question_identity_agreement"]
            else "Matched resolver timing; reconstructed connections differ from historical primary edges")
        return result

    timing.hydrate = hydrate
    timing.expand = checked_expand
    runner.replay = checked_replay
    return runner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-stats", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--reference-inputs", type=Path, required=True)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.mode, a.repetitions, a.warmups = "replay", 1, 0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        p.error("Output already exists")
    asyncio.run(install_checks().run(a))
    print(json.dumps({"output": str(a.output), "status": "completed"}), flush=True)


if __name__ == "__main__":
    main()
