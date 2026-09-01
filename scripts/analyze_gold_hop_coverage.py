#!/usr/bin/env python3
"""Measure whether materialized HOP edges connect MuSiQue gold paragraphs."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
from pathlib import Path

from neo4j import GraphDatabase


def _source_from_paragraph_id(paragraph_id: str) -> str:
    return paragraph_id.replace(":", "_", 1) + ".txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--corpus-tag", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_]+", args.corpus_tag):
        raise ValueError("--corpus-tag may contain only letters, digits, and underscores")
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]
    chunk_label = f"PR_{args.corpus_tag}_Chunk"
    cypher = (
        f"MATCH (x:{chunk_label})-[:HOP_ANSWER]->(y:{chunk_label}) "
        "RETURN DISTINCT x.source AS source, y.source AS target"
    )
    with GraphDatabase.driver(uri, auth=(user, password)) as driver, driver.session() as session:
        hop_pairs = {(str(row["source"]), str(row["target"])) for row in session.run(cypher)}

    queries = json.loads(args.queries.read_text())
    by_depth: dict[str, list[dict[str, object]]] = {}
    details: list[dict[str, object]] = []
    for row in queries:
        depth = str(row["category"])
        sources = list(dict.fromkeys(_source_from_paragraph_id(str(value)) for value in row["evidence_paragraph_ids"]))
        undirected = {source: set() for source in sources}
        direct_gold_edges = 0
        for source, target in itertools.combinations(sources, 2):
            if (source, target) in hop_pairs or (target, source) in hop_pairs:
                direct_gold_edges += 1
                undirected[source].add(target)
                undirected[target].add(source)
        seen: set[str] = set()
        stack = sources[:1]
        while stack:
            source = stack.pop()
            if source in seen:
                continue
            seen.add(source)
            stack.extend(undirected[source] - seen)
        directed_path_exists = any(
            all((ordering[index], ordering[index + 1]) in hop_pairs for index in range(len(ordering) - 1))
            for ordering in itertools.permutations(sources)
        )
        detail = {
            "query_id": str(row["_id"]),
            "category": depth,
            "direct_gold_edges": direct_gold_edges,
            "any_gold_edge": direct_gold_edges > 0,
            "gold_subgraph_connected": len(seen) == len(sources),
            "directed_path_exists": directed_path_exists,
            "gold_sources": len(sources),
        }
        by_depth.setdefault(depth, []).append(detail)
        details.append(detail)

    result: dict[str, object] = {
        "queries_path": str(args.queries),
        "corpus_tag": args.corpus_tag,
        "unique_hop_source_pairs": len(hop_pairs),
        "definition": (
            "A gold edge exists when at least one HOP_ANSWER relation connects chunks from two gold paragraph files; "
            "direction is ignored for any-edge/connectivity and retained for directed-path existence."
        ),
        "by_depth": {},
        "details": details,
    }
    for depth, rows in sorted(by_depth.items()):
        n = len(rows)
        result["by_depth"][depth] = {
            "queries": n,
            "any_gold_edge_rate": sum(bool(row["any_gold_edge"]) for row in rows) / n,
            "gold_subgraph_connected_rate": sum(bool(row["gold_subgraph_connected"]) for row in rows) / n,
            "directed_path_exists_rate": sum(bool(row["directed_path_exists"]) for row in rows) / n,
            "mean_direct_gold_edges": sum(int(row["direct_gold_edges"]) for row in rows) / n,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
