"""Audit gold-evidence accessibility within frozen, token-matched retrieval pools.

Intermediate prefixes measure candidate supply only. They have no new selector
or reader outputs and are not asserted to match intermediate token budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.benchmark_failures import metric_value
from scripts.ablation_statistics import cluster_interval
from scripts.campaign_runtime import atomic_json
from utils.hotpotqa import project_sentences
from utils.metrics import _official_multihoprag_fact_match

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "data/results/paper-pool-replay-20260930-b5"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def indexed(nodes: list[dict]) -> dict[str, dict]:
    result = {str(node["id"]): node for node in nodes}
    if len(result) != len(nodes):
        raise ValueError("Repeated candidate identity")
    return result


def content(node: dict) -> tuple:
    return (node.get("source"), node.get("page"), node.get("sent_id"),
            node.get("title") or node.get("doc"), node["text"])


def prefix_ids(ordered_ids: list[str], starts: set[str], fraction: float) -> list[str]:
    """Retain every start and the first fraction of non-starts, in saved order."""
    if not 0 <= fraction <= 1:
        raise ValueError("Prefix fraction must lie in [0, 1]")
    if len(set(ordered_ids)) != len(ordered_ids) or not starts <= set(ordered_ids):
        raise ValueError("Candidates must be unique and contain every initial start")
    extras = [nid for nid in ordered_ids if nid not in starts]
    keep = starts | set(extras[:math.floor(fraction * len(extras))])
    return [nid for nid in ordered_ids if nid in keep]


def coverage(ids, hits: dict[str, set]) -> set:
    return set().union(*(hits[nid] for nid in ids))


def evidence_witnesses(starts: set[str], graph: set[str], direct: set[str], hits: dict[str, set]) -> dict:
    """Partition graph-new gold units using actual supporting passage identities."""
    if not starts <= graph & direct:
        raise ValueError("Both pools must retain the initial starts")
    new = coverage(graph, hits) - coverage(starts, hits)
    common, graph_only = graph & direct, graph - direct
    rows = []
    for unit in sorted(new):
        gw = sorted(nid for nid in graph if unit in hits[nid])
        dw = sorted(nid for nid in direct if unit in hits[nid])
        cw = sorted(common & set(gw))
        category = "shared_id" if cw else "distinct_only" if dw else "graph_exclusive"
        rows.append({"unit": unit, "category": category, "graph_witnesses": gw,
                     "direct_witnesses": dw, "common_witnesses": cw,
                     "graph_only_witnesses": sorted(graph_only & set(gw))})
    counts = {key: sum(r["category"] == key for r in rows)
              for key in ("shared_id", "distinct_only", "graph_exclusive")}
    counts["shared_id_with_graph_only_witness"] = sum(
        r["category"] == "shared_id" and bool(r["graph_only_witnesses"]) for r in rows)
    counts["graph_new_units"] = len(new)
    counts["graph_only_passages"] = len(graph_only)
    counts["graph_only_gold_bearing_passages"] = sum(bool(hits[nid]) for nid in graph_only)
    counts["graph_only_new_gold_bearing_passages"] = sum(bool(hits[nid] & new) for nid in graph_only)
    return {"counts": counts, "units": rows}


def paired_scores(left: dict, right: dict, queries: dict, metric: str, *, seed: int, repeats: int) -> dict:
    """Strict aggregation: an unavailable metric cannot silently shrink a population."""
    ids = sorted(queries)
    if not ids:
        raise ValueError("Empty score population")
    a, b = [], []
    for qid in ids:
        x, y = metric_value(left[qid], metric), metric_value(right[qid], metric)
        if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Unavailable required metric {metric}: {qid}")
        a.append(x)
        b.append(y)
    interval = cluster_interval([x - y for x, y in zip(a, b, strict=True)],
                                [queries[q].get("original_query_id") or q for q in ids],
                                seed=seed, repeats=repeats)
    return {"graph": sum(a) / len(a), "direct": sum(b) / len(b), "graph_minus_direct": interval}


def reader_sensitivity(root: Path, protocol: dict) -> dict:
    """Reaggregate saved scores; exclude every occurrence of development questions."""
    development = read(root / protocol["development_groups"])["datasets"]
    comparisons = {}
    for name, spec in protocol["reader_comparisons"].items():
        dataset = spec["dataset"]
        query_path = root / protocol["datasets"][dataset]["queries"]
        if digest(query_path) != development[dataset]["queries_sha256"]:
            raise ValueError("Reader development/query population changed")
        queries = {q["_id"]: q for q in read(query_path)}
        dev_ids = set(development[dataset]["original_question_ids"])
        excluded = {qid for qid, q in queries.items() if (q.get("original_query_id") or qid) in dev_ids}
        if excluded != set(development[dataset]["released_query_ids"]):
            raise ValueError("Development groups do not include all released occurrences")
        arms = []
        for arm in ("prehop_replay", spec["right"]):
            raw = read(root / spec["directory"] / f"{arm}.json")["details"]
            rows = {r["query_id"]: r for r in raw}
            if len(rows) != len(raw) or set(rows) != set(queries):
                raise ValueError("Saved reader population mismatch or duplicate query IDs")
            arms.append(rows)
        result = {"excluded_ids": sorted(excluded), "metrics": {}}
        for metric, population in spec["metrics"].items():
            full = {qid: q for qid, q in queries.items()
                    if population == "all" or q["question_type"] != "null_query"}
            retained = {qid: q for qid, q in full.items() if qid not in excluded}
            result["metrics"][metric] = {
                label: paired_scores(*arms, rows, metric, seed=protocol["uncertainty"]["seed"],
                                     repeats=protocol["uncertainty"]["resamples"])
                for label, rows in (("full", full), ("development_excluded", retained))}
        comparisons[name] = result
    return comparisons


def row_curves(starts: set[str], graph: list[str], direct: list[str], hits: dict[str, set],
               gold: set, fractions: list[float]) -> list[dict]:
    if not gold or not starts <= set(graph):
        raise ValueError("Nonempty gold and preservation of initial starts are required")
    initial = coverage(starts, hits)
    graph_units = coverage(graph, hits)
    if not set().union(*hits.values()) <= gold:
        raise ValueError("Passage hits must contain only evaluation gold units")
    new_units = graph_units - initial
    graph_added = set(graph) - starts
    rows = []
    for fraction in fractions:
        ids = prefix_ids(direct, starts, fraction)
        units = coverage(ids, hits)
        rows.append({
            "fraction": fraction, "candidates": len(ids), "added_candidates": len(ids) - len(starts),
            "direct_recall": len(units) / len(gold), "graph_recall": len(graph_units) / len(gold),
            "graph_exclusive_recall": len(graph_units - units) / len(gold),
            "direct_exclusive_recall": len(units - graph_units) / len(gold),
            "new_graph_units": len(new_units), "new_graph_accessible": len(new_units & units),
            "new_graph_unavailable": len(new_units - units),
            "graph_added_passages": len(graph_added), "graph_added_in_direct": len(graph_added & set(ids)),
        })
    return rows


def analyze(root: Path, run: Path) -> dict:
    protocol_path = run / "protocol.json"
    protocol = read(protocol_path)
    if (run / "analysis.json").exists():
        raise ValueError("Preserve prior results; choose a new run directory")
    for name, expected in protocol["inputs"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Source changed: {name}")
    manifest = read(root / ARCHIVE / "historical-source-manifest.json")
    prior = read(root / ARCHIVE / "replayed-pools.json")
    verified = {}

    def frozen(name):
        path = root / name
        actual = digest(path)
        if actual != manifest[name]:
            raise ValueError(f"Archived source changed: {name}")
        verified[name] = actual
        return read(path)

    sentence_store = "data/hotpotqa_corpus/sentences.sqlite3"
    if digest(root / sentence_store) != manifest[sentence_store]:
        raise ValueError("HotpotQA corpus sentence mapping changed")
    verified[sentence_store] = manifest[sentence_store]
    sentence_cache = {}
    datasets = {}
    for dataset, spec in protocol["datasets"].items():
        queries = {q["_id"]: q for q in frozen(spec["queries"])}
        previous = {r["query_id"]: r for r in prior["datasets"][dataset]["details"]}
        expected_nulls = set(prior["datasets"][dataset]["excluded_null_ids"])
        if set(queries) != set(previous) | expected_nulls:
            raise ValueError("Query populations differ from the audited endpoint report")
        input_ids = {p.stem for p in (root / spec["budget_dir"] / "inputs").glob("*.json")}
        if input_ids != set(queries):
            raise ValueError("Prepared input population differs from released queries")
        rows, excluded = [], []
        for number, (qid, query) in enumerate(sorted(queries.items()), 1):
            base = spec["budget_dir"]
            source = frozen(f"{base}/source/{qid}.json")
            prepared = frozen(f"{base}/inputs/{qid}.json")
            if prepared["source_input_sha256"] != verified[f"{base}/source/{qid}.json"]:
                raise ValueError("Prepared/source identity mismatch")
            if prepared["query_id"] != qid or source["query_id"] != qid:
                raise ValueError("Query ID mismatch")
            initial = indexed(source["direct"])
            pools = {arm: indexed(prepared["pools"][arm]) for arm in ("prehop_replay", "direct_tokens")}
            if list(pools["prehop_replay"]) != [str(n["id"]) for n in source["expanded"]]:
                raise ValueError("Graph pool differs from the frozen expansion")
            nodes = {}
            for group in [initial, *pools.values()]:
                for nid, node in group.items():
                    if nid in nodes and content(nodes[nid]) != content(node):
                        raise ValueError("Conflicting passage content for one identity")
                    nodes[nid] = node
            if dataset == "multihoprag":
                facts = query.get("evidence_facts") or []
                if not facts:
                    if query["question_type"] != "null_query" or qid not in expected_nulls:
                        raise ValueError("Unexpected missing evaluation facts")
                    excluded.append(qid)
                    continue
                gold = set(range(len(facts)))
                hits = {nid: {i for i, fact in enumerate(facts)
                              if _official_multihoprag_fact_match(fact, n["text"])} for nid, n in nodes.items()}
            else:
                gold = set(map(tuple, query["supporting_facts"]))
                hits = {}
                for nid, node in nodes.items():
                    key = content(node)
                    if key not in sentence_cache:
                        sentence_cache[key] = set(map(tuple, project_sentences([node], root / sentence_store)))
                    hits[nid] = sentence_cache[key] & gold
            prior_row = previous[qid]
            for arm, pool in pools.items():
                expected = prior_row["coverage"]["pool"][arm]
                expected = set(map(tuple, expected)) if dataset == "hotpotqa" else set(expected)
                if coverage(pool, hits) != expected:
                    raise ValueError("Canonical coverage differs from independently audited endpoint")
            if len(coverage(initial, hits)) != prior_row["initial_covered"]:
                raise ValueError("Initial evidence coverage changed")
            curves = row_curves(set(initial), list(pools["prehop_replay"]), list(pools["direct_tokens"]),
                                hits, gold, protocol["prefix_fractions"])
            rows.append({"query_id": qid, "original_query_id": query.get("original_query_id") or qid,
                         "gold_units": len(gold), "question_type": query["question_type"], "curve": curves,
                         "witnesses": evidence_witnesses(set(initial), set(pools["prehop_replay"]),
                                                        set(pools["direct_tokens"]), hits)})
            if number % 500 == 0:
                print(dataset, number, "source identities and evidence checked", flush=True)
        if len(rows) != spec["eligible_rows"] or len(excluded) != spec["null_rows"]:
            raise ValueError("Evaluation population changed")
        summaries = []
        for i, fraction in enumerate(protocol["prefix_fractions"]):
            points = [r["curve"][i] for r in rows]
            summary = {"fraction": fraction}
            for metric in ["candidates", "added_candidates"]:
                summary["mean_" + metric] = sum(p[metric] for p in points) / len(points)
            for metric in ["new_graph_units", "new_graph_accessible", "new_graph_unavailable",
                           "graph_added_passages", "graph_added_in_direct"]:
                summary[metric] = sum(p[metric] for p in points)
            summary["questions_with_graph_exclusive_evidence"] = sum(p["new_graph_unavailable"] > 0 for p in points)
            for metric in ["direct_recall", "graph_exclusive_recall", "direct_exclusive_recall"]:
                summary[metric] = cluster_interval(
                    [p[metric] for p in points], [r["original_query_id"] for r in rows],
                    seed=protocol["uncertainty"]["seed"], repeats=protocol["uncertainty"]["resamples"])
            summaries.append(summary)
        datasets[dataset] = {"rows": len(rows), "original_questions": len({r["original_query_id"] for r in rows}),
                             "excluded_null_ids": excluded, "curve": summaries, "details": rows,
                             "witness_counts": {key: sum(r["witnesses"]["counts"][key] for r in rows)
                                                for key in rows[0]["witnesses"]["counts"]},
                             "audited_endpoint": prior["datasets"][dataset]["summary"]}
    atomic_json(run / "verified-inputs.json", verified)
    report = {"protocol_sha256": digest(protocol_path), "script_sha256": digest(Path(__file__)),
              "verified_inputs_sha256": digest(run / "verified-inputs.json"), "datasets": datasets,
              "scope": protocol["prefix_rule"], "new_model_calls": 0}
    if "reader_comparisons" in protocol:
        report["reader_sensitivity"] = reader_sensitivity(root, protocol)
    atomic_json(run / "analysis.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = analyze(args.root.resolve(), args.run.resolve())
    for dataset, values in report["datasets"].items():
        print(dataset, json.dumps(values["curve"][-1]), flush=True)


if __name__ == "__main__":
    main()
