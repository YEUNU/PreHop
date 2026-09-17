"""Evaluate saved passage lists without retrieval or generation calls.

The manifest maps dataset names to condition records containing ``path``,
``sha256``, and optionally ``sources_field`` (default: retrieved_sources).
Dataset names are multihoprag or hotpotqa. Query annotations and the sentence
store are explicit command-line inputs; source artifacts are never modified.
"""
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from utils.hotpotqa import project_sentences

METRICS = ("Hits@4", "Hits@10", "MRR@10", "MAP@10", "Recall@10")


def rank_metrics(matches, gold_count):
    """MultiHop-RAG rank formula plus distinct-gold recall.

    Each element is the set of gold identities matched in one ranked passage.
    No-evidence queries are ineligible, rather than zero-scoring queries.
    With more than ten gold items the official capped MAP denominator can
    produce values above one; do not silently change that formula.
    """
    if gold_count == 0:
        return None
    seen = set()
    first = None
    weighted = 0.0
    for rank, found in enumerate(matches[:10], 1):
        if found and first is None:
            first = rank
        new = set(found) - seen
        weighted += len(new) / rank
        seen.update(found)
    return dict(zip(METRICS, (
        float(any(matches[:4])), float(first is not None),
        1 / first if first else 0.0,
        weighted / min(gold_count, 10), len(seen) / gold_count,
    ), strict=True))


def multihop_metrics(sources, facts):
    # The benchmark strips literal spaces and newlines, preserving case.
    norm = lambda s: str(s).replace(" ", "").replace("\n", "")
    gold = [norm(f) for f in facts]
    matches = [
        {fact for fact in gold if fact in norm(s.get("text") or s.get("content") or "")}
        for s in sources[:10]
    ]
    return rank_metrics(matches, len(gold))


def hotpot_metrics(sources, gold_facts, project):
    gold = set(map(tuple, gold_facts))
    matches = [gold & set(map(tuple, project(s))) for s in sources[:10]]
    return rank_metrics(matches, len(gold))


def bootstrap(rows, resamples=10000, seed=42):
    """Row-weighted means and original-question cluster percentile intervals."""
    import numpy as np
    groups = {}
    for row in rows:
        groups.setdefault(row["original_query_id"], []).append([row[k] for k in METRICS])
    sums = np.array([np.sum(v, axis=0) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, resamples, 100):
        indices = rng.integers(0, len(groups), (min(100, resamples-start), len(groups)))
        draws.append(sums[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None])
    ci = np.quantile(np.concatenate(draws), [.025, .975], axis=0)
    mean = sums.sum(axis=0) / counts.sum()
    return {k: {"mean": float(mean[i]), "ci95": ci[:, i].tolist()} for i, k in enumerate(METRICS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--multihop-queries", type=Path, required=True)
    parser.add_argument("--hotpot-queries", type=Path, required=True)
    parser.add_argument("--sentence-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    inputs = [args.manifest, args.multihop_queries, args.hotpot_queries, args.sentence_store]
    hashes = {str(p): sha(p) for p in inputs}
    queries = {dataset: {q["_id"]: q for q in json.loads(path.read_text())}
               for dataset, path in [("multihoprag", args.multihop_queries), ("hotpotqa", args.hotpot_queries)]}

    @lru_cache(maxsize=100000)
    def project(serialized):
        return project_sentences([json.loads(serialized)], args.sentence_store)

    out = {"definition": "MultiHop-RAG ranking formula; HotpotQA title/sentence-ID adaptation; distinct evidence Recall@10",
           "resamples": 10000, "seed": 42, "inputs": hashes, "datasets": {}}
    for dataset, conditions in json.loads(args.manifest.read_text()).items():
        out["datasets"][dataset] = {}
        for name, info in conditions.items():
            path = Path(info["path"])
            assert sha(path) == info["sha256"], path
            raw = path.read_text()
            data = [json.loads(line) for line in raw.splitlines()] if path.suffix == ".jsonl" else json.loads(raw)
            rows = data["details"] if isinstance(data, dict) else data
            assert len(rows) == len(queries[dataset])
            assert {r["query_id"] for r in rows} == set(queries[dataset])
            details = []
            for row in rows:
                q = queries[dataset][row["query_id"]]
                sources = row[info.get("sources_field", "retrieved_sources")]
                if dataset == "multihoprag":
                    facts = q.get("evidence_facts", [])
                    metrics = multihop_metrics(sources, facts)
                else:
                    facts = q["supporting_facts"]
                    metrics = hotpot_metrics(sources, facts, lambda s: project(json.dumps(s, sort_keys=True)))
                if metrics is None:
                    continue
                details.append({"query_id": row["query_id"], "original_query_id": q.get("original_query_id") or row["query_id"],
                                "gold_count": len(facts), "passages@10": len(sources[:10]), **metrics})
            result = {**info, "rows": len(details), "clusters": len({r["original_query_id"] for r in details}),
                      "metrics": bootstrap(details), "details": details}
            if dataset == "multihoprag":
                result["by_fact_count"] = {str(n): bootstrap([r for r in details if r["gold_count"] == n]) for n in [2, 3, 4]}
            out["datasets"][dataset][name] = result
            assert sha(path) == info["sha256"]
            print(dataset, name, {k: round(v["mean"], 6) for k, v in result["metrics"].items()}, flush=True)
    assert all(sha(Path(p)) == value for p, value in hashes.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
