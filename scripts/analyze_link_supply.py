"""Evaluate the paper's frozen question-link versus shuffled-link candidate pools.

Keep archived per-query budgets and all sampling realizations unchanged. Gold is
used only to score the frozen pools. Write reports into a new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.ablation_statistics import cluster_interval
from utils.io import atomic_text_writer

ARMS = ("question", "shuffled")
METRICS = ("added_fact_recall", "complete_pool_gain", "any_new_fact")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value: object) -> None:
    with atomic_text_writer(path) as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)


def source(protocol: dict, key: str) -> Path:
    entry = protocol["sources"][key]
    path = Path(entry["path"])
    if digest(path) != entry["sha256"]:
        raise ValueError(f"Source changed: {key}")
    return path


def rewire_edges(edges, sources, seed, attempts_per_edge=10):
    """Directed endpoint swaps preserve both degrees and reject invalid edges."""
    original = set(map(tuple, edges))
    if any(s == t or sources[s] == sources[t] for s, t in original):
        raise ValueError("HOP edges must connect different sources")
    rewired = sorted(original)
    present = set(rewired)
    rng = random.Random(seed)
    accepted = 0
    attempts = attempts_per_edge * len(rewired) if len(rewired) > 1 else 0
    for _ in range(attempts):
        i, j = rng.sample(range(len(rewired)), 2)
        a, b = rewired[i]
        c, d = rewired[j]
        if a == c or b == d or sources[a] == sources[d] or sources[c] == sources[b]:
            continue
        if (a, d) in present or (c, b) in present:
            continue
        present.remove((a, b))
        present.remove((c, d))
        present.update(((a, d), (c, b)))
        rewired[i], rewired[j] = (a, d), (c, b)
        accepted += 1
    return sorted(present), {
        "seed": seed, "attempts": attempts, "accepted_swaps": accepted,
        "original_edges_retained": len(original & present), "edges": len(original),
    }


def additions(starts, base, adjacency):
    return set().union(*(adjacency.get(s, set()) for s in starts)) - set(base)


def coupled_sample(pools, budget, seed):
    """Shared random priorities avoid spurious exclusivity for identical pools."""
    pools = [set(pool) for pool in pools]
    if budget < 0 or any(len(pool) < budget for pool in pools):
        raise ValueError("Candidate budget exceeds an available pool")
    rng = random.Random(seed)
    priority = {node: rng.random() for node in sorted(set().union(*pools))}
    return [sorted(pool, key=lambda node: (priority[node], node))[:budget] for pool in pools]


def evaluate(prepared_dir: Path, protocol: dict, output: Path):
    from utils.metrics import _official_multihoprag_fact_match

    if output.resolve() == prepared_dir.resolve():
        raise ValueError("Reports must not overwrite the archived experiment")
    prepared = json.loads((prepared_dir / "prepared.json").read_text())
    if digest(prepared_dir / "protocol.json") != prepared["protocol_sha256"]:
        raise ValueError("Protocol changed after preparation")
    for name, expected in prepared["files"].items():
        if digest(prepared_dir / name) != expected:
            raise ValueError(f"Prepared candidates changed: {name}")
    snapshot = json.loads((prepared_dir / "snapshot.json").read_text())
    with np.load(prepared_dir / "samples.npz") as archive:
        samples = archive["samples"]
    # The archived experiment sampled three arms jointly. Retain its exact
    # per-query budgets and samples; dropping an arm must never resample others.
    recorded_arms = protocol.get("conditions", ["question", "body", "shuffled"])
    if set(ARMS) - set(recorded_arms) or samples.shape[2] != len(recorded_arms):
        raise ValueError("Prepared candidate arms do not match the protocol")
    samples = samples[:, :, [recorded_arms.index(arm) for arm in ARMS], :]
    queries = {q["_id"]: q for q in json.loads(source(protocol, "queries").read_text())}
    rows = snapshot["rows"]
    if set(queries) != {r["query_id"] for r in rows}:
        raise ValueError("Query populations differ")
    nodes = snapshot["nodes"]
    index_of = {n["id"]: i for i, n in enumerate(nodes)}
    # uint64 bit masks handle all distinct facts in one query, not just one per passage.
    details, random_values = [], []
    for i, row in enumerate(rows):
        query = queries[row["query_id"]]
        facts = list(dict.fromkeys(query["evidence_facts"]))
        if not facts:
            continue  # Official evidence-bearing population; all null IDs reported below.
        if len(facts) > 63:
            raise ValueError("Too many facts for the evidence bit mask")
        all_mask = (1 << len(facts)) - 1
        relevant = set(samples[:, i].ravel().tolist()) - {-1}
        relevant.update(index_of[n] for key in ("base", "question") for n in row[key])
        masks = np.zeros(len(nodes) + 1, dtype=np.uint64)  # Last entry makes padding -1 neutral.
        for j in relevant:
            masks[j] = sum(1 << f for f, fact in enumerate(facts)
                           if _official_multihoprag_fact_match(fact, nodes[j]["text"]))
        base = int(np.bitwise_or.reduce(masks[[index_of[n] for n in row["base"]]], initial=np.uint64(0)))
        covered = np.bitwise_or.reduce(masks[samples[:, i]], axis=2)
        new = covered & np.uint64(all_mask ^ base)
        values = np.empty((len(samples), len(ARMS), len(METRICS)))
        for repeat in range(len(samples)):
            for arm in range(len(ARMS)):
                recovered = int(new[repeat, arm])
                values[repeat, arm] = (recovered.bit_count() / len(facts),
                                      float((int(covered[repeat, arm]) | base) == all_mask) - float(base == all_mask),
                                      float(recovered != 0))
        missing = {f for f in range(len(facts)) if not base & (1 << f)}
        detail = {"query_id": row["query_id"], "original_query_id": row["original_query_id"],
                  "facts": len(facts), "missing_facts": len(missing), "base_fact_recall": base.bit_count()/len(facts),
                  "base_complete": base == all_mask, "budget": row["budget"],
                  "base_candidates": len(row["base"]), "question_full_added": len(row["question"]),
                  "matched": {arm: dict(zip(METRICS, values.mean(axis=0)[j].tolist(), strict=True))
                              for j, arm in enumerate(ARMS)}}
        details.append(detail)
        random_values.append(values)
        if (i + 1) % 250 == 0:
            print(f"Evaluated candidate coverage {i + 1}/{len(rows)}", flush=True)
    if len(details) != protocol["population"]["eligible_fact_queries"]:
        raise ValueError("Evidence-bearing population differs")
    groups = [r["original_query_id"] for r in details]
    def interval(values):
        return cluster_interval(values, groups, seed=protocol["inference"]["bootstrap_seed"],
                                repeats=protocol["inference"]["resamples"])
    condition = {arm: {key: interval([r["matched"][arm][key] for r in details]) for key in METRICS} for arm in ARMS}
    contrasts = {f"{a}_minus_{b}": {key: interval([r["matched"][a][key] - r["matched"][b][key]
                                                  for r in details]) for key in METRICS}
                 for a, b in (("question", "shuffled"),)}
    realizations = np.mean(np.stack(random_values), axis=0)
    report = {"analysis": protocol["analysis"], "protocol_sha256": digest(prepared_dir / "protocol.json"),
              "evaluated_script_sha256": digest(Path(__file__)), "prepared_queries": len(rows),
              "eligible_queries": len(details), "null_queries": len(rows) - len(details),
              "null_query_ids": [r["query_id"] for r in rows if not queries[r["query_id"]]["evidence_facts"]],
              "baseline": {"fact_recall": interval([r["base_fact_recall"] for r in details]),
                           "complete_pool_coverage": interval([float(r["base_complete"]) for r in details])},
              "budget": {"mean": float(np.mean([r["budget"] for r in details])),
                         "histogram": dict(Counter(r["budget"] for r in details)),
                         "mean_base_candidates": float(np.mean([r["base_candidates"] for r in details]))},
              "conditions": condition, "contrasts": contrasts,
              "realization_variation": {"scope": "SD across fixed sampling/shuffled-graph repetitions; not a CI",
                                        "sd": realizations.std(axis=0, ddof=1).tolist(),
                                        "metrics": METRICS, "arms": ARMS},
              "interpretation": "Pre-selector coverage with archived counts and samples; not final MAP/QA or matched tokens.",
              "prepared_source": str(prepared_dir.resolve()), "details": details}
    save_json(output / "comparison.json", report)
    with atomic_text_writer(output / "summary.csv") as stream:
        stream.write("condition,mean_added_passages,added_fact_recall,complete_pool_gain,any_new_fact\n")
        for arm in ARMS:
            stream.write(f"{arm},{report['budget']['mean']}," +
                         ",".join(str(condition[arm][key]["mean"]) for key in METRICS) + "\n")
    print(json.dumps({k: report[k] for k in ("budget", "conditions", "contrasts")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True,
                        help="Archived protocol, snapshot, samples and hash manifest")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.prepared / "protocol.json").read_text())
    evaluate(args.prepared, protocol, args.output)


if __name__ == "__main__":
    main()
