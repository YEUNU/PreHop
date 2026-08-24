#!/usr/bin/env python
"""Build a stratified query sample (balanced by category/question_type) for
any active multi-hop-shaped dataset (multihoprag, musique).

Uses one implementation for MultiHop-RAG and MuSiQue so the
datasets do not need near-duplicate sampling scripts. Graph baselines are slow
(hoprag ~160s/query), so k-fold figures run on a balanced sample instead of
the full query set. Equal count per category with a fixed seed keeps the
sample reproducible and each fold/category balanced.

    python data/make_sample.py --dataset musique --per-type 50    # n=150 (3 hop counts)
    python data/make_sample.py --dataset multihoprag --per-type 50 --seed 42

Output: data/<dataset>_sample<N>_queries.json (N = per_type * num_categories).
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["multihoprag", "musique"],
        help="which dataset's full query file to sample from",
    )
    ap.add_argument("--per-type", type=int, default=50, help="queries sampled per category")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible)")
    ap.add_argument("--out", default=None, help="output path (default derives from total n)")
    args = ap.parse_args()

    full_path = DATA_DIR / f"{args.dataset}_queries.json"
    if not full_path.exists():
        print(f"ERROR: {full_path} not found — run data/prepare_{args.dataset}.py first.")
        return 2

    with open(full_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    by_type: dict[str, list] = defaultdict(list)
    for q in queries:
        by_type[q.get("question_type", "unknown")].append(q)

    rng = random.Random(args.seed)
    sample = []
    for qtype in sorted(by_type):
        pool = sorted(by_type[qtype], key=lambda q: str(q.get("_id", "")))  # stable order before sampling
        if len(pool) < args.per_type:
            print(f"WARN: {qtype} has only {len(pool)} (< {args.per_type}); taking all.")
            picked = pool
        else:
            picked = rng.sample(pool, args.per_type)
        sample.extend(picked)
        print(f"  {qtype}: {len(picked)}")

    total = len(sample)
    out_path = Path(args.out) if args.out else (DATA_DIR / f"{args.dataset}_sample{total}_queries.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"Wrote {total} queries (seed={args.seed}) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
