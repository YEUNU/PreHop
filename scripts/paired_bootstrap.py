"""Query-level paired bootstrap: prehop vs each baseline, any of the three
multi-hop datasets (MultiHop-RAG / HotpotQA / MuSiQue).

Per-fold CIs from a single run (see kfold_analysis.py) overlap across
strategies, so they cannot establish whether prehop's lead over a given
baseline is real. This script pairs the per-query scores by query string (all
strategies ran the identical sample file for a dataset), computes the paired
difference (prehop - baseline) per query, and bootstraps the mean diff to a
95% CI. A diff whose CI excludes 0 is a statistically separated win/loss.

- Judge / hallucination: computed over judged rows (sentinel -1 dropped); both
  sides of a pair must be valid. hallucination is lower-is-better.
- Retrieval metrics (mrr@10/map@10/hits@4/hits@10/evidence_doc_recall): a
  query with no gold evidence (e.g. MultiHop-RAG's null_query category, where
  every strategy scores 0 by construction) is excluded — the diff is over
  gold-bearing queries only. Gold-lessness is detected from the row's
  `expected_sources` (docs/facts) rather than a dataset-specific category
  name, so this works unmodified for HotpotQA/MuSiQue too (every query in
  those two datasets carries gold evidence, so the exclusion never fires).

The dataset name/tag is read from each result file's own `dataset`/
`corpus_tag` fields — nothing dataset-specific needs to be passed in.

Outputs: a stats JSON + tidy CSV next to the prehop run, and a forest-plot
PNG into fig/ — all named after the dataset's corpus tag.

Usage:
  python scripts/paired_bootstrap.py \
    --prehop data/results/<new>/prehop/multihoprag/prehop_multihoprag.json \
    --baselines data/results/<base>/{naive,hoprag,ms_graphrag}/multihoprag/*.json \
    --out-dir data/results/<new>

  python scripts/paired_bootstrap.py \
    --prehop data/results/<new>/prehop/hotpotqa/prehop_hotpotqa.json \
    --baselines data/results/<base>/{naive,hoprag,ms_graphrag}/hotpotqa/*.json \
    --out-dir data/results/<new>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

JUDGE_METRICS = ["llm_judge_score", "hallucination"]
RETRIEVAL_METRICS = ["mrr@10", "map@10", "hits@4", "hits@10", "evidence_doc_recall"]
LOWER_IS_BETTER = {"hallucination"}
N_BOOT = 10000
SEED = 42


def _load(path: str) -> tuple[str, str, dict[str, dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    strat = data["strategy"]
    corpus_tag = data.get("corpus_tag") or "unknown"
    by_query = {r["query"]: r for r in data["details"] if isinstance(r, dict)}
    return strat, corpus_tag, by_query


def _has_gold(row: dict) -> bool:
    expected = row.get("expected_sources") or {}
    return bool(expected.get("docs")) or bool(expected.get("facts"))


def _paired(prehop: dict[str, dict], base: dict[str, dict], metric: str) -> np.ndarray:
    diffs = []
    retrieval = metric in RETRIEVAL_METRICS
    for q, pr in prehop.items():
        ba = base.get(q)
        if ba is None:
            continue
        if retrieval and not _has_gold(pr):
            continue  # gold-less (e.g. MultiHop-RAG null_query)
        pv, bv = pr.get(metric), ba.get(metric)
        if pv is None or bv is None:
            continue
        if metric in JUDGE_METRICS and (pv < 0 or bv < 0):
            continue  # unjudged sentinel
        diffs.append(float(pv) - float(bv))
    return np.asarray(diffs, dtype=float)


def _bootstrap(diffs: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(diffs)
    if n == 0:
        return {"n": 0}
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "n": n,
        "mean_diff": float(diffs.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prehop", required=True)
    ap.add_argument("--baselines", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fig", default=None, help="default: fig/<corpus_tag>_bootstrap_forest.png")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    treatment_strat, corpus_tag, prehop = _load(args.prehop)
    baselines = {}
    for p in args.baselines:
        strat, _, rows = _load(p)
        baselines[strat] = rows

    fig_path = Path(args.fig) if args.fig else Path(f"fig/{corpus_tag}_bootstrap_forest.png")

    metrics = JUDGE_METRICS + RETRIEVAL_METRICS
    results: dict[str, dict[str, dict]] = {m: {} for m in metrics}
    for m in metrics:
        for strat, base in baselines.items():
            results[m][strat] = _bootstrap(_paired(prehop, base, m), rng)

    out_dir = Path(args.out_dir)
    (out_dir / f"{corpus_tag}_paired_bootstrap.json").write_text(
        json.dumps(
            {"dataset": corpus_tag, "treatment": treatment_strat, "n_boot": N_BOOT, "seed": SEED, "results": results},
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out_dir / f"{corpus_tag}_paired_bootstrap.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "baseline", "n", "mean_diff", "ci95_low", "ci95_high", "significant"])
        for m in metrics:
            for strat, st in results[m].items():
                if st.get("n", 0):
                    w.writerow(
                        [
                            m,
                            strat,
                            st["n"],
                            f"{st['mean_diff']:.4f}",
                            f"{st['ci95_low']:.4f}",
                            f"{st['ci95_high']:.4f}",
                            st["significant"],
                        ]
                    )

    _plot(results, metrics, fig_path, treatment_strat, corpus_tag)

    # console
    print(f"{treatment_strat} vs baselines on {corpus_tag} — paired bootstrap (N={N_BOOT}, seed={SEED})")
    print(f"  diff = {treatment_strat} - baseline; * = 95% CI excludes 0 (significant)")
    for m in metrics:
        arrow = " (lower better)" if m in LOWER_IS_BETTER else ""
        print(f"\n{m}{arrow}:")
        for strat, st in results[m].items():
            if not st.get("n"):
                continue
            star = " *" if st["significant"] else "  "
            print(
                f"  vs {strat:12s} Δ={st['mean_diff']:+.4f}  [{st['ci95_low']:+.4f}, {st['ci95_high']:+.4f}]{star} (n={st['n']})"
            )


def _plot(results: dict, metrics: list[str], fig_path: Path, treatment_strat: str, corpus_tag: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baselines = list(next(iter(results.values())).keys())
    colors = {"naive": "#888888", "hoprag": "#4C72B0", "ms_graphrag": "#DD8452"}
    ncol = len(metrics)
    fig, axes = plt.subplots(1, ncol, figsize=(3.0 * ncol, 3.4), sharey=True)
    if ncol == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        lower = m in LOWER_IS_BETTER
        ys = list(range(len(baselines)))[::-1]
        for y, strat in zip(ys, baselines):
            st = results[m].get(strat, {})
            if not st.get("n"):
                continue
            md, lo, hi = st["mean_diff"], st["ci95_low"], st["ci95_high"]
            sig = st["significant"]
            c = colors.get(strat, "#333333")
            ax.plot([lo, hi], [y, y], color=c, lw=2.2, solid_capstyle="round", alpha=1.0 if sig else 0.45)
            ax.plot(
                [md],
                [y],
                "o",
                color=c,
                ms=7,
                alpha=1.0 if sig else 0.45,
                markeredgecolor="black" if sig else "none",
                markeredgewidth=0.8,
            )
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
        title = m + ("\n(lower better)" if lower else "")
        ax.set_title(title, fontsize=10)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"vs {b}" for b in baselines], fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.margins(y=0.25)

    fig.suptitle(
        f"{treatment_strat} − baseline on {corpus_tag} (query-level paired bootstrap, 95% CI)  •  solid = CI excludes 0",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved figure -> {fig_path}")


if __name__ == "__main__":
    main()
