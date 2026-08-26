"""Query-level paired bootstrap: prehop vs each baseline, on the active
multi-hop datasets (MultiHop-RAG / MuSiQue).

This script pairs per-query scores by stable query ID, computes the paired
difference (prehop - baseline) per query, and bootstraps the mean diff to a
95% CI. A diff whose CI excludes 0 is a statistically separated win/loss.

- Judge / hallucination are excluded from the primary bootstrap by default.
  Pass ``--include-judge`` to emit a clearly labelled supplemental analysis;
  judged rows only (sentinel -1 dropped), and hallucination is lower-is-better.
- Dataset-specific metrics are selected from the result artifact: official
  MultiHop-RAG ranking/custom fact recall or MuSiQue answer/support metrics.
  A query with no gold evidence (e.g. MultiHop-RAG's null_query category) is
  excluded from retrieval comparisons. Gold-lessness is detected from the row's
  `expected_sources` (docs/facts) rather than a dataset-specific category
  name, so this works for the active datasets (every MuSiQue query carries gold
  evidence, so the exclusion never fires there). Runtime errors and every
  negative sentinel are excluded for all metric families.

The dataset name/tag is read from each result file's own `dataset`/
`corpus_tag` fields — nothing dataset-specific needs to be passed in.

Outputs: a stats JSON + tidy CSV next to the prehop run, and a forest-plot
PNG into fig/ — all named after the dataset's corpus tag.

Usage:
  python scripts/paired_bootstrap.py \
    --prehop data/results/<new>/prehop/multihoprag/prehop_multihoprag.json \
    --baselines data/results/<base>/{naive,hoprag,ms_graphrag}/multihoprag/*.json \
    --out-dir data/results/<new>

"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SUPPLEMENTAL_JUDGE_METRICS = ["llm_judge_score", "groundedness", "hallucination"]
MULTIHOPRAG_METRICS = [
    "answer_em",
    "answer_f1",
    "official_qa_accuracy",
    "official_mrr@10",
    "official_map@10",
    "official_hits@4",
    "official_hits@10",
    "evidence_fact_recall@4",
    "evidence_fact_recall@10",
    "evidence_doc_precision",
    "evidence_doc_recall",
    "evidence_doc_f1",
]
MUSIQUE_METRICS = [
    "answer_em",
    "answer_f1",
    "official_answer_em",
    "official_answer_f1",
    "paragraph_support_precision",
    "paragraph_support_recall",
    "paragraph_support_f1",
    "evidence_doc_precision",
    "evidence_doc_recall",
    "evidence_doc_f1",
]
RETRIEVAL_METRICS = set(MULTIHOPRAG_METRICS[3:] + MUSIQUE_METRICS[4:])
LOWER_IS_BETTER = {"hallucination"}
N_BOOT = 10000
SEED = 42


def _load(path: str, *, allow_legacy: bool = False) -> tuple[str, str, dict, dict[str, dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    strat = data["strategy"]
    corpus_tag = data.get("corpus_tag") or "unknown"
    rows = [row for row in data["details"] if isinstance(row, dict)]
    by_query: dict[str, dict] = {}
    for row in rows:
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            if not allow_legacy:
                raise ValueError(
                    f"{path}: result row lacks stable query_id; pass --allow-legacy-exploratory only for exploratory analysis"
                )
            query_id = f"legacy-query:{row.get('query', '')}"
        if query_id in by_query:
            raise ValueError(f"{path}: duplicate query identity {query_id!r}")
        by_query[query_id] = row
    return strat, corpus_tag, data, by_query


def _validate_artifact_pair(
    treatment: dict,
    baseline: dict,
    *,
    allow_exploratory: bool = False,
    allow_legacy: bool = False,
) -> None:
    """Fail fast unless two result artifacts are scientifically comparable."""
    for label, artifact in (("treatment", treatment), ("baseline", baseline)):
        if artifact.get("status") != "completed":
            raise ValueError(f"{label} artifact status must be 'completed', got {artifact.get('status')!r}")
        scope = artifact.get("evaluation_scope")
        if scope != "full_benchmark" and not allow_exploratory:
            raise ValueError(
                f"{label} artifact scope is {scope!r}; pass --allow-exploratory for an explicitly exploratory comparison"
            )
        required_identity = {
            "corpus_manifest_fingerprint",
            "index_manifest_fingerprint",
            "corpus_index_fingerprint_status",
        }
        if not required_identity.issubset(artifact) and not allow_legacy:
            raise ValueError(
                f"{label} artifact lacks corpus/index identity metadata; "
                "pass --allow-legacy-exploratory only for non-paper analysis"
            )

    for key in (
        "dataset",
        "corpus_tag",
        "evaluation_scope",
        "corpus_manifest_fingerprint",
        "index_manifest_fingerprint",
        "corpus_index_fingerprint_status",
    ):
        if treatment.get(key) != baseline.get(key):
            raise ValueError(
                f"Incompatible artifacts: {key} differs ({treatment.get(key)!r} != {baseline.get(key)!r})"
            )

    def _pair_ids(artifact: dict) -> set[str]:
        identities: set[str] = set()
        for row in artifact.get("details") or []:
            if not isinstance(row, dict):
                continue
            query_id = row.get("query_id")
            if not isinstance(query_id, str) or not query_id.strip():
                if not allow_legacy:
                    raise ValueError("Artifact row lacks stable query_id")
                query_id = f"legacy-query:{row.get('query', '')}"
            identities.add(query_id)
        return identities

    treatment_ids = _pair_ids(treatment)
    baseline_ids = _pair_ids(baseline)
    if treatment_ids != baseline_ids:
        raise ValueError(
            "Incompatible artifacts: stable query ID sets differ "
            f"({len(treatment_ids)} treatment vs {len(baseline_ids)} baseline)"
        )


def _has_gold(row: dict) -> bool:
    expected = row.get("expected_sources") or {}
    return bool(expected.get("docs")) or bool(expected.get("facts")) or bool(expected.get("paragraph_ids"))


def _paired(prehop: dict[str, dict], base: dict[str, dict], metric: str) -> np.ndarray:
    diffs = []
    retrieval = metric in RETRIEVAL_METRICS
    for q, pr in prehop.items():
        ba = base.get(q)
        if ba is None:
            continue
        if pr.get("error") or ba.get("error"):
            continue
        if retrieval and not _has_gold(pr):
            continue  # gold-less (e.g. MultiHop-RAG null_query)
        pv, bv = pr.get(metric), ba.get(metric)
        if pv is None or bv is None:
            continue
        try:
            pv, bv = float(pv), float(bv)
        except (TypeError, ValueError):
            continue
        # Every negative value is an ineligible sentinel, irrespective of
        # metric family.  This prevents MuSiQue's N/A retrieval fields from
        # becoming a spurious all-zero paired sample.
        if pv < 0 or bv < 0:
            continue
        diffs.append(pv - bv)
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
    ap.add_argument("--include-judge", action="store_true", help="also run supplemental LLM-judge bootstrap metrics")
    ap.add_argument("--allow-exploratory", action="store_true", help="allow non-full but otherwise compatible artifacts")
    ap.add_argument(
        "--allow-legacy-exploratory",
        action="store_true",
        help="allow old artifacts without query IDs/fingerprint metadata; never use for paper claims",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    treatment_strat, corpus_tag, treatment_artifact, prehop = _load(
        args.prehop,
        allow_legacy=args.allow_legacy_exploratory,
    )
    baselines = {}
    for p in args.baselines:
        strat, _, baseline_artifact, rows = _load(p, allow_legacy=args.allow_legacy_exploratory)
        _validate_artifact_pair(
            treatment_artifact,
            baseline_artifact,
            allow_exploratory=args.allow_exploratory or args.allow_legacy_exploratory,
            allow_legacy=args.allow_legacy_exploratory,
        )
        baselines[strat] = rows

    fig_path = Path(args.fig) if args.fig else Path(f"fig/{corpus_tag}_bootstrap_forest.png")

    dataset_marker = str(corpus_tag).lower()
    if dataset_marker == "multihoprag":
        metrics = MULTIHOPRAG_METRICS
    elif dataset_marker == "musique":
        metrics = MUSIQUE_METRICS
    else:
        # Unknown artifacts are still safely processed, but no dataset-only
        # metric is silently claimed to be applicable.
        metrics = []
    judge_metrics = SUPPLEMENTAL_JUDGE_METRICS if args.include_judge else []
    all_metrics = metrics + judge_metrics
    results: dict[str, dict[str, dict]] = {m: {} for m in all_metrics}
    for m in all_metrics:
        for strat, base in baselines.items():
            results[m][strat] = _bootstrap(_paired(prehop, base, m), rng)

    out_dir = Path(args.out_dir)
    (out_dir / f"{corpus_tag}_paired_bootstrap.json").write_text(
        json.dumps(
            {
                "dataset": corpus_tag,
                "treatment": treatment_strat,
                "n_boot": N_BOOT,
                "seed": SEED,
                "pair_key": "legacy_query_text" if args.allow_legacy_exploratory else "query_id",
                "exploratory_override": bool(args.allow_exploratory or args.allow_legacy_exploratory),
                "primary_metrics": metrics,
                "supplemental_judge_metrics": judge_metrics,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out_dir / f"{corpus_tag}_paired_bootstrap.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "baseline", "n", "mean_diff", "ci95_low", "ci95_high", "significant"])
        for m in all_metrics:
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

    _plot(results, all_metrics, fig_path, treatment_strat, corpus_tag)

    # console
    print(f"{treatment_strat} vs baselines on {corpus_tag} — paired bootstrap (N={N_BOOT}, seed={SEED})")
    print(f"  diff = {treatment_strat} - baseline; * = 95% CI excludes 0 (significant)")
    for m in all_metrics:
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
