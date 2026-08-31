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
    --exclude-queries data/multihoprag_sample200_queries.json \
    --out-dir data/results/<new>

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
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
    allow_index_variant: bool = False,
    expected_ablation_differences: set[str] | None = None,
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

    identity_keys = [
        "dataset",
        "evaluation_scope",
        "corpus_manifest_fingerprint",
        "index_manifest_fingerprint",
        "corpus_index_fingerprint_status",
    ]
    if not allow_index_variant:
        identity_keys.insert(1, "corpus_tag")
    for key in identity_keys:
        if treatment.get(key) != baseline.get(key):
            raise ValueError(f"Incompatible artifacts: {key} differs ({treatment.get(key)!r} != {baseline.get(key)!r})")

    expected_ablation_differences = expected_ablation_differences or set()
    if expected_ablation_differences:
        controlled_metadata = (
            "active_index_snapshot",
            "models",
            "query_provenance",
            "evaluation_provenance",
            "benchmark_concurrency",
            "judge_enabled",
        )
        changed_metadata = [key for key in controlled_metadata if treatment.get(key) != baseline.get(key)]
        if changed_metadata:
            raise ValueError(
                f"Incompatible query-only ablation pair: controlled metadata differs for {changed_metadata!r}"
            )
        treatment_ablation = treatment.get("ablation")
        baseline_ablation = baseline.get("ablation")
        if not isinstance(treatment_ablation, dict) or not isinstance(baseline_ablation, dict):
            raise ValueError("Ablation-contract validation requires ablation metadata in both artifacts")
        observed_differences = {
            key
            for key in treatment_ablation.keys() | baseline_ablation.keys()
            if treatment_ablation.get(key) != baseline_ablation.get(key)
        }
        if observed_differences != expected_ablation_differences:
            raise ValueError(
                "Incompatible query-only ablation pair: expected only "
                f"{sorted(expected_ablation_differences)!r} to differ, observed "
                f"{sorted(observed_differences)!r}"
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


def _dataset_marker(artifact: dict, corpus_tag: str) -> str:
    """Normalize dataset identity independently of an experiment corpus tag."""
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(artifact.get("dataset") or corpus_tag).lower(),
    )


def _load_excluded_query_ids(path: str | None) -> set[str]:
    if not path:
        return set()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Excluded-query file must contain a JSON list")
    query_ids = {
        str(row.get("_id") or row.get("query_id") or "").strip() if isinstance(row, dict) else str(row or "").strip()
        for row in payload
    }
    if not query_ids or "" in query_ids:
        raise ValueError("Excluded-query file contains a blank or empty query-ID set")
    return query_ids


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
    ap.add_argument(
        "--allow-exploratory", action="store_true", help="allow non-full but otherwise compatible artifacts"
    )
    ap.add_argument(
        "--allow-index-variant",
        action="store_true",
        help=(
            "allow different corpus tags for an explicit index-changing comparison; "
            "dataset, corpus fingerprint, evaluation scope, and query IDs must still match"
        ),
    )
    ap.add_argument(
        "--expected-ablation-difference",
        action="append",
        default=[],
        help=(
            "require this ablation metadata key, and no unlisted key, to differ; "
            "repeat for multiple intentional query-only differences"
        ),
    )
    ap.add_argument(
        "--exclude-queries",
        default=None,
        help="JSON list of fixed development rows/IDs to exclude from confirmatory pairs",
    )
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
    excluded_query_ids = _load_excluded_query_ids(args.exclude_queries)
    unknown_exclusions = excluded_query_ids - set(prehop)
    if unknown_exclusions:
        raise ValueError(f"Excluded query IDs are absent from the treatment artifact: {sorted(unknown_exclusions)[:5]}")
    prehop = {query_id: row for query_id, row in prehop.items() if query_id not in excluded_query_ids}
    baselines = {}
    for p in args.baselines:
        strat, _, baseline_artifact, rows = _load(p, allow_legacy=args.allow_legacy_exploratory)
        _validate_artifact_pair(
            treatment_artifact,
            baseline_artifact,
            allow_exploratory=args.allow_exploratory or args.allow_legacy_exploratory,
            allow_legacy=args.allow_legacy_exploratory,
            allow_index_variant=args.allow_index_variant,
            expected_ablation_differences=set(args.expected_ablation_difference),
        )
        baselines[strat] = {query_id: row for query_id, row in rows.items() if query_id not in excluded_query_ids}

    fig_path = Path(args.fig) if args.fig else Path(f"fig/{corpus_tag}_bootstrap_forest.png")

    dataset_marker = _dataset_marker(treatment_artifact, corpus_tag)
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
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{corpus_tag}_paired_bootstrap.json").write_text(
        json.dumps(
            {
                "dataset": corpus_tag,
                "treatment": treatment_strat,
                "n_boot": N_BOOT,
                "seed": SEED,
                "pair_key": "legacy_query_text" if args.allow_legacy_exploratory else "query_id",
                "analysis_scope": (
                    "heldout_excluding_fixed_development_ids" if excluded_query_ids else "complete_artifact_query_set"
                ),
                "excluded_query_count": len(excluded_query_ids),
                "excluded_query_ids_sha256": (
                    hashlib.sha256("\n".join(sorted(excluded_query_ids)).encode()).hexdigest()
                    if excluded_query_ids
                    else None
                ),
                "exploratory_override": bool(args.allow_exploratory or args.allow_legacy_exploratory),
                "index_variant_override": bool(args.allow_index_variant),
                "expected_ablation_differences": sorted(set(args.expected_ablation_difference)),
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
