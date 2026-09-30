"""Benchmark scope, observed metric aggregation, and completion labels."""
from pathlib import Path
from typing import Any

from core.benchmark_failures import POLICY as FAILURE_POLICY
from core.benchmark_failures import metric_value
from utils.hotpotqa import HIPPORAG_PROTOCOL, HIPPORAG_QUERY_COUNT
from utils.io import _safe_float
from utils.metrics import extract_final_answer

OFFICIAL_SPLIT_QUERY_COUNTS = {"multihoprag": 2556, "hotpotqa": HIPPORAG_QUERY_COUNT}


def _extract_stage_timing(trace: Any) -> dict[str, float]:
    """Pull rewrite/retrieve/traversal/synthesis timing out of a prehop-style
    `interaction_trace` (see models/prehop/graphrag.py's `run_workflow`),
    if present, so they land as top-level numeric fields on `result_item`
    and get auto-averaged into the corresponding ``avg_*`` fields by
    `_recompute_aggregates`. This keeps the stage split
    rather than only the aggregate `latency`.

    Other strategies' traces don't carry these keys, so this returns {} for
    them — deliberately not defaulting to 0.0, which would misreport "zero
    latency" instead of "not measured" once averaged.
    """
    if not isinstance(trace, list):
        return {}
    timing: dict[str, float] = {}
    for step in trace:
        if not isinstance(step, dict):
            continue
        if step.get("step") == "query_rewrite":
            if "rewrite_ms" in step:
                timing["rewrite_ms"] = float(step.get("rewrite_ms") or 0.0)
        elif step.get("step") == "retrieve":
            for key in (
                "retrieve_ms",
                "traversal_ms",
                "graph_expand_ms",
                "deterministic_score_ms",
                "candidate_order_ms",
            ):
                if key in step:
                    timing[key] = float(step.get(key) or 0.0)
        elif step.get("step") == "synthesis" and "synthesis_ms" in step:
            timing["synthesis_ms"] = float(step.get("synthesis_ms") or 0.0)
        elif str(step.get("step", "")).endswith("_official_retrieval") and "worker_queue_seconds" in step:
            timing["worker_queue_seconds"] = float(step["worker_queue_seconds"])
    return timing

def _apply_answer_label(result_item: dict[str, Any]) -> None:
    """Attach deterministic labels from answer EM or the null-refusal metric."""
    from utils.abstain import is_abstain

    answer_text = str(result_item.get("answer", "") or "")
    has_error = bool(result_item.get("error"))
    # Detect abstain on the EXTRACTED final answer (\\boxed{} / 'Final Answer:'),
    # not the full CoT body which often uses 'insufficient evidence' mid-reason.
    final_answer = extract_final_answer(answer_text).lower()
    abstained = is_abstain(final_answer)
    result_item["final_answer_extracted"] = final_answer[:300]

    # The headline correctness/label is deterministic.  A null query uses its
    # explicit refusal metric; other rows use EM.
    primary = (
        result_item.get("null_refusal")
        if result_item.get("question_type") == "null_query"
        else result_item.get("answer_em")
    )
    primary_score = _safe_float(primary, -1.0)
    primary_score = 0.0 if has_error else primary_score
    result_item["primary_answer_score"] = primary_score
    if has_error or primary_score < 0:
        answer_attempted = -1.0 if not has_error else 0.0
        primary_label = "Incorrect Answer" if has_error else "Unscored"
    else:
        answer_attempted = 0.0 if abstained else 1.0
        primary_label = "Correct Answer" if primary_score >= 0.5 else ("Refusal" if abstained else "Incorrect Answer")
    result_item["answer_attempted"] = answer_attempted
    result_item["answer_label"] = primary_label

def _recompute_aggregates(s: dict[str, Any]) -> None:
    """Recompute avg_<metric>, category_summaries and the 3-way label counts
    from ``s['details']`` in place. Averages skip
    the UNJUDGED sentinel (-1); every real metric is in [0, 1] (or latency >= 0).
    """
    rows = s.get("details") or []
    s["failure_policy"] = FAILURE_POLICY
    s["query_failure_count"] = sum(bool(row.get("error")) for row in rows)
    s["query_failure_rate"] = s["query_failure_count"] / len(rows) if rows else 0.0

    def _eligible_values(subset: list[dict], key: str) -> list[float]:
        return [value for row in subset if (value := metric_value(row, key)) is not None]

    def _avg(subset: list[dict], key: str) -> float:
        vals = _eligible_values(subset, key)
        return sum(vals) / len(vals) if vals else 0.0

    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    for key in numeric_keys:
        s[f"avg_{key}"] = _avg(rows, key)
        s[f"eligible_{key}_count"] = len(_eligible_values(rows, key))

    cats: dict[str, list] = {}
    for r in rows:
        cats.setdefault(r.get("category", "Uncategorized"), []).append(r)
    cat_summaries: dict[str, Any] = {}
    for cat, cat_list in cats.items():
        cat_sum: dict[str, Any] = {"count": len(cat_list)}
        for key in numeric_keys:
            cat_sum[f"avg_{key}"] = _avg(cat_list, key)
            cat_sum[f"eligible_{key}_count"] = len(_eligible_values(cat_list, key))
        cat_summaries[cat] = cat_sum
    s["category_summaries"] = cat_summaries

    label_counts = {"Correct Answer": 0, "Incorrect Answer": 0, "Refusal": 0}
    for r in rows:
        label = r.get("answer_label")
        if label in label_counts:
            label_counts[label] += 1
    total = sum(label_counts.values()) or 1  # deterministically scored rows only
    s["correct_count"] = label_counts["Correct Answer"]
    s["incorrect_count"] = label_counts["Incorrect Answer"]
    s["refusal_count"] = label_counts["Refusal"]
    s["correct_rate"] = label_counts["Correct Answer"] / total
    s["incorrect_rate"] = label_counts["Incorrect Answer"] / total
    s["refusal_rate"] = label_counts["Refusal"] / total

def _update_summary_status(summary: dict[str, Any]) -> None:
    rows = summary.get("details") or []
    if any(row.get("failure_scope") == "target" for row in rows):
        summary["status"] = "failed"
    elif len(rows) < int(summary.get("total_queries", len(rows)) or 0):
        summary["status"] = "in_progress"
    else:
        summary["status"] = "completed_unadmitted"

def _evaluation_scope(
    dataset: str,
    evaluated_count: int,
    source: str,
    *,
    manifest: dict | None = None,
) -> tuple[str, int | None]:
    """Classify an artifact from its actual evaluated row count.

    A complete expected split is ``full_benchmark`` even if its filename is
    unconventional.  Incomplete files explicitly named as samples remain
    ``sample_exploratory``; all other incomplete selections (including CLI
    ``--limit``) are ``subset_exploratory``.
    """
    if manifest and manifest.get("protocol") == HIPPORAG_PROTOCOL:
        return ("released_benchmark" if evaluated_count == manifest["query_count"] else "subset_exploratory"), manifest["query_count"]
    expected = OFFICIAL_SPLIT_QUERY_COUNTS.get(str(dataset).lower())
    if expected is not None and evaluated_count == expected:
        return ("released_benchmark" if str(dataset).lower() == "hotpotqa" else "full_benchmark"), expected
    if "sample" in Path(source).name.lower():
        return "sample_exploratory", expected
    return "subset_exploratory", expected

def _aggregate_seed_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean / std / 95% CI per metric across N seeds.

    CI = mean ± 1.96 * std / sqrt(N)  (normal-approx; fine for N>=3 + smooth metrics).
    Per-category aggregates are computed only over keys that appear in every seed.
    """
    import math

    def _agg_keys(values: list[float]) -> dict[str, float]:
        n = len(values)
        if n == 0:
            return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0}
        mean = sum(values) / n
        if n == 1:
            return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean, "n": 1}
        var = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(var)
        margin = 1.96 * std / math.sqrt(n)
        return {"mean": mean, "std": std, "ci95_low": mean - margin, "ci95_high": mean + margin, "n": n}

    if not summaries:
        return {}

    avg_keys = sorted({k for s in summaries for k in s if k.startswith("avg_")})
    overall: dict[str, Any] = {}
    for key in avg_keys:
        metric = key.removeprefix("avg_")
        vals = [
            _safe_float(s[key], 0.0)
            for s in summaries
            if key in s and _safe_float(s.get(f"eligible_{metric}_count"), 0.0) > 0
        ]
        if vals:
            overall[key] = _agg_keys(vals)

    # Category-level aggregation: only categories that all seeds reported
    common_cats: set[str] | None = None
    for s in summaries:
        cats = set((s.get("category_summaries") or {}).keys())
        common_cats = cats if common_cats is None else (common_cats & cats)
    common_cats = common_cats or set()

    categories: dict[str, dict[str, Any]] = {}
    for cat in sorted(common_cats):
        cat_keys = sorted(
            {k for s in summaries for k in (s.get("category_summaries", {}).get(cat, {}) or {}) if k.startswith("avg_")}
        )
        per_cat = {}
        for key in cat_keys:
            metric = key.removeprefix("avg_")
            vals = [
                _safe_float(s["category_summaries"][cat][key], 0.0)
                for s in summaries
                if cat in (s.get("category_summaries") or {})
                and key in s["category_summaries"][cat]
                and _safe_float(s["category_summaries"][cat].get(f"eligible_{metric}_count"), 0.0) > 0
            ]
            if vals:
                per_cat[key] = _agg_keys(vals)
        per_cat["count"] = int(summaries[0].get("category_summaries", {}).get(cat, {}).get("count", 0))
        categories[cat] = per_cat

    return {"overall": overall, "categories": categories}
