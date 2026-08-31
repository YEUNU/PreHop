"""Verify the predeclared relative-improvement gate on official metrics.

The gate is intentionally aggregate-only and dataset-specific only in the
official metric definitions. It selects the strongest supplied non-Prehop
baseline independently for each metric and requires Prehop to exceed that
value by the requested relative margin on every metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

OFFICIAL_METRICS = {
    "multihoprag": {
        "Hits@4": "avg_official_hits@4",
        "Hits@10": "avg_official_hits@10",
        "MRR@10": "avg_official_mrr@10",
        "MAP@10": "avg_official_map@10",
    },
    "musique": {
        "Answer EM": "avg_official_answer_em",
        "Answer F1": "avg_official_answer_f1",
        "Support precision": "avg_paragraph_support_precision",
        "Support recall": "avg_paragraph_support_recall",
        "Support F1": "avg_paragraph_support_f1",
    },
}


def _load_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{artifact_path}: benchmark artifact must be a JSON object")
    payload = dict(payload)
    payload["_path"] = str(artifact_path)
    return payload


def _dataset_key(artifact: dict[str, Any]) -> str:
    marker = re.sub(
        r"[^a-z0-9]+",
        "",
        str(artifact.get("dataset") or artifact.get("corpus_tag") or "").casefold(),
    )
    if marker == "multihoprag":
        return "multihoprag"
    if marker == "musique":
        return "musique"
    raise ValueError(f"Unsupported benchmark dataset identity: {marker!r}")


def _validate_artifacts(
    prehop: dict[str, Any],
    baselines: list[dict[str, Any]],
    *,
    allow_exploratory: bool,
) -> str:
    if prehop.get("strategy") != "prehop":
        raise ValueError("Treatment artifact must use strategy='prehop'")
    if not baselines:
        raise ValueError("At least one non-Prehop baseline artifact is required")
    if any(baseline.get("strategy") == "prehop" for baseline in baselines):
        raise ValueError("Baseline artifacts must not use strategy='prehop'")

    dataset = _dataset_key(prehop)
    reference = {
        "dataset": dataset,
        "evaluation_scope": prehop.get("evaluation_scope"),
        "corpus_manifest_fingerprint": prehop.get("corpus_manifest_fingerprint"),
        "evaluated_query_ids_sha256": prehop.get("evaluated_query_ids_sha256"),
        "evaluated_queries_count": prehop.get("evaluated_queries_count"),
    }
    for artifact in [prehop, *baselines]:
        path = artifact.get("_path", "<artifact>")
        if artifact.get("status") != "completed":
            raise ValueError(f"{path}: status must be 'completed'")
        if artifact.get("corpus_index_fingerprint_status") != "matched":
            raise ValueError(f"{path}: corpus/index fingerprint status must be 'matched'")
        if artifact.get("evaluation_scope") != "full_benchmark" and not allow_exploratory:
            raise ValueError(
                f"{path}: only full_benchmark artifacts are eligible; "
                "pass --allow-exploratory only for development diagnostics"
            )
        observed = {
            "dataset": _dataset_key(artifact),
            "evaluation_scope": artifact.get("evaluation_scope"),
            "corpus_manifest_fingerprint": artifact.get("corpus_manifest_fingerprint"),
            "evaluated_query_ids_sha256": artifact.get("evaluated_query_ids_sha256"),
            "evaluated_queries_count": artifact.get("evaluated_queries_count"),
        }
        if observed != reference:
            differing = [key for key in reference if observed[key] != reference[key]]
            raise ValueError(f"{path}: incompatible evaluation identity fields: {differing}")
    return dataset


def evaluate_gate(
    prehop: dict[str, Any],
    baselines: list[dict[str, Any]],
    *,
    margin: float = 0.10,
    allow_exploratory: bool = False,
) -> dict[str, Any]:
    """Return a metric-by-metric strongest-baseline gate report."""
    if margin < 0:
        raise ValueError("Relative improvement margin must be non-negative")
    dataset = _validate_artifacts(
        prehop,
        baselines,
        allow_exploratory=allow_exploratory,
    )
    rows: list[dict[str, Any]] = []
    for metric, field in OFFICIAL_METRICS[dataset].items():
        try:
            prehop_value = float(prehop[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Prehop artifact lacks valid metric {field!r}") from error
        baseline_values: list[tuple[float, dict[str, Any]]] = []
        for baseline in baselines:
            try:
                value = float(baseline[field])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{baseline.get('_path', '<artifact>')}: lacks valid metric {field!r}") from error
            if value < 0:
                raise ValueError(f"{baseline.get('_path', '<artifact>')}: metric {field!r} is ineligible")
            baseline_values.append((value, baseline))
        strongest_value, strongest = max(
            baseline_values,
            key=lambda item: (item[0], str(item[1].get("strategy") or "")),
        )
        target = strongest_value * (1.0 + margin)
        passed = prehop_value + 1e-12 >= target
        rows.append(
            {
                "metric": metric,
                "field": field,
                "prehop": prehop_value,
                "strongest_baseline": strongest_value,
                "strongest_strategy": strongest.get("strategy"),
                "strongest_artifact": strongest.get("_path"),
                "required": target,
                "relative_improvement": ((prehop_value / strongest_value) - 1.0 if strongest_value > 0 else None),
                "pass": passed,
            }
        )
    return {
        "dataset": dataset,
        "margin": margin,
        "evaluation_scope": prehop.get("evaluation_scope"),
        "prehop_artifact": prehop.get("_path"),
        "baseline_artifacts": [baseline.get("_path") for baseline in baselines],
        "metrics": rows,
        "pass": all(row["pass"] for row in rows),
        "paper_eligible": prehop.get("evaluation_scope") == "full_benchmark",
    }


def _write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "prehop",
                "strongest_strategy",
                "strongest_baseline",
                "required",
                "relative_improvement",
                "pass",
            ],
        )
        writer.writeheader()
        for row in report["metrics"]:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Require Prehop to exceed the strongest baseline on every official metric",
    )
    parser.add_argument("--prehop", required=True, help="Prehop benchmark JSON or summary JSON")
    parser.add_argument(
        "--baselines",
        nargs="+",
        required=True,
        help="Non-Prehop benchmark JSON or summary JSON files",
    )
    parser.add_argument("--margin", type=float, default=0.10, help="required relative gain (default: 0.10)")
    parser.add_argument(
        "--allow-exploratory",
        action="store_true",
        help="allow a sample/subset comparison, which remains ineligible for paper claims",
    )
    parser.add_argument("--output", required=True, help="output JSON path; a CSV is written beside it")
    args = parser.parse_args()

    report = evaluate_gate(
        _load_artifact(args.prehop),
        [_load_artifact(path) for path in args.baselines],
        margin=args.margin,
        allow_exploratory=args.allow_exploratory,
    )
    _write_report(report, Path(args.output))
    print(
        f"{report['dataset']} official-metric gate: "
        f"{'PASS' if report['pass'] else 'FAIL'} "
        f"(required relative gain={report['margin']:.1%})"
    )
    for row in report["metrics"]:
        print(
            f"  {row['metric']}: prehop={row['prehop']:.6f}, "
            f"strongest={row['strongest_baseline']:.6f} "
            f"({row['strongest_strategy']}), required={row['required']:.6f}, "
            f"{'PASS' if row['pass'] else 'FAIL'}"
        )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
