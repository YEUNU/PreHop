"""Dataset-specific official reports, separate from auxiliary diagnostics."""
from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

from utils.io import _write_json

UPSTREAM = {
    "multihoprag": {
        "repository": "https://github.com/yixuantt/MultiHop-RAG",
        "revision": "c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8",
        "files": ["retrieval_evaluate.py", "qa_evaluate.py"],
    },
    "hotpotqa": {
        "repository": "https://github.com/hotpotqa/hotpot",
        "revision": "3635853403a8735609ee997664e1528f4480762a",
        "files": ["hotpot_evaluate_v1.py"],
    },
}
RANK_FIELDS = {
    "Hits@4": "official_hits@4", "Hits@10": "official_hits@10",
    "MRR@10": "official_mrr@10", "MAP@10": "official_map@10",
}


def dataset_key(name: str) -> str:
    """Resolve saved display names and corpus tags to the official evaluator."""
    dataset = str(name).lower().replace("-", "").replace("_", "")
    if dataset not in UPSTREAM:
        raise ValueError(f"Unsupported official dataset: {dataset}")
    return dataset


def _number(row: dict[str, Any], key: str) -> float:
    if row.get("error"):
        return 0.0
    value = float(row[key])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Missing official score {key} for {row.get('query_id')}")
    return value


def build_reports(result: dict[str, Any]) -> tuple[dict, dict]:
    """Export recorded official scores without changing their prediction boundary.

    Null MultiHop-RAG questions remain in QA and are excluded from retrieval.
    Failed rows stay in the applicable denominators with zero quality scores.
    Official values use the evaluator's native scale, including MAP above one.
    """
    dataset = dataset_key(result.get("dataset", ""))
    rows = result.get("details", [])
    if not rows:
        raise ValueError("Official report requires evaluated rows")
    ids = [str(r["query_id"]) for r in rows]
    if len(ids) != len(set(ids)) or any(not q for q in ids):
        raise ValueError("Official report requires unique, nonempty occurrence IDs")
    official_rows, diagnostic_rows = [], []
    for row in rows:
        common = {k: row[k] for k in ("query_id", "original_query_id", "question_type") if k in row}
        common["failed"] = bool(row.get("error"))
        if dataset == "multihoprag":
            metrics = {"qa_accuracy": _number(row, "official_qa_accuracy")}
            metrics["retrieval"] = None if row.get("question_type") == "null_query" else {
                name: _number(row, field) for name, field in RANK_FIELDS.items()
            }
        else:
            from utils.hotpotqa import METRICS
            metrics = {key: _number(row, "hotpot_" + key) for key in METRICS}
        official_rows.append({**common, "metrics": metrics})
        auxiliary = {k: row[k] for k in (
            "null_refusal", "answer_attempted", "evidence_fact_recall@4", "evidence_fact_recall@10",
            "exact_fact_recall@10", "all_facts@10", "evidence_doc_precision", "evidence_doc_recall",
            "evidence_doc_f1", "common_retrieval_metrics",
        ) if k in row}
        if dataset == "multihoprag":
            auxiliary.update({"normalized_" + key: row[key] for key in ("answer_em", "answer_f1", "answer_precision", "answer_recall") if key in row})
        diagnostic_rows.append({**common, "metrics": auxiliary})

    report = {
        "schema": "official-dataset-results-v1", "dataset": dataset,
        "strategy": result.get("strategy"), "source_status": result.get("status"),
        "evaluation_scope": result.get("evaluation_scope"),
        "dataset_protocol": result.get("dataset_protocol"),
        "rows": len(rows), "expected_rows": result.get("total_queries", result.get("queries_count")),
        "failed_rows": sum(r["failed"] for r in official_rows),
        "official_implementation": UPSTREAM[dataset],
        "prediction_adapter": {
            "answer": "Shared last-boxed-or-final-label extraction; otherwise the entire response. This adapter precedes official scoring.",
            "support": "Gold-independent complete-corpus-sentence projection from returned passages." if dataset == "hotpotqa" else None,
        },
        "scale": "Native unscaled evaluator values; no display-percent conversion or clipping.",
        "details": official_rows,
    }
    if dataset == "multihoprag":
        def qa_group(group):
            accuracy = statistics.mean(r["metrics"]["qa_accuracy"] for r in group)
            return {"rows": len(group), "precision": accuracy, "recall": accuracy, "f1": accuracy, "accuracy": accuracy}

        by_type = {kind: qa_group([r for r in official_rows if r.get("question_type", "") == kind])
                   for kind in sorted({r.get("question_type", "") for r in official_rows})}
        non_null = [r for r in official_rows if r.get("question_type") != "null_query"]
        retrieval = {"rows": len(non_null), "metrics": {
            name: statistics.mean(r["metrics"]["retrieval"][name] for r in non_null) if non_null else None
            for name in RANK_FIELDS
        }}
        report["qa"] = {"overall": qa_group(official_rows), "by_question_type": by_type,
                        "non_null": qa_group(non_null) if non_null else None}
        report["retrieval"] = retrieval
    else:
        from utils.hotpotqa import METRICS
        report["metrics"] = {key: statistics.mean(r["metrics"][key] for r in official_rows) for key in METRICS}
    diagnostics = {
        "schema": "auxiliary-dataset-results-v1", "dataset": dataset, "rows": len(rows),
        "classification": "Auxiliary diagnostics; these do not replace the dataset's official metrics.",
        "details": diagnostic_rows,
    }
    if "common_retrieval_evaluation" in result:
        diagnostics["common_retrieval_evaluation"] = result["common_retrieval_evaluation"]
    return report, diagnostics


def write_reports(result: dict[str, Any], result_file: Path, *, output_dir: Path | None = None) -> tuple[Path, Path]:
    official, diagnostics = build_reports(result)
    directory = output_dir or result_file.parent
    directory.mkdir(parents=True, exist_ok=True)
    official_path = directory / f"{result_file.stem}.official.json"
    diagnostic_path = directory / f"{result_file.stem}.diagnostics.json"
    for report in (official, diagnostics):
        report["source_result"] = result_file.name
    _write_json(official_path, official)
    _write_json(diagnostic_path, diagnostics)
    return official_path, diagnostic_path
