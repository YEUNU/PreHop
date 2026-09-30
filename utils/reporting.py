from pathlib import Path
from typing import Any

from utils.io import _safe_float, _write_json, _write_jsonl


def trace_detail_row(item: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "idx": item.get("idx", idx),
        "query_id": item.get("query_id", ""),
        "query": item.get("query", ""),
        "interaction_trace": item.get("interaction_trace", []),
    }


def compact_detail_row(item: dict[str, Any], idx: int) -> dict[str, Any]:
    """Canonical JSONL projection; complete evidence remains in the main artifact."""
    trace = item.get("interaction_trace", [])
    return {
        "idx": item.get("idx", idx),
        "query_id": item.get("query_id", ""),
        "query": item.get("query", ""),
        "category": item.get("category", ""),
        "answer": item.get("answer", ""),
        "ground_truth": item.get("ground_truth", ""),
        "final_answer_extracted": item.get("final_answer_extracted", ""),
        "answer_em": _safe_float(item.get("answer_em", -1.0), -1.0),
        "answer_f1": _safe_float(item.get("answer_f1", -1.0), -1.0),
        "answer_precision": _safe_float(item.get("answer_precision", -1.0), -1.0),
        "answer_recall": _safe_float(item.get("answer_recall", -1.0), -1.0),
        "official_answer_em": _safe_float(item.get("official_answer_em", -1.0), -1.0),
        "official_answer_f1": _safe_float(item.get("official_answer_f1", -1.0), -1.0),
        "official_qa_accuracy": _safe_float(item.get("official_qa_accuracy", -1.0), -1.0),
        "null_refusal": _safe_float(item.get("null_refusal", -1.0), -1.0),
        "answer_attempted": _safe_float(item.get("answer_attempted", 0.0)),
        "doc_match": _safe_float(item.get("doc_match", 0.0)),
        "page_match": _safe_float(item.get("page_match", 0.0)),
        "evidence_doc_precision": _safe_float(item.get("evidence_doc_precision", -1.0), -1.0),
        "evidence_doc_recall": _safe_float(item.get("evidence_doc_recall", -1.0), -1.0),
        "evidence_doc_f1": _safe_float(item.get("evidence_doc_f1", -1.0), -1.0),
        "official_hits@4": _safe_float(item.get("official_hits@4", -1.0), -1.0),
        "official_hits@10": _safe_float(item.get("official_hits@10", -1.0), -1.0),
        "official_mrr@10": _safe_float(item.get("official_mrr@10", -1.0), -1.0),
        "official_map@10": _safe_float(item.get("official_map@10", -1.0), -1.0),
        "evidence_fact_recall@4": _safe_float(item.get("evidence_fact_recall@4", -1.0), -1.0),
        "evidence_fact_recall@10": _safe_float(item.get("evidence_fact_recall@10", -1.0), -1.0),
        "exact_fact_recall@10": _safe_float(item.get("exact_fact_recall@10", -1.0), -1.0),
        "all_facts@10": _safe_float(item.get("all_facts@10", -1.0), -1.0),
        **{key: item[key] for key in item if key.startswith("hotpot_")},
        **{key: item[key] for key in ("predicted_supporting_facts", "support_prediction_policy") if key in item},
        "latency": _safe_float(item.get("latency", 0.0)),
        "error": item.get("error", ""),
        "trace_steps": _collect_trace_steps(trace),
    }


def _collect_trace_steps(trace: Any) -> list[str]:
    if not isinstance(trace, list):
        return []
    steps = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        step = str(item.get("step", "") or "").strip()
        if step:
            steps.append(step)
    return steps


def _build_failure_records(details: list[dict[str, Any]], top_k: int = 30) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for idx, item in enumerate(details, start=1):
        primary_score = item.get("primary_answer_score", item.get("answer_em", None))
        has_error = bool(item.get("error"))
        is_failure = has_error
        if primary_score is not None:
            is_failure = is_failure or (_safe_float(primary_score, 0.0) < 1.0)
        if not is_failure:
            continue
        failures.append(
            {
                "rank_hint": idx,
                "query": item.get("query", ""),
                "category": item.get("category", ""),
                "primary_answer_score": _safe_float(primary_score, -1.0),
                "primary_answer_label": item.get("answer_label", "Unscored"),
                "doc_match": _safe_float(item.get("doc_match", 0.0)),
                "page_match": _safe_float(item.get("page_match", 0.0)),
                "latency": _safe_float(item.get("latency", 0.0)),
                "answer": item.get("answer", ""),
                "ground_truth": item.get("ground_truth", ""),
                "error": item.get("error", ""),
                "trace_steps": _collect_trace_steps(item.get("interaction_trace", [])),
            }
        )
    failures.sort(
        key=lambda item: (
            item.get("primary_answer_score", -1.0),
            -item.get("doc_match", 0.0),
            -item.get("latency", 0.0),
        )
    )
    return failures[: max(1, top_k)]


def _write_model_report_artifacts(
    summary: dict[str, Any],
    result_file: Path,
    *,
    preserve_trace_artifacts: bool = False,
) -> None:
    """Write summary, compact details, traces, failures and dataset reports.

    Existing trace artifacts are preserved when requested by checkpoint/resume.
    """
    details = summary.get("details", [])
    if not isinstance(details, list):
        details = []

    stem = result_file.stem
    overview = {key: value for key, value in summary.items() if key != "details"}

    summary_json_file = result_file.with_name(f"{stem}.summary.json")
    details_jsonl_file = result_file.with_name(f"{stem}.details.jsonl")
    traces_jsonl_file = result_file.with_name(f"{stem}.traces.jsonl")
    failures_jsonl_file = result_file.with_name(f"{stem}.failures_topk.jsonl")

    _write_json(summary_json_file, overview)

    dataset = str(summary.get("dataset", "")).lower().replace("-", "").replace("_", "")
    if details and dataset in {"multihoprag", "hotpotqa"}:
        from utils.official_results import write_reports
        write_reports(summary, result_file)

    detail_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for idx, item in enumerate(details, start=1):
        detail_rows.append(compact_detail_row(item, idx))
        trace_rows.append(trace_detail_row(item, idx))
    _write_jsonl(details_jsonl_file, detail_rows)
    if not preserve_trace_artifacts:
        _write_jsonl(traces_jsonl_file, trace_rows)

    failures = _build_failure_records(details, top_k=30)
    _write_jsonl(failures_jsonl_file, failures)
