"""Persist and resume query checkpoints without reinterpreting recorded results."""
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from utils.io import _write_json, _write_jsonl
from utils.reporting import _compute_stage_diagnostics, _write_model_report_artifacts, trace_detail_row

logger = logging.getLogger("Prehop")


def _slim_details(details: list | None) -> list:
    """Strip interaction traces and private fields from the main result JSON."""
    return [
        {k: v for k, v in d.items() if k != "interaction_trace" and not k.startswith("_")} if isinstance(d, dict) else d
        for d in (details or [])
    ]

def _write_slim_main(s: dict[str, Any], result_file: Path) -> None:
    _write_json(result_file, {**s, "details": _slim_details(s.get("details"))})

def _read_jsonl_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            rows.append(row)
    return rows

def _resume_benchmark_rows(
    result_file: Path,
    benchmark_data: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Restore saved rows and traces in the current input order.

    Terminal runtime-error rows are retained; only unexecuted queries resume.
    Prior provenance is retained without rechecking configuration equality.
    """
    prior = json.loads(result_file.read_text(encoding="utf-8"))
    manifest_position = {str(item["_id"]): idx for idx, item in enumerate(benchmark_data, start=1)}
    prior_rows = prior.get("details")

    trace_file = result_file.with_name(f"{result_file.stem}.traces.jsonl")
    trace_rows = _read_jsonl_file(trace_file)

    # Traces are published before the main commit point. A crash between those
    # writes may leave a newer trace file; only committed query IDs are resumed.
    traces_by_id = {}
    for trace_row in trace_rows:
        query_id = str(trace_row.get("query_id") or "")
        if query_id in traces_by_id:
            raise ValueError(f"Duplicate trace query ID: {query_id}")
        traces_by_id[query_id] = trace_row
    retained: list[dict[str, Any]] = []
    for raw_row in prior_rows:
        query_id = str(raw_row.get("query_id") or "")
        if query_id not in traces_by_id:
            raise ValueError(f"Missing trace for committed query ID: {query_id}")
        trace_row = traces_by_id[query_id]
        expected_idx = manifest_position[query_id]
        retained.append({**raw_row, "idx": expected_idx, "interaction_trace": trace_row.get("interaction_trace", [])})

    retained.sort(key=lambda row: int(row["idx"]))

    retained_ids = sorted(str(row["query_id"]) for row in retained)
    resume_metadata = {
        "requested": True,
        "prior_status": prior.get("status"),
        "initial_rows": len(prior_rows),
        "retained_rows": len(retained),
        "rerun_error_rows": 0,
        "retained_query_ids_sha256": hashlib.sha256("\n".join(retained_ids).encode()).hexdigest(),
        "prior_query_provenance": prior.get("query_provenance"),
        "prior_evaluation_provenance": prior.get("evaluation_provenance"),
    }
    return retained, resume_metadata

def _benchmark_checkpoint_due(completed: int, total: int, every: int) -> bool:
    """Return whether an incremental artifact checkpoint is due."""
    return completed == total or completed % every == 0

def _order_benchmark_rows(rows: list[dict[str, Any]]) -> None:
    """Normalize completed rows to immutable input-manifest order in place."""
    rows.sort(key=lambda row: int(row["idx"]))


def write_checkpoint(summary: dict[str, Any], result_file: Path) -> None:
    """Publish mandatory resume evidence before optional derived reports."""
    rows = summary.get("details") or []
    trace_file = result_file.with_name(f"{result_file.stem}.traces.jsonl")
    _write_jsonl(trace_file, [trace_detail_row(row, idx) for idx, row in enumerate(rows, start=1)])
    _write_slim_main(summary, result_file)
    try:
        _write_model_report_artifacts(summary, result_file, preserve_trace_artifacts=True)
        _write_json(result_file.with_name(f"{result_file.stem}.stage_diagnostics.json"), _compute_stage_diagnostics(rows))
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Checkpoint saved; failed to write derived reports for %s: %s", result_file, exc)
