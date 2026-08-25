"""Merge interrupted matrix fragments into cumulative timing tables.

Each matrix run writes an ``attempt_journal.jsonl``.  A Ctrl-C records the
elapsed partial attempt, while a later run records the resumed attempt.  This
tool sums those fragments per dataset/strategy and preserves the latest
terminal result (graph statistics/artifact measurements) for the structural
columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.run_index_matrix import _phase_timing, _write_json, _write_tables
except ModuleNotFoundError:  # direct ``python scripts/merge_index_matrix_runs.py``
    from run_index_matrix import _phase_timing, _write_json, _write_tables


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _fragments(run_dir: Path) -> list[dict[str, Any]]:
    journal = _read_jsonl(run_dir / "attempt_journal.jsonl")
    fragments = [row["result"] for row in journal if row.get("event") == "attempt_finished" and isinstance(row.get("result"), dict)]
    if fragments:
        return fragments
    results_path = run_dir / "results.json"
    if results_path.is_file():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    return []


def merge(run_dirs: list[Path]) -> list[dict[str, Any]]:
    by_target: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for run_dir in run_dirs:
        for fragment in _fragments(run_dir):
            target = str(fragment.get("target") or "")
            if not target:
                continue
            identity = (str(fragment.get("run_id") or run_dir.name), int(fragment.get("attempt", 1)))
            by_target.setdefault(target, {})[identity] = fragment

    merged: list[dict[str, Any]] = []
    for target, identity_map in sorted(by_target.items()):
        fragments = sorted(identity_map.values(), key=lambda item: float(item.get("finished_at", item.get("started_at", 0))))
        terminal = next((item for item in reversed(fragments) if item.get("status") in {"complete", "measurement_failed"}), fragments[-1])
        result = dict(terminal)
        result["fragment_count"] = len(fragments)
        result["measurement_attempts"] = len(fragments)
        result["elapsed_seconds"] = sum(float(item.get("elapsed_seconds", 0.0)) for item in fragments)
        result["user_cpu_seconds"] = sum(float(item.get("user_cpu_seconds", 0.0)) for item in fragments)
        result["system_cpu_seconds"] = sum(float(item.get("system_cpu_seconds", 0.0)) for item in fragments)
        result["max_rss_bytes"] = max((int(item.get("max_rss_bytes", 0) or 0) for item in fragments), default=0)
        phase_totals: dict[str, float] = {}
        for item in fragments:
            for phase, seconds in _phase_timing(item).items():
                phase_totals[phase] = phase_totals.get(phase, 0.0) + seconds
        result["phase_timing_seconds"] = dict(sorted(phase_totals.items()))
        result["fragment_statuses"] = [item.get("status") for item in fragments]
        result["files_per_second"] = (
            result["input_file_count"] / result["elapsed_seconds"] if result["elapsed_seconds"] else 0.0
        )
        merged.append(result)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="completed or interrupted artifact run directories")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dirs = [path.resolve() for path in args.run_dirs]
    missing = [str(path) for path in run_dirs if not path.is_dir()]
    if missing:
        parser.error(f"run directory not found: {missing}")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = merge(run_dirs)
    if not results:
        parser.error("no attempt fragments or results found in the supplied run directories")
    _write_json(out_dir / "results.json", results)
    _write_json(
        out_dir / "manifest.json",
        {"merged_from": [str(path) for path in run_dirs], "target_count": len(results)},
    )
    _write_tables(out_dir, results)
    print(f"Merged {len(results)} targets into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
