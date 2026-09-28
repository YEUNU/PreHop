"""Save dataset-specific official metrics separately from auxiliary diagnostics."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.io import _write_json
from utils.official_results import build_reports


def export(result_path: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    raw = result_path.read_bytes()
    official, diagnostics = build_reports(json.loads(raw))
    directory = output_dir or result_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, report in [("official", official), ("diagnostics", diagnostics)]:
        report.update(source_result=str(result_path), source_sha256=hashlib.sha256(raw).hexdigest())
        path = directory / f"{result_path.stem}.{suffix}.json"
        _write_json(path, report)
        paths.append(path)
    return tuple(paths)


def export_comparison(result_paths: list[Path], output_dir: Path) -> list[Path]:
    """Export explicitly selected complete results, grouped by dataset.

    This aggregates recorded scores; it does not generate or rescore answers.
    Each dataset/strategy has one source and a matching question population.
    """
    groups = {}
    populations = {}
    for path in result_paths:
        raw = path.read_bytes()
        result = json.loads(raw)
        rows = result.get("details", [])
        if (result.get("status") not in {"completed", "completed_unadmitted"}
                or result.get("evaluation_scope") not in {"full_benchmark", "released_benchmark"}
                or len(rows) != result.get("total_queries", result.get("queries_count"))):
            raise ValueError(f"Comparison requires a complete result: {path}")
        official, _ = build_reports(result)
        dataset, strategy = official["dataset"], official["strategy"]
        if not strategy or Path(strategy).name != strategy or strategy in {".", ".."}:
            raise ValueError(f"Missing or invalid strategy: {path}")
        group = groups.setdefault(dataset, {})
        if strategy in group:
            raise ValueError(f"Select one final result for {dataset}/{strategy}")
        population = (
            result.get("dataset_protocol"), result.get("evaluation_scope"),
            {r["query_id"]: (
                r.get("original_query_id") or r["query_id"], r.get("question_type"), r.get("ground_truth"),
                (r.get("expected_sources") or {}).get(
                    "facts" if dataset == "multihoprag" else "supporting_facts", []),
            ) for r in rows},
        )
        if dataset in populations and populations[dataset] != population:
            raise ValueError(f"Mismatched question population or annotations: {path}")
        populations[dataset] = population
        official.update(source_result=str(path), source_sha256=hashlib.sha256(raw).hexdigest())
        group[strategy] = (path.stem, official)

    written = []
    for dataset, group in groups.items():
        directory = output_dir / dataset
        summaries, csv_rows = [], []
        for strategy, (stem, official) in group.items():
            path = directory / strategy / f"{stem}.official.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(path, official)
            written.append(path)
            summary = {k: v for k, v in official.items() if k != "details"}
            summaries.append(summary)
            flat = {k: official[k] for k in ("strategy", "rows", "failed_rows", "source_result", "source_sha256")}

            def flatten(prefix, value, target):
                if isinstance(value, dict):
                    for key, child in value.items():
                        flatten(f"{prefix}.{key}", child, target)
                else:
                    target[prefix] = value

            for key in ("qa", "retrieval", "metrics"):
                if key in official:
                    flatten(key, official[key], flat)
            csv_rows.append(flat)
        path = directory / "comparison.json"
        _write_json(path, {"dataset": dataset, "results": summaries})
        written.append(path)
        path = directory / "comparison.csv"
        fields = list(dict.fromkeys(k for row in csv_rows for k in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, nargs="+", help="Explicit final result paths; no automatic run discovery")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if len(args.result) == 1:
        paths = export(args.result[0], args.output_dir)
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required when exporting a comparison")
        paths = export_comparison(args.result, args.output_dir)
    print(json.dumps({"written": [str(p) for p in paths]}))


if __name__ == "__main__":
    main()
