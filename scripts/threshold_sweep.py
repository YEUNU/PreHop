"""Offline HOP-edge cosine-threshold sensitivity sweep.

`RAG_HOP_THRESHOLD` (τ_hop) gates offline
  HOP-edge construction (models/prehop/indexing/hop_edges.py). Each sweep
  point rebuilds ONLY the HOP edges for the given corpus tag (`main.py
  --mode hop_rebuild` — chunks/Q-/Q+/embeddings untouched, see CLAUDE.md
  "Re-index note"), then benchmarks against the same queries file.

Each sweep point runs in its own subprocess with the threshold set via env
var BEFORE Python starts, because RAGConfig reads env vars once at class-
definition time (core/config.py) — mutating os.environ mid-process would not
take effect. Each point gets a deterministic RAG_BENCHMARK_TIMESTAMP so the
result file path is predictable without scraping subprocess stdout.

Usage:
  python scripts/threshold_sweep.py \
    --values 0.75,0.78,0.82,0.85,0.90 --corpus-tag multihoprag \
    --queries_file data/multihoprag_sample200_queries.json \
    --model gemma-4-31b-it --out-dir data/results/hop_sweep

Outputs: <out-dir>/hop_sweep.json (full) + <out-dir>/hop_sweep.csv
(tidy: value, total_hop_edges, avg_hop_out_degree_per_eligible_chunk,
avg_llm_judge_score, avg_mrr@10, avg_hits@10, avg_latency, avg_retrieve_ms,
avg_traversal_ms, avg_synthesis_ms).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

HOP_ENV = "RAG_HOP_THRESHOLD"

QUALITY_KEYS = [
    "avg_llm_judge_score",
    "avg_hallucination",
    "avg_mrr@10",
    "avg_map@10",
    "avg_hits@4",
    "avg_hits@10",
    "avg_evidence_doc_recall",
    "avg_doc_match",
    "avg_latency",
    "avg_retrieve_ms",
    "avg_traversal_ms",
    "avg_synthesis_ms",
]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value)


def _run(cmd: list[str], env: dict) -> None:
    print(
        f"$ {' '.join(cmd)}  (env override: { {k: v for k, v in env.items() if k in (HOP_ENV, 'RAG_BENCHMARK_TIMESTAMP')} })"
    )
    subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}, check=True)


def _sweep_point_hop(value: str, corpus_tag: str, queries_file: str, model: str, timestamp: str) -> dict:
    env = {HOP_ENV: value, "RAG_BENCHMARK_TIMESTAMP": timestamp}
    _run([PYTHON, "main.py", "--mode", "hop_rebuild", "--strategy", "prehop", "--corpus-tag", corpus_tag], env)

    stats_files = sorted((ROOT / "data" / "index_stats").glob(f"prehop_{corpus_tag}_*.json"))
    graph_stats = json.loads(stats_files[-1].read_text()) if stats_files else {}

    _run(
        [
            PYTHON,
            "main.py",
            "--mode",
            "benchmark",
            "--strategy",
            "prehop",
            "--corpus-tag",
            corpus_tag,
            "--queries_file",
            queries_file,
            "--model",
            model,
        ],
        env,
    )

    result_path = ROOT / "data" / "results" / timestamp / "prehop" / corpus_tag / f"prehop_{corpus_tag}.json"
    summary = json.loads(result_path.read_text()) if result_path.exists() else {}
    return {"value": value, "graph_stats": graph_stats, "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", required=True, help="comma-separated threshold values, e.g. 0.75,0.80,0.85")
    ap.add_argument("--corpus-tag", required=True)
    ap.add_argument("--queries_file", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    values = [v.strip() for v in args.values.split(",") if v.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points = []
    for value in values:
        timestamp = f"sweep_hop_{_slug(value)}"
        point = _sweep_point_hop(value, args.corpus_tag, args.queries_file, args.model, timestamp)
        points.append(point)

    (out_dir / "hop_sweep.json").write_text(json.dumps(points, indent=2), encoding="utf-8")

    with (out_dir / "hop_sweep.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["value", "total_hop_edges", "avg_hop_out_degree_per_eligible_chunk", *QUALITY_KEYS])
        for point in points:
            gs = point.get("graph_stats") or {}
            s = point.get("summary") or {}
            w.writerow(
                [
                    point["value"],
                    gs.get("total_hop_edges", ""),
                    gs.get("avg_hop_out_degree_per_eligible_chunk", ""),
                    *[s.get(k, "") for k in QUALITY_KEYS],
                ]
            )

    print(f"\nSweep complete -> {out_dir}/hop_sweep.{{json,csv}}")
    for point in points:
        s = point.get("summary") or {}
        print(
            f"  hop={point['value']}: judge={s.get('avg_llm_judge_score', 'n/a')} "
            f"mrr@10={s.get('avg_mrr@10', 'n/a')} latency={s.get('avg_latency', 'n/a')}"
        )


if __name__ == "__main__":
    main()
