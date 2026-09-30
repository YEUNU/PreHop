"""Plan isolated ablations; service work requires an explicit --execute."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.prehop_ablation import COMMON, PROFILES


def plan(args):
    reuse = getattr(args, "reuse_existing_index", False)
    overrides = {
        **COMMON,
        **PROFILES[args.profile],
        "HOP_SEMANTIC_VARIANT": getattr(args, "hop_semantic_variant", COMMON["HOP_SEMANTIC_VARIANT"]),
    }
    env = {
        "RAG_" + key: str(value).lower() if isinstance(value, bool) else str(value)
        for key, value in overrides.items()
        if key not in {"DEFAULT_TOP_K"}
    }
    env.update(
        {
            "RAG_PREHOP_ABLATION_PROFILE": args.profile,
            "RAG_PAPER_MODE": "false",
            "RAG_INDEX_NAMESPACE": args.namespace,
            "RAG_RUN_ID": args.run_id,
            "RAG_BENCHMARK_TIMESTAMP": f"ablations/{args.run_id}/{args.profile}",
            "RAG_LLM_SEED": "",
            "RAG_ABLATION_REUSE_EXISTING_INDEX": str(reuse).lower(),
        }
    )
    env["RAG_ABLATION_DIRECT_INPUTS"] = str(Path(args.direct_inputs).resolve()) if getattr(args,"direct_inputs",None) else ""
    if args.index_stats:
        env["RAG_INDEX_STATS_PATH"] = str(Path(args.index_stats).resolve())
    if args.index_stats and Path(args.index_stats).is_file():
        stats = json.loads(Path(args.index_stats).read_text())
        env["RAG_RUN_ID"] = stats["run_id"]
    command = [
        sys.executable,
        str(ROOT / "main.py"),
        "--mode",
        args.mode,
        "--strategy",
        "prehop",
        "--corpus-tag",
        args.corpus_tag,
        "--dataset",
        str(Path(args.dataset).resolve()),
        "--queries_file",
        str(Path(args.queries).resolve()),
    ]
    return {
        "environment": env,
        "command": command,
        "output": str(ROOT / "data/results" / env["RAG_BENCHMARK_TIMESTAMP"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--mode", choices=("benchmark",), required=True)
    parser.add_argument("--hop-semantic-variant", choices=("body_only", "body_bridge_min"), default="body_only")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--corpus-tag", choices=("multihoprag", "hotpotqa"))
    parser.add_argument("--dataset")
    parser.add_argument("--queries")
    parser.add_argument("--index-stats")
    parser.add_argument("--direct-inputs", help="Frozen direct retrieval directory for controlled quality comparisons")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reuse-existing-index", action="store_true", help="Read an existing question index for A/B benchmarks only")
    args = parser.parse_args()
    if not all((args.profile, args.run_id, args.corpus_tag, args.dataset, args.queries)):
        parser.error("benchmark requires --profile, --run-id, --corpus-tag, --dataset and --queries")
    task = plan(args)
    print(json.dumps(task, indent=2), flush=True)
    if not args.execute:
        return
    env = dict(os.environ)
    # Do not inherit a campaign's resume/reuse/output routing into this run.
    for key in ("RAG_INDEX_REUSE_LINK", "RAG_INDEX_STATS_PATH", "RAG_BENCHMARK_RESUME"):
        env.pop(key, None)
    env.update(task["environment"])
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "ablation_plan.json").write_text(json.dumps(task, indent=2))
    subprocess.run(task["command"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
