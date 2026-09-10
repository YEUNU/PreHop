"""Plan isolated ablations; service work requires an explicit --execute."""

import argparse
import asyncio
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
        "ABLATION_Q_MINUS": args.profile != "body_body",
        "ABLATION_Q_PLUS": args.profile != "body_body",
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
            "RAG_JUDGE_ENABLED": "false",
            "RAG_LLM_SEED": "",
            "RAG_ABLATION_REUSE_EXISTING_INDEX": str(reuse).lower(),
        }
    )
    env["RAG_ABLATION_DIRECT_INPUTS"] = str(Path(args.direct_inputs).resolve()) if getattr(args,"direct_inputs",None) else ""
    timing = getattr(args, "connection_timing", None)
    if timing:
        env.update(RAG_CONNECTION_TIMING_MODE=timing,
                   RAG_CONNECTION_TIMING_STORE=str(Path(args.timing_store).resolve()))
    else:
        env.update(RAG_CONNECTION_TIMING_MODE="", RAG_CONNECTION_TIMING_STORE="")
    if args.reference:
        env["RAG_BODY_LINK_REFERENCE"] = str(Path(args.reference).resolve())
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
    clone_source = getattr(args, "clone_body_from", None)
    if clone_source:
        command = [sys.executable, str(ROOT / "scripts/clone_prehop_body.py"),
                   str(Path(clone_source).resolve()), str(Path(args.dataset).resolve())]
    return {
        "environment": env,
        "command": command,
        "output": str(ROOT / "data/results" / env["RAG_BENCHMARK_TIMESTAMP"]),
    }


async def export_reference(namespace, output):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from core.neo4j_service import Neo4jService
    from models.prehop.indexing.body_links import make_reference, read_body_rows

    service = Neo4jService()

    class Reader:
        chunk_label = f"PR_{namespace}_Chunk"
        retry_query = staticmethod(service.execute_query)

    try:
        await service.execute_query(
            "MATCH (m:RAGIndexSnapshot {strategy:'prehop', index_namespace:$namespace}) RETURN m.status AS status",
            {"namespace": namespace},
        )
        await service.execute_query(
            f"MATCH (c:PR_{namespace}_Chunk) OPTIONAL MATCH (c)-[:HAS_Q_PLUS]->(q) RETURN count(q) AS questions"
        )
        await service.execute_query(
            f"MATCH (:PR_{namespace}_Chunk)-[h:HOP_ANSWER]->() RETURN DISTINCT h.type AS type"
        )
        reference = {"schema": "prehop-body-link-reference-v1", "namespace": namespace, "nodes": {}}
        async for row in read_body_rows(Reader(), degrees=True):
            reference["nodes"].update(make_reference(namespace, [row])["nodes"])

        def write_reference():
            with Path(output).open("x") as handle:
                json.dump(reference, handle, ensure_ascii=False, sort_keys=True, indent=2)

        await asyncio.to_thread(write_reference)
    finally:
        await service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--mode", choices=("index", "benchmark", "export-reference"), required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--corpus-tag", choices=("multihoprag", "hotpotqa"))
    parser.add_argument("--dataset")
    parser.add_argument("--queries")
    parser.add_argument("--index-stats")
    parser.add_argument("--reference")
    parser.add_argument("--connection-timing", choices=("precomputed", "online"))
    parser.add_argument("--direct-inputs", help="Frozen direct retrieval directory for controlled quality comparisons")
    parser.add_argument("--timing-store", help="Newly built matched destination snapshot shared by timing arms")
    parser.add_argument("--clone-body-from", help="Completed question index statistics; copy stored bodies without inference")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reuse-existing-index", action="store_true", help="Read an existing question index for A/B benchmarks only")
    args = parser.parse_args()
    if args.mode == "export-reference":
        if not args.reference:
            parser.error("export-reference requires --reference (new output file)")
        if args.execute:
            asyncio.run(export_reference(args.namespace, args.reference))
        else:
            print(json.dumps({"read_only_export": args.namespace, "new_file": args.reference}, indent=2))
        return
    if not all((args.profile, args.run_id, args.corpus_tag, args.dataset, args.queries)):
        parser.error("index/benchmark require --profile, --run-id, --corpus-tag, --dataset and --queries")
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
