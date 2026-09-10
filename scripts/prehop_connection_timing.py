"""Build matched links or replay paired activations, without changing a graph."""
import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def bootstrap(values, seed=42, repeats=10000):
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return [means[int(.025*repeats)], means[min(repeats-1,int(.975*repeats))]]


async def replay(engine, activations, store, namespace, repetitions=3, warmups=1):
    from models.prehop.connection_timing import expand
    details = []
    for qid, starts in activations.items():
        for _ in range(warmups):
            for arm in ("precomputed","online"):
                await expand(engine, starts, set(), arm, store, namespace)
        for repetition in range(repetitions):
            pair = {}
            order = ("precomputed","online") if repetition % 2 == 0 else ("online","precomputed")
            for arm in order:
                _, pair[arm] = await expand(engine, starts, set(), arm, store, namespace)
            identical = sum(pair["precomputed"]["destinations"][s] == pair["online"]["destinations"][s] for s in sorted(set(starts)))
            details.append({"query_id":qid,"repetition":repetition,"order":order,"arms":pair,
                            "identical_starts":identical,"starts":len(set(starts)),
                            "saving_ms":pair["online"]["connection_ms"]-pair["precomputed"]["connection_ms"]})
    per_query = [statistics.mean(r["saving_ms"] for r in details if r["query_id"] == qid) for qid in activations]
    starts = sum(r["starts"] for r in details)
    return {"queries":len(activations),"repetitions":repetitions,"warmups_per_arm_per_query":warmups,
            "identical_start_fraction":sum(r["identical_starts"] for r in details)/starts if starts else None,
            "paired_saving_ms":statistics.mean(per_query) if per_query else None,
            "paired_saving_ms_ci95":bootstrap(per_query),"details":details}


async def run(args):
    from dotenv import load_dotenv

    from core.prehop_ablation import COMMON, PROFILES
    load_dotenv(ROOT / ".env")
    stats = json.loads(args.index_stats.read_text())
    namespace = stats.get("index_policy",{}).get("index_namespace","")
    if stats.get("status") != "complete" or stats.get("strategy") != "prehop" or not re.fullmatch(r"[A-Za-z0-9_]+",namespace):
        raise ValueError("Use completed Prehop question-index statistics")
    for key,value in {**COMMON,**PROFILES["question_full"],"ABLATION_Q_MINUS":True,"ABLATION_Q_PLUS":True}.items():
        os.environ["RAG_"+key] = str(value).lower() if isinstance(value,bool) else str(value)
    os.environ.update(RAG_PREHOP_ABLATION_PROFILE="question_full",RAG_PAPER_MODE="false",
                      RAG_ABLATION_REUSE_EXISTING_INDEX="true",RAG_INDEX_NAMESPACE=namespace,
                      RAG_CONNECTION_TIMING_MODE="",RAG_CONNECTION_TIMING_STORE="",RAG_LLM_SEED="")
    from core.neo4j_service import Neo4jService
    from core.prehop_ablation import validate_ablation_index_policy
    from models.prehop.connection_timing import build, graph_fingerprint, metadata
    from models.prehop.graphrag import GraphRAG
    validate_ablation_index_policy(stats["index_policy"],stats["index_policy_sha256"],stats["corpus_tag"])
    engine = GraphRAG(strategy="prehop",corpus_tag=stats["corpus_tag"])
    try:
        snapshots = await engine.retry_query("MATCH (m:RAGIndexSnapshot {strategy:'prehop', index_namespace:$namespace}) RETURN m.status AS status",{"namespace":namespace})
        if not snapshots or any(r["status"] != "complete" for r in snapshots):
            raise ValueError("Source graph is not complete")
        if args.mode == "build":
            result = await build(engine,args.store,namespace)
        else:
            if metadata(args.store,namespace).get("graph_fingerprint") != await graph_fingerprint(engine):
                raise ValueError("Timing snapshot differs from the current frozen graph")
            if not args.activations:
                raise ValueError("Replay requires --activations JSON mapping query IDs to starting passage IDs")
            activations = json.loads(args.activations.read_text())
            if not isinstance(activations,dict) or not activations or any(not isinstance(v,list) or any(not isinstance(x,str) for x in v) for v in activations.values()):
                raise ValueError("Invalid activation manifest")
            result = await replay(engine,activations,args.store,namespace,args.repetitions,args.warmups)
            result["activations_sha256"] = hashlib.sha256(args.activations.read_bytes()).hexdigest()
        from scripts.datasets.prepare_hotpotqa import digest_file
        result.update(index_stats_sha256=digest_file(args.index_stats),
                      timing_store_sha256=digest_file(args.store))
        with args.output.open("x") as f:
            json.dump(result,f,indent=2)
    finally:
        await Neo4jService.global_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("build","replay"),required=True)
    parser.add_argument("--index-stats",type=Path,required=True)
    parser.add_argument("--store",type=Path,required=True)
    parser.add_argument("--activations",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repetitions",type=int,default=3)
    parser.add_argument("--warmups",type=int,default=1)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0 or args.output.exists():
        parser.error("Require positive repetitions, nonnegative warmups, and a new output")
    if args.execute:
        asyncio.run(run(args))
    else:
        print(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2))


if __name__ == "__main__":
    main()
