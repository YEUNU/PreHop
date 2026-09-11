"""Build isolated Neo4j timing links or replay paired activations."""
import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



async def replay(engine, activations, store, namespace, repetitions=1, warmups=0, groups=None, exclusions=None):
    from models.prehop.connection_timing import expand
    from scripts.ablation_statistics import cluster_interval
    details = []
    for query_index, (qid, starts) in enumerate(activations.items()):
        for _ in range(warmups):
            for arm in ("precomputed","online"):
                await expand(engine, starts, set((exclusions or {}).get(qid,[])), arm, store, namespace)
        for repetition in range(repetitions):
            pair = {}
            order = ("precomputed","online") if (query_index + repetition) % 2 == 0 else ("online","precomputed")
            for arm in order:
                _, pair[arm] = await expand(engine, starts, set((exclusions or {}).get(qid,[])), arm, store, namespace)
            identical = sum(pair["precomputed"]["destinations"][s] == pair["online"]["destinations"][s] for s in sorted(set(starts)))
            jaccard = [len(set(pair["precomputed"]["destinations"][s]) & set(pair["online"]["destinations"][s])) / len(u) if (u := set(pair["precomputed"]["destinations"][s]) | set(pair["online"]["destinations"][s])) else 1.0 for s in sorted(set(starts))]
            details.append({"destination_jaccard":statistics.mean(jaccard) if jaccard else None,"query_id":qid,"repetition":repetition,"order":order,"arms":pair,
                            "identical_starts":identical,"starts":len(set(starts)),
                            "saving_ms":pair["online"]["connection_ms"]-pair["precomputed"]["connection_ms"]})
    per_query = [statistics.mean(r["saving_ms"] for r in details if r["query_id"] == qid) for qid in activations]
    starts = sum(r["starts"] for r in details)
    interval=cluster_interval(per_query,[(groups or {}).get(qid,qid) for qid in activations])
    arm_summary={arm:{"mean_ms":statistics.mean(v) if (v:=[r["arms"][arm]["connection_ms"] for r in details]) else None,
                      "p95_ms":sorted(v)[min(len(v)-1,int(.95*len(v)))] if v else None} for arm in ("precomputed","online")}
    return {"measurement_scope":"fixed_start_connection_replay","includes_direct_retrieval":False,"includes_answer_generation":False,"arm_summary":arm_summary,"paired_cluster_interval":interval,"queries":len(activations),"repetitions":repetitions,"warmups_per_arm_per_query":warmups,
            "zero_start_queries":sum(not v for v in activations.values()),
            "zero_link_start_observations":sum(not destinations for r in details for destinations in r["arms"]["precomputed"]["destinations"].values()),
            "start_observations":starts,
            "identical_start_fraction":sum(r["identical_starts"] for r in details)/starts if starts else None,
            "paired_saving_ms":statistics.mean(per_query) if per_query else None,
            "paired_saving_ms_ci95":interval["ci95"],"details":details}


async def run(args):
    from dotenv import load_dotenv

    from core.prehop_ablation import COMMON, PROFILES
    load_dotenv(ROOT / ".env")
    stats = json.loads(args.index_stats.read_text())
    namespace = stats.get("index_policy",{}).get("index_namespace","")
    reference=json.loads(args.reference_inputs.read_text()) if getattr(args,"reference_inputs",None) else None
    policy = {} if reference else {**COMMON,**PROFILES["question_full"],"ABLATION_Q_MINUS":True,"ABLATION_Q_PLUS":True}
    for key,value in policy.items():
        os.environ["RAG_"+key] = str(value).lower() if isinstance(value,bool) else str(value)
    os.environ.update(RAG_PREHOP_ABLATION_PROFILE="question_full",RAG_PAPER_MODE="false",
                      RAG_ABLATION_REUSE_EXISTING_INDEX="true",RAG_INDEX_NAMESPACE=namespace,
                      RAG_CONNECTION_TIMING_MODE="",RAG_CONNECTION_TIMING_STORE="",RAG_LLM_SEED="")
    reference=json.loads(args.reference_inputs.read_text()) if getattr(args,'reference_inputs',None) else None
    if reference:
        ablation=reference['reference_settings'].get('ablation') or {}
        for key in (*COMMON,'HYPO_CHANNEL_VARIANT','HOP_LINK_VARIANT','QPLUS_HOP_ACTIVATION'):
            value=ablation.get(key.lower())
            if value is not None:os.environ['RAG_'+key]=str(value).lower() if isinstance(value,bool) else str(value)
        os.environ['RAG_HOP_SEED_POLICY']=reference['start_policy']
        os.environ['RAG_PREHOP_ABLATION_PROFILE']=ablation.get('representation_ablation','')
    from core.neo4j_service import Neo4jService
    from models.prehop.connection_timing import build
    from models.prehop.graphrag import GraphRAG
    engine = GraphRAG(strategy="prehop",corpus_tag=stats["corpus_tag"])
    try:
        await engine.retry_query("MATCH (m:RAGIndexSnapshot {strategy:'prehop', index_namespace:$namespace}) RETURN m.status AS status",{"namespace":namespace})
        if args.mode == "build":
            result = await build(engine,args.store,namespace)
        else:
            activations = reference["activations"] if reference else json.loads(args.activations.read_text())
            queries=json.loads(args.queries.read_text()) if args.queries else []
            groups={q['_id']:q.get('original_query_id',q['_id']) for q in queries}
            result = await replay(engine,activations,args.store,namespace,args.repetitions,args.warmups,groups,
                                  reference.get('excluded') if reference else None)
            result['namespace']=namespace
            if reference:
                result['reference_settings']=reference['reference_settings']
                result['reference_starts']=activations
                result['reference_issues']=list(reference['reference_issues'])
                if reference['namespaces']!=[namespace]:result['reference_issues'].append('reference_graph_namespace_differs')
                result['reference_result']=reference['reference_result']
                agreement={}
                for row in result['details']:
                    qid=row['query_id'];old=reference['historical_destinations'].get(qid)
                    if old is not None:
                        current=row['arms']['precomputed']['destinations']
                        same=all(set(current.get(k,[]))==set(old.get(k,[])) for k in set(current)|set(old))
                        agreement[qid]=agreement.get(qid,True) and same
                result['historical_destination_agreement']=agreement
            preparation=json.loads(args.store.read_text())
            saving=result.get("paired_saving_ms")
            result["time_break_even_queries"]=(preparation["connection_build_seconds"]/(saving/1000)) if saving and saving>0 else None
            result["preparation_seconds"]=preparation["connection_build_seconds"]
            result["activations_sha256"] = hashlib.sha256((args.reference_inputs if reference else args.activations).read_bytes()).hexdigest()
        from scripts.datasets.hotpotqa_common import digest_file
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
    parser.add_argument("--reference-inputs",type=Path)
    parser.add_argument("--queries",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repetitions",type=int,default=1)
    parser.add_argument("--warmups",type=int,default=0)
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
