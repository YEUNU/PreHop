"""Run common direct retrieval once, without graph expansion or answer generation."""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


async def run(args):
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    from core.prehop_ablation import COMMON, PROFILES
    stats=json.loads(args.index_stats.read_text())
    for k,v in {**COMMON,**PROFILES[args.profile],'ABLATION_Q_MINUS':True,'ABLATION_Q_PLUS':True}.items():
        os.environ['RAG_'+k]=str(v).lower() if isinstance(v,bool) else str(v)
    os.environ.update(RAG_PAPER_MODE='false',RAG_PREHOP_ABLATION_PROFILE=args.profile,
        RAG_INDEX_NAMESPACE=stats['index_policy']['index_namespace'],RAG_PREHOP_TRACE='false',
        RAG_CONNECTION_TIMING_MODE='',RAG_ABLATION_DIRECT_INPUTS='')
    from core.neo4j_service import Neo4jService
    from models.prehop.ablation_inputs import write_input
    from models.prehop.graphrag import GraphRAG
    engine=GraphRAG(strategy='prehop',corpus_tag=stats['corpus_tag'])
    queries=json.loads(args.queries.read_text());semaphore=asyncio.Semaphore(args.workers)
    details={};completed=0
    async def one(query):
        nonlocal completed
        async with semaphore:
            text=engine._normalize_entity_term(query['query']) or query['query']
            started=time.perf_counter()
            try:
                embedding=await engine.llm.get_embedding(text)
                _,nodes=await engine._retrieve_with_candidate_pool(text,12,query_embedding=embedding,select_final=False)
                value={'query':text,'query_embedding':embedding,'base_candidates':nodes,
                    'original_retrieve_ms':(time.perf_counter()-started)*1000}
                await asyncio.to_thread(write_input,args.output,text,value)
                details[query['_id']]={'starts':[n['id'] for n in nodes if not n.get('role_body_owner_only')],
                    'original_query_id':query.get('original_query_id',query['_id']), 'error':None}
            except Exception as error:
                logging.getLogger(__name__).exception("Common retrieval failed for %s",query["_id"])
                details[query['_id']]={'starts':[],'original_query_id':query.get('original_query_id',query['_id']),
                    'error':type(error).__name__+': '+str(error)}
            completed+=1
            if completed%50==0:print(f'Common retrieval: {completed}/{len(queries)}',flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        # Duplicate texts share one deterministic frozen retrieval realization.
        unique={engine._normalize_entity_term(q['query']):q for q in queries}
        await asyncio.gather(*(one(q) for q in unique.values()))
        bytext={engine._normalize_entity_term(q['query']):details[q['_id']] for q in unique.values()}
        details={q['_id']:dict(bytext[engine._normalize_entity_term(q['query'])],original_query_id=q.get('original_query_id',q['_id'])) for q in queries}
        result={'profile':args.profile,'source_index':str(args.index_stats.resolve()),'queries':len(queries),
            'unique_retrievals':len(unique),'failed_rows':sum(bool(r['error']) for r in details.values()),
            'latency_scope':'frozen-prefix downstream benchmark; shared retrieval preparation reported separately',
            'details':details}
        (args.output/'manifest.json').write_text(json.dumps(result,indent=2))
        (args.output/'activations.json').write_text(json.dumps({k:v['starts'] for k,v in details.items() if not v['error']}))
        print(json.dumps({'queries':len(queries),'failed_rows':result['failed_rows'],'output':str(args.output)}))
    finally:await Neo4jService.global_close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index-stats',type=Path,required=True);p.add_argument('--queries',type=Path,required=True)
    p.add_argument('--profile',choices=['question_full','question_body'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    asyncio.run(run(p.parse_args()))


if __name__=='__main__':main()
