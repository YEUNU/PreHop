"""Observe co-evidence connectivity; never supply gold to retrieval or linking."""
import argparse
import asyncio
import json
import random
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def map_evidence(nodes, query, dataset, sentence_store=None):
    """Return document-level evidence groups and all observed passage carriers."""
    if dataset == 'hotpotqa':
        from utils.hotpotqa import project_sentences
        facts=set(map(tuple,query['supporting_facts']))
        groups={title:set() for title,_ in facts}
        for node in nodes:
            if node.get('title') not in groups:
                continue
            matched=facts & set(map(tuple,project_sentences([node],sentence_store)))
            for title,_ in matched:
                groups[title].add(node['id'])
    else:
        from utils.metrics import _official_multihoprag_fact_match
        groups={title:set() for title in query.get('evidence_docs',[])}
        for node in nodes:
            title=node.get('title')
            if title in groups and any(_official_multihoprag_fact_match(f,str(node.get('text') or '')) for f in query.get('evidence_facts',[])):
                groups[title].add(node['id'])
    return groups


def adjacency_of(edges):
    adjacency={}
    for source,destination in edges:
        adjacency.setdefault(source,set()).add(destination)
    return adjacency


def connectivity(groups, edges):
    adjacency=edges if isinstance(edges,dict) else adjacency_of(edges)
    titles=list(groups)
    eligible=len(titles)>=2
    observed=[]
    for a in titles:
        for b in titles:
            if a!=b:
                observed.append(any(adjacency.get(s,set()) & groups[b] for s in groups[a]))
    carriers=set().union(*groups.values()) if groups else set()
    cover=[]
    complete=[]
    for start in carriers:
        reach={start}|adjacency.get(start,set())
        n=sum(bool(reach & ids) for ids in groups.values())
        cover.append(n/len(groups))
        complete.append(n==len(groups))
    return {'eligible_multidocument':eligible,'evidence_groups':len(groups),
        'unmapped_groups':sum(not ids for ids in groups.values()),
        'any_inter_evidence_hop':float(any(observed)) if eligible else None,
        'directed_group_pair_coverage':statistics.mean(observed) if eligible else None,
        'oracle_start_one_hop_group_coverage':statistics.mean(cover) if eligible and cover else (0.0 if eligible else None),
        'oracle_start_all_groups_rate':statistics.mean(complete) if eligible and complete else (0.0 if eligible else None),
        'oracle_start_passages':len(carriers)}


def random_edges(nodes, edges, seed):
    """Degree-matched null, excluding the source document and duplicate targets."""
    rng=random.Random(seed)
    byid={n['id']:n for n in nodes}
    degrees={}
    for s,t in set(map(tuple,edges)):
        degrees[s]=degrees.get(s,0)+1
    result=[]
    population=sorted(byid)
    for s,d in sorted(degrees.items()):
        chosen=set()
        for _ in range(max(100,d*20)):
            if len(chosen)==d:break
            t=rng.choice(population)
            if t!=s and byid[t].get('source')!=byid[s].get('source'):chosen.add(t)
        if len(chosen)<d:
            choices=[i for i in population if i not in chosen and i!=s and byid[i].get('source')!=byid[s].get('source')]
            chosen.update(rng.sample(choices,min(d-len(chosen),len(choices))))
        result.extend((s,t) for t in sorted(chosen))
    return result


def summarize(rows):
    keys=['any_inter_evidence_hop','directed_group_pair_coverage','oracle_start_one_hop_group_coverage','oracle_start_all_groups_rate']
    return {k:{'mean':statistics.mean(v) if (v:=[r[k] for r in rows if r.get(k) is not None]) else None,
               'eligible_queries':sum(r.get(k) is not None for r in rows)} for k in keys}


async def export_graph(service,namespace):
    label=f'PR_{namespace}_Chunk'
    nodes=[];after=''
    while True:
        rows=await service.execute_query(f'MATCH (c:{label}) WHERE c.id>$after WITH c ORDER BY c.id LIMIT 512 '
            'RETURN c.id AS id,c.source AS source,c.title AS title,c.text AS text ORDER BY id',{'after':after})
        if not rows:break
        nodes.extend(rows);after=rows[-1]['id']
    edges={}
    for relation in ('HOP_ANSWER','NEXT'):
        rows=await service.execute_query(f'MATCH (s:{label})-[:{relation}]->(t:{label}) RETURN DISTINCT s.id AS src,t.id AS dst')
        edges[relation]=[(r['src'],r['dst']) for r in rows]
    return nodes,edges


async def run(args):
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    from core.neo4j_service import Neo4jService
    service=Neo4jService()
    try:nodes,edges=await export_graph(service,args.namespace)
    finally:await service.close()
    queries=json.loads(args.queries.read_text())
    mappings=[map_evidence(nodes,q,args.dataset,args.sentence_store) for q in queries]
    hop_adj=adjacency_of(edges['HOP_ANSWER'])
    next_adj=adjacency_of(edges['NEXT'])
    union_adj=adjacency_of(edges['HOP_ANSWER']+edges['NEXT'])
    details=[]
    for q,groups in zip(queries,mappings):
        details.append({'query_id':q['_id'],'original_query_id':q.get('original_query_id',q['_id']),
            'question_type':q.get('question_type'),'groups':{k:sorted(v) for k,v in groups.items()},
            **connectivity(groups,hop_adj),
            'next_only':connectivity(groups,next_adj),
            'hop_plus_next':connectivity(groups,union_adj)})
    null=[]
    for seed in range(args.random_repeats):
        shuffled=adjacency_of(random_edges(nodes,edges['HOP_ANSWER'],seed))
        null.append({'seed':seed,'summary':summarize([connectivity(g,shuffled) for g in mappings])})
    report={'namespace':args.namespace,'dataset':args.dataset,'query_rows':len(queries),
        'unique_original_queries':len({r['original_query_id'] for r in details}),
        'interpretation':'co-evidence connectivity diagnostic; gold starts are oracle, not deployed retrieval or annotated reasoning directions',
        'missing_mapping_policy':'retain all query rows; unavailable evidence groups count as unreachable',
        'summary':summarize(details),'question_types':{t:summarize([r for r in details if r['question_type']==t]) for t in sorted({r['question_type'] for r in details})},
        'random_degree_matched_null':null,'details':details}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({'output':str(args.output),'queries':len(details),'summary':report['summary']}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--namespace',required=True);p.add_argument('--queries',type=Path,required=True)
    p.add_argument('--dataset',choices=['multihoprag','hotpotqa'],required=True)
    p.add_argument('--sentence-store',type=Path,default=ROOT/'data/hotpotqa_corpus/sentences.sqlite3')
    p.add_argument('--random-repeats',type=int,default=20);p.add_argument('--output',type=Path,required=True)
    asyncio.run(run(p.parse_args()))


if __name__=='__main__':main()
