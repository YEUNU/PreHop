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
from scripts.ablation_statistics import cluster_interval

METRICS = ('any_inter_evidence_hop','directed_group_pair_coverage',
           'oracle_start_one_hop_group_coverage','oracle_start_all_groups_rate',
           'oracle_start_other_document_recall')
SCOPES = ('hop_only','next_only','hop_plus_next')
CONTRACT = 'co-evidence-connectivity-v3'



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


def connectivity(groups, edges, source_by_id=None):
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
    complete=[];other_recall=[]
    owners={node:title for title,ids in groups.items() for node in ids}
    sources=source_by_id if source_by_id is not None else owners
    for start in carriers:
        reach={start}|adjacency.get(start,set())
        n=sum(bool(reach & ids) for ids in groups.values())
        cover.append(n/len(groups))
        complete.append(n==len(groups))
        # Remove the entire source document, including non-gold sibling chunks.
        origin=sources.get(start,start)
        other_groups=[ids for title,ids in groups.items() if start not in ids]
        external={node for node in adjacency.get(start,set()) if sources.get(node,node)!=origin}
        if other_groups:
            other_recall.append(sum(bool(external & ids) for ids in other_groups)/len(other_groups))
    return {'eligible_multidocument':eligible,'evidence_groups':len(groups),
        'unmapped_groups':sum(not ids for ids in groups.values()),
        'any_inter_evidence_hop':float(any(observed)) if eligible else None,
        'directed_group_pair_coverage':statistics.mean(observed) if eligible else None,
        'oracle_start_one_hop_group_coverage':statistics.mean(cover) if eligible and cover else (0.0 if eligible else None),
        'oracle_start_all_groups_rate':statistics.mean(complete) if eligible and complete else (0.0 if eligible else None),
        'oracle_start_other_document_recall':statistics.mean(other_recall) if eligible and other_recall else (0.0 if eligible else None),
        'oracle_start_passages':len(carriers)}


def random_edges(nodes, edges, seed):
    """Degree-matched null, excluding the source document and duplicate targets."""
    rng=random.Random(seed)
    byid={n['id']:n for n in nodes}
    sources={n['id']:n.get('source') or n.get('title') or n['id'] for n in nodes}
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
            if t!=s and sources[t]!=sources[s]:chosen.add(t)
        if len(chosen)<d:
            choices=[i for i in population if i not in chosen and i!=s and sources[i]!=sources[s]]
            chosen.update(rng.sample(choices,min(d-len(chosen),len(choices))))
        result.extend((s,t) for t in sorted(chosen))
    return result


def summarize(rows, *, inference=True, resamples=10000, seed=42):
    """Both estimands use all eligible release occurrences; unmapped groups remain."""
    output={}
    for key in METRICS:
        eligible=[r for r in rows if r.get(key) is not None]
        values=[r[key] for r in eligible]
        groups=[r.get('original_query_id',r['query_id']) for r in eligible]
        if inference:
            report=cluster_interval(values,groups,repeats=resamples,seed=seed)
        else:
            bygroup={}
            for group,value in zip(groups,values,strict=True):bygroup.setdefault(group,[]).append(value)
            mean=statistics.mean(values) if values else None
            report={'mean':mean,'release_row_mean':mean,
                    'unique_question_macro_mean':statistics.mean(map(statistics.mean,bygroup.values())) if bygroup else None,
                    'rows':len(values),'clusters':len(bygroup)}
        output[key]={**report,'total_rows':len(rows),'eligible_queries':len(values),
                     'ineligible_rows':len(rows)-len(values)}
    return output


def scoped_rows(rows,scope):
    return [{'query_id':r['query_id'],'original_query_id':r['original_query_id'],**r[scope]} for r in rows]


def random_variation(values):
    values=[v for v in values if v is not None]
    return {'realizations':len(values),'mean':statistics.mean(values) if values else None,
            'sd_across_graphs':statistics.stdev(values) if len(values)>1 else None,
            'min':min(values) if values else None,'max':max(values) if values else None,
            'scope':'variation across random destination graphs; not a question-bootstrap confidence interval'}


def analyze_graph(nodes,edges,queries,mappings,*,random_repeats=20,resamples=10000,seed=42):
    sources={n['id']:n.get('source') or n.get('title') or n['id'] for n in nodes}
    def evaluate(hop_edges):
        adjacency={scope:adjacency_of(es) for scope,es in zip(SCOPES,
            (hop_edges,edges['NEXT'],hop_edges+edges['NEXT']),strict=True)}
        return [{'query_id':q['_id'],'original_query_id':q.get('original_query_id',q['_id']),
                 'question_type':q.get('question_type'),
                 **{scope:connectivity(groups,adjacency[scope],sources) for scope in SCOPES}}
                for q,groups in zip(queries,mappings,strict=True)]
    details=evaluate(edges['HOP_ANSWER'])
    for row,groups in zip(details,mappings,strict=True):
        row.update(row['hop_only']) # Existing analysis commands retain their field names.
        row['groups']={k:sorted(v) for k,v in groups.items()}
    null=[]
    for graph_seed in range(random_repeats):
        rows=evaluate(random_edges(nodes,edges['HOP_ANSWER'],graph_seed))
        null.append({'seed':graph_seed,'details':rows,
                     'summary':{scope:summarize(scoped_rows(rows,scope),inference=False) for scope in SCOPES}})
    summaries={scope:summarize(scoped_rows(details,scope),resamples=resamples,seed=seed) for scope in SCOPES}
    null_summary={}
    for scope in SCOPES:
        for i,row in enumerate(details):
            means={key:statistics.mean(v) if (v:=[n['details'][i][scope][key] for n in null
                if n['details'][i][scope][key] is not None]) else None for key in METRICS}
            row.setdefault('random_mean',{})[scope]=means
            row.setdefault('observed_minus_random',{})[scope]={key:row[scope][key]-means[key]
                if row[scope][key] is not None and means[key] is not None else None for key in METRICS}
        null_summary[scope]={}
        for key in METRICS:
            delta_rows=[{'query_id':r['query_id'],'original_query_id':r['original_query_id'],
                         key:r['observed_minus_random'][scope][key]} for r in details]
            mean_rows=[{'query_id':r['query_id'],'original_query_id':r['original_query_id'],
                        key:r['random_mean'][scope][key]} for r in details]
            null_summary[scope][key]={
                'random_graph_variation':{estimand:random_variation([n['summary'][scope][key][estimand] for n in null])
                                          for estimand in ('release_row_mean','unique_question_macro_mean')},
                'random_mean_question_inference':summarize(mean_rows,resamples=resamples,seed=seed)[key],
                'observed_minus_random_question_inference':summarize(delta_rows,resamples=resamples,seed=seed)[key]}
    metrics=[f'{scope}.{key}' for scope in SCOPES for key in METRICS]
    metrics += [f'observed_minus_random.{scope}.{key}' for scope in SCOPES for key in METRICS]
    return {'analysis_contract':CONTRACT,'comparison_metrics':metrics,'query_rows':len(queries),
        'unique_original_queries':len({r['original_query_id'] for r in details}),
        'interpretation':'co-evidence diagnostic; oracle starts, not deployed retrieval or annotated reasoning directions',
        'missing_mapping_policy':'all release rows retained; metrics require at least two gold document groups; missing groups remain unreachable',
        'other_document_policy':'one-step destinations excluding every chunk in the start document; denominator excludes the start gold document',
        'uncertainty_policy':'row-weighted and original-question macro estimands; paired original-question bootstrap conditional on graphs; random graph SD/min/max reported separately',
        'random_policy':{'realizations':random_repeats,'graph_seeds':list(range(random_repeats)),
                         'next_edges':'fixed; only HOP destinations are randomized',
                         'difference':'observed minus per-occurrence mean over fixed random graphs'},
        'summary':summaries['hop_only'],'relation_summaries':summaries,
        'question_types':{t:{scope:summarize(scoped_rows([r for r in details if r['question_type']==t],scope),
                                          resamples=resamples,seed=seed) for scope in SCOPES}
                          for t in sorted({r['question_type'] for r in details},key=str)},
        'random_degree_matched_null':null,'observed_vs_random':null_summary,'details':details}


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
    report=analyze_graph(nodes,edges,queries,mappings,random_repeats=args.random_repeats,
                         resamples=args.resamples,seed=args.bootstrap_seed)
    report.update(namespace=args.namespace,dataset=args.dataset)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({'output':str(args.output),'queries':report['query_rows'],'summary':report['summary']}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--namespace',required=True);p.add_argument('--queries',type=Path,required=True)
    p.add_argument('--dataset',choices=['multihoprag','hotpotqa'],required=True)
    p.add_argument('--sentence-store',type=Path,default=ROOT/'data/hotpotqa_corpus/sentences.sqlite3')
    p.add_argument('--resamples',type=int,default=10000);p.add_argument('--bootstrap-seed',type=int,default=42)
    p.add_argument('--random-repeats',type=int,default=20);p.add_argument('--output',type=Path,required=True)
    asyncio.run(run(p.parse_args()))


if __name__=='__main__':main()
