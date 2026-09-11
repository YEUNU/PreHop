"""Read original full-run traces to recover actual HOP starts, without inference."""
import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.estimate_connection_total import reference_settings


def recorded_starts(candidates,ablation,excluded=()):
    policy=ablation.get('hop_seed_policy','qplus')
    excluded=set(excluded)
    return sorted({str(n['id']) for n in candidates if n['id'] not in excluded and
        not n.get('role_body_owner_only') and
        (policy=='all' or (n.get('dependency_seed') and n.get('matched_qplus_ids')))})


def prepare(result):
    settings=reference_settings(result);ablation=result.get('ablation',{})
    issues=[]
    if result.get('latency_scope')=='frozen_prefix_downstream_only' or ablation.get('direct_inputs'):
        issues.append('reference_is_not_measured_end_to_end')
    if ablation.get('question_schema')!='legacy' or ablation.get('qplus_hop_activation')!='owner' or ablation.get('hop_edge_filter')!='none':
        issues.append('reference_connection_rule_not_supported_by_owner_resolver')
    bypath=defaultdict(set)
    for row in result['details']:
        if not row.get('error') and row.get('prehop_trace'):
            bypath[row['prehop_trace']['events_path']].add(row['query_id'])
    direct={};excluded={};destinations={};session_namespaces=set()
    for path,qids in bypath.items():
        path=Path(path)
        for line in path.open():
            e=json.loads(line);qid=e.get('identity',{}).get('query_id');name=e['event']
            if name=='session_start':
                payload=json.loads(gzip.decompress((path.parent/e['payload']).read_bytes()))
                if payload.get('namespace'):session_namespaces.add(payload['namespace'])
                continue
            if qid not in qids:continue
            if name not in ('RetrieveMixin._retrieve_with_candidate_pool.result','TraversalMixin.graph_search.start',
                            'SimilarityScoringMixin._score_and_select.start'):continue
            payload=json.loads(gzip.decompress((path.parent/e['payload']).read_bytes()))
            if name=='RetrieveMixin._retrieve_with_candidate_pool.result':
                # Retain identities/activation flags, not another copy of vectors or text.
                direct[qid]=[{k:n.get(k) for k in ('id','role_body_owner_only','dependency_seed','matched_qplus_ids')}
                             for n in payload[1]]
            elif name=='TraversalMixin.graph_search.start':
                excluded[qid]=payload.get('excluded_chunk_ids') or []
                if payload.get('depth')!=1:issues.append('reference_depth_is_not_one')
            else:
                pairs=defaultdict(set)
                for node in payload['candidates']:
                    for route in node.get('retrieval_paths',[]):
                        if route.get('kind')=='hop':pairs[route['source_chunk_id']].add(node['id'])
                destinations[qid]={k:sorted(v) for k,v in pairs.items()}
    starts={qid:recorded_starts(nodes,ablation,excluded.get(qid,[])) for qid,nodes in direct.items()}
    return {'contract':'recorded-reference-starts-v1','reference_settings':settings,
        'reference_issues':sorted(set(issues)),'namespaces':sorted(session_namespaces),
        'reference_queries':len(result['details']),'mapped_queries':len(starts),
        'missing_queries':[r['query_id'] for r in result['details'] if r['query_id'] not in starts],
        'start_policy':ablation.get('hop_seed_policy','qplus'),
        'start_policy_origin':'recorded ablation setting' if 'hop_seed_policy' in ablation else 'primary Q+-owner method',
        'activations':starts,'excluded':excluded,'historical_destinations':destinations,
        'reference_original_ids':{r['query_id']:r.get('original_query_id',r['query_id']) for r in result['details']}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=prepare(json.loads(a.reference_result.read_text()));result['reference_result']=str(a.reference_result.resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({'output':str(a.output),'mapped_queries':result['mapped_queries'],'reference_issues':result['reference_issues']}))


if __name__=='__main__':main()
