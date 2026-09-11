"""Compare natural timing runs and expose changes in their actual inputs/outputs."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.ablation_statistics import compare
from scripts.analyze_ablation_links import read_events


def timing_events(result):
    paths=sorted({r['prehop_trace']['events_path'] for r in result['details'] if r.get('prehop_trace')})
    return read_events(paths,'connection_timing')


def agreement(stored,online):
    ids=sorted(stored.keys() & online.keys())
    rows=[]
    for qid in ids:
        a,b=stored[qid],online[qid]
        rows.append({'query_id':qid,'same_starts':set(a['starts'])==set(b['starts']),
            'same_destinations':a['destinations']==b['destinations'],
            'connection_saving_ms':b['connection_ms']-a['connection_ms']})
    return {'paired_event_queries':len(rows),'missing_precomputed':sorted(online.keys()-stored.keys()),
        'missing_online':sorted(stored.keys()-online.keys()),
        'identical_start_query_fraction':sum(r['same_starts'] for r in rows)/len(rows) if rows else None,
        'identical_destination_query_fraction':sum(r['same_destinations'] for r in rows)/len(rows) if rows else None,
        'details':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--precomputed',type=Path,required=True);p.add_argument('--online',type=Path,required=True)
    p.add_argument('--queries',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();stored=json.loads(a.precomputed.read_text());online=json.loads(a.online.read_text())
    metrics=['latency','retrieve_ms','graph_expand_ms','official_map@10','all_facts@10','official_qa_accuracy','hotpot_sp_f1','hotpot_joint_f1','hotpot_f1']
    result=compare(online,stored,metrics,json.loads(a.queries.read_text()))
    result['measurement_scope']='natural_end_to_end'
    result['includes_direct_retrieval']=True
    result['includes_answer_generation']=True
    result['agreement']=agreement(timing_events(stored),timing_events(online))
    result['interpretation']='natural end-to-end supplement; input/output differences and unseeded generation prevent a pure placement interpretation'
    a.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
