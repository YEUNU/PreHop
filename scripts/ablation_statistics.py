"""Paired descriptive intervals with original-question cluster resampling."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.benchmark_failures import metric_value


def cluster_interval(values,groups,*,seed=42,repeats=10000):
    clusters=defaultdict(list)
    for value,group in zip(values,groups,strict=True):
        clusters[group].append(float(value))
    if not clusters:
        return {'mean':None,'ci95':None,'release_row_mean':None,'release_row_ci95':None,
                'unique_question_macro_mean':None,'unique_question_macro_ci95':None,'rows':0,'clusters':0}
    clusters=dict(sorted(clusters.items()))
    sums=np.array([sum(v) for v in clusters.values()]);counts=np.array([len(v) for v in clusters.values()])
    rng=np.random.default_rng(seed);means=[];macro_means=[]
    for start in range(0,repeats,128):
        draws=rng.integers(0,len(sums),size=(min(128,repeats-start),len(sums)))
        means.extend((sums[draws].sum(axis=1)/counts[draws].sum(axis=1)).tolist())
        macro_means.extend((sums[draws]/counts[draws]).mean(axis=1).tolist())
    return {'mean':float(sums.sum()/counts.sum()),'ci95':np.quantile(means,[.025,.975]).tolist(),
        'release_row_mean':float(sums.sum()/counts.sum()),'release_row_ci95':np.quantile(means,[.025,.975]).tolist(),
        'rows':len(values),'clusters':len(clusters),'unique_question_macro_mean':float(np.mean(sums/counts)),
        'unique_question_macro_ci95':np.quantile(macro_means,[.025,.975]).tolist(),
        'resamples':repeats,'seed':seed,'inference_unit':'original question; all release occurrences retained'}


def compare(left,right,metrics,queries):
    a={r['query_id']:r for r in left['details']};b={r['query_id']:r for r in right['details']}
    groups={q['_id']:q.get('original_query_id',q['_id']) for q in queries}
    shared=sorted(a.keys() & b.keys());output={}
    connectivity = all(r.get('analysis_contract') == 'co-evidence-connectivity-v3' for r in (left,right))
    if connectivity:
        # Pending commands from older plans acquire the full post-hoc metric set.
        metrics=list(dict.fromkeys([*metrics,*left['comparison_metrics'],*right['comparison_metrics']]))
    def value(row,key):
        if not connectivity:return metric_value(row,key)
        current=row
        for part in key.split('.'):
            if not isinstance(current,dict):return None
            current=current.get(part)
        return float(current) if isinstance(current,(int,float)) and np.isfinite(current) else None
    for metric in metrics:
        values=[];units=[]
        for qid in shared:
            x,y=value(a[qid],metric),value(b[qid],metric)
            if x is not None and y is not None:
                values.append(x-y);units.append(groups.get(qid,qid))
        output[metric]=cluster_interval(values,units)
    return {'difference':'left minus right','left_rows':len(a),'right_rows':len(b),'paired_rows':len(shared),
        'unpaired_left':sorted(a.keys()-b.keys()),'unpaired_right':sorted(b.keys()-a.keys()),
        'left_failures':sum(bool(r.get('error')) for r in a.values()),'right_failures':sum(bool(r.get('error')) for r in b.values()),
        'analysis_contract':'paired-connectivity-v3' if connectivity else 'paired-quality-v2',
        'pairing':'occurrence differences first; resample the same original-question clusters for both conditions',
        'interval_scope':'question-cluster uncertainty conditional on saved graphs; random-graph variation is separate; pointwise descriptive',
        'metrics':output}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True)
    p.add_argument('--queries',type=Path,required=True);p.add_argument('--metrics',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=compare(json.loads(a.left.read_text()),json.loads(a.right.read_text()),a.metrics,json.loads(a.queries.read_text()))
    a.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
