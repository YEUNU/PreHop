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
    if not clusters:return {'mean':None,'ci95':None,'rows':0,'clusters':0}
    sums=np.array([sum(v) for v in clusters.values()]);counts=np.array([len(v) for v in clusters.values()])
    rng=np.random.default_rng(seed);means=[]
    for start in range(0,repeats,128):
        draws=rng.integers(0,len(sums),size=(min(128,repeats-start),len(sums)))
        means.extend((sums[draws].sum(axis=1)/counts[draws].sum(axis=1)).tolist())
    return {'mean':float(sums.sum()/counts.sum()),'ci95':np.quantile(means,[.025,.975]).tolist(),
        'rows':len(values),'clusters':len(clusters),'unique_question_macro_mean':float(np.mean(sums/counts)),
        'resamples':repeats,'seed':seed,'inference_unit':'original question; all release occurrences retained'}


def compare(left,right,metrics,queries):
    a={r['query_id']:r for r in left['details']};b={r['query_id']:r for r in right['details']}
    groups={q['_id']:q.get('original_query_id',q['_id']) for q in queries}
    shared=sorted(a.keys() & b.keys());output={}
    for metric in metrics:
        values=[];units=[]
        for qid in shared:
            x,y=metric_value(a[qid],metric),metric_value(b[qid],metric)
            if x is not None and y is not None and x>=0 and y>=0:
                values.append(x-y);units.append(groups.get(qid,qid))
        output[metric]=cluster_interval(values,units)
    return {'difference':'left minus right','left_rows':len(a),'right_rows':len(b),'paired_rows':len(shared),
        'unpaired_left':sorted(a.keys()-b.keys()),'unpaired_right':sorted(b.keys()-a.keys()),
        'left_failures':sum(bool(r.get('error')) for r in a.values()),'right_failures':sum(bool(r.get('error')) for r in b.values()),
        'interval_scope':'pointwise descriptive; secondary endpoints are exploratory, no omnibus significance claim',
        'metrics':output}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True)
    p.add_argument('--queries',type=Path,required=True);p.add_argument('--metrics',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=compare(json.loads(a.left.read_text()),json.loads(a.right.read_text()),a.metrics,json.loads(a.queries.read_text()))
    a.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
