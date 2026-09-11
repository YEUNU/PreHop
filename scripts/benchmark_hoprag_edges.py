"""Measure native versus cached HopRAG dense blocks on existing question caches."""
import argparse
import ast
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.hoprag.exact_edges import PreparedDense


def main():
    import pickle
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    pending=[];answerable=[]
    for file in sorted((a.cache/'docs').glob('*.pkl')):
        with file.open('rb') as f:_,nodes=pickle.load(f)
        for _node,groups in nodes.values():
            answerable.extend(np.array(q[2]) for q in groups.get('answerable',[]))
            for q in groups.get('pending',[]):
                if len(pending)<256:pending.append(np.array(q[2]))
        del nodes
    pdf=pd.DataFrame({'embedding':pending});adf=pd.DataFrame({'embedding':answerable})
    tree=ast.parse(a.source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pending_dot_answerable')
    ns={'np':np,'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(a.source),'exec'),ns)  # noqa: S102 - pinned upstream numeric function
    native=ns['pending_dot_answerable'];start=time.perf_counter();reference=np.asarray(native(pdf.iloc[:8],adf)).reshape(8,len(adf));native_seconds=time.perf_counter()-start
    start=time.perf_counter();cached=PreparedDense(adf);prepare=time.perf_counter()-start
    report={'answerable':len(adf),'pending_sample':len(pdf),'dtype':str(cached.dtype),'cuda':cached.cuda,'native_8_seconds':native_seconds,'prepare_seconds':prepare,'blocks':[]}
    print(json.dumps({k:v for k,v in report.items() if k!='blocks'}),flush=True)
    baseline=None
    for block in (8,32,64,128):
        start=time.perf_counter();outputs=[]
        for i in range(0,len(pdf),block):outputs.append(cached(pdf.iloc[i:i+block],adf))
        seconds=time.perf_counter()-start;values=np.concatenate(outputs);del outputs
        if baseline is None:baseline=values
        row={'block':block,'seconds':seconds,'pending_per_second':len(pdf)/seconds,
             'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             'score_block_mib':block*len(adf)*cached.dtype.itemsize/1024**2,
             'native_first8_bitwise_equal':bool(np.array_equal(values[:8],reference)),
             'block8_bitwise_equal':bool(np.array_equal(values,baseline)),
             'max_absolute_difference':float(np.max(np.abs(values-baseline)))}
        report['blocks'].append(row);print(json.dumps(row),flush=True)
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
