"""Recompute paired uncertainty from the completed paper result catalog.

Reads benchmark artifacts without modifying them; writes the derived analysis
next to the catalog. Requires NumPy. Run from the repository root.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

meta=json.loads(Path('artifacts/paper_revision_20260909/completed_metrics.json').read_text())
runs={}
sources={}
for name,m in meta['methods'].items():
 p=Path(m['path']); raw=p.read_bytes(); assert hashlib.sha256(raw).hexdigest()==m['sha256']
 d=json.loads(raw); rows=d['details']; assert len(rows)==2556 and not any(r.get('error') for r in rows)

 for r in rows:
  norm=lambda x:x.replace(' ','').replace('\n','')
  facts=set(map(norm,r['expected_sources']['facts'])); bodies=[norm(x.get('text','')) for x in r['retrieved_sources'][:10]]
  r['all_facts@10']=float(all(any(f in b for b in bodies) for f in facts)) if facts else 0.0
 runs[name]={r['query_id']:r for r in rows}; sources[name]={'path':str(p),'sha256':m['sha256']}
base=runs['prehop']; ids=sorted(k for k,r in base.items() if r['expected_sources']['facts']); assert len(ids)==2255
out={}
for name,rows in runs.items():
 assert set(rows)==set(base)
 for k in ids: assert rows[k]['query']==base[k]['query'] and rows[k]['expected_sources']['facts']==base[k]['expected_sources']['facts']
 if name=='prehop':continue
 delta=np.array([[base[k][metric]-rows[k][metric] for metric in ['official_map@10','all_facts@10']] for k in ids],dtype=float)
 rng=np.random.default_rng(42); samples=[]
 for _ in range(100):samples.append(delta[rng.integers(0,len(ids),size=(100,len(ids)))].mean(axis=1))
 ci=np.quantile(np.concatenate(samples),[.025,.975],axis=0)
 out[name]={metric:{'difference':float(delta[:,j].mean()),'ci95':ci[:,j].tolist()} for j,metric in enumerate(['MAP@10','AllFacts@10'])}
p=Path('artifacts/paper_revision_20260909/current_paired_bootstrap.json')
p.write_text(json.dumps({'sources':sources,'eligible_queries':2255,'resamples':10000,'seed':42,'method':'paired query percentile bootstrap; unadjusted; Prehop minus comparator; fixed outputs; both metrics on 0-1 scale','comparisons':out},indent=2)+'\n')
print(json.dumps(out,indent=2))
