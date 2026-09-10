"""Reconstruct primary-run link utility from immutable candidate traces.

Run from the repository root using the project environment (orjson required).
Only the derived analysis artifact is written.
"""
import gzip
import hashlib
import json
import statistics

import orjson

json.loads = orjson.loads
from pathlib import Path

p=Path('data/results/prehop-original-query-20260910-multihoprag-prehop/prehop/multihoprag/seed_42/prehop_multihoprag.json');d=json.loads(p.read_text());rows={r['query_id']:r for r in d['details']};e=Path(d['details'][0]['prehop_trace']['events_path']);out=[]
norm=lambda s:s.replace(' ','').replace('\n','')
for line in e.open():
 event=json.loads(line)
 if event['event']!='SimilarityScoringMixin._score_and_select.start':continue
 q=event['identity']['query_id'];r=rows[q];raw=gzip.decompress((e.parent/event['payload']).read_bytes());assert hashlib.sha256(raw).hexdigest()==event['payload_sha256'];payload=json.loads(raw);cs=payload['candidates'];assert len({n['id'] for n in cs})==len(cs)
 kinds={n['id']:{x['kind'] for x in n['retrieval_paths']} for n in cs}; D={i for i,k in kinds.items() if 'direct' in k};H={i for i,k in kinds.items() if 'hop' in k};N={i for i,k in kinds.items() if 'next' in k};S={n['chunk_id'] for n in r['retrieved_sources']};assert S<=set(kinds)
 facts=set(map(norm,r['expected_sources']['facts'])); matched={n['id']:{f for f in facts if f in norm(n['text'])} for n in cs}
 def F(ids, matched=matched):return set().union(*(matched[i] for i in ids))
 if facts:
  out.append({'query_id':q,'hop_relevance':sum(bool(matched[i]) for i in H)/len(H) if H else None,'added':len(F(H-D)-F(D))/len(facts),'retained':len(F((H-D)&S)-F(D))/len(facts),'retained_next_overlap':len(F((H-D)&S&N)-F(D))/len(facts),'hop_count':len(H),'direct_count':len(D)})
 if len(out) and len(out)%500==0:print('eligible processed',len(out),flush=True)
assert len(out)==2255 and len({r['query_id'] for r in out})==2255
agg={k:{'n':len(v:=[r[k] for r in out if r[k] is not None]),'mean':statistics.mean(v)} for k in ['hop_relevance','added','retained','retained_next_overlap','hop_count','direct_count']}
Path('artifacts/paper_revision_20260909/current_link_utility.json').write_text(json.dumps({'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'trace':str(e),'trace_sha256':hashlib.sha256(e.read_bytes()).hexdigest(),'configuration':'primary original-query run, not representation A','aggregates':agg,'details':out},indent=2)+'\n');print(agg)
