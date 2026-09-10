"""Execute an explicit experiment DAG with two slots and isolated timing jobs."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def ready_jobs(jobs,states,active):
    ready=[j for j in jobs if states[j['id']]['state']=='pending' and
           all(states[d]['state']=='completed' for d in j.get('after',[]))]
    ready.sort(key=lambda j:(j.get('priority',5),j['id']))
    if any(j.get('exclusive') for j in active):return []
    # Timing waits for an idle campaign; useful non-timing work can continue.
    exclusive=next((j for j in ready if j.get('exclusive')),None)
    if exclusive and not active:return [exclusive]
    return [j for j in ready if not j.get('exclusive')][:max(0,2-len(active))]


def primary(campaign,phase,strategy,dataset):
    if phase=='index':
        subprocess.run([sys.executable,'-u','scripts/index_matrix.py','child',campaign,
            '--phase','index','--strategy',strategy,'--dataset',dataset],check=True)
    else:
        from core.index_reuse import prepare_completed
        receipt=f'data/results/{campaign}/index/{dataset}/{strategy}/completion.json'
        prepare_completed(campaign,strategy,dataset,receipt)
        subprocess.run([sys.executable,'-u','scripts/paper_stage_runner.py','_reuse_benchmark',campaign,
            '--strategy',strategy,'--dataset',dataset],check=True)
        subprocess.run([sys.executable,'scripts/record_paper_completion.py',f'{campaign}-{dataset}-{strategy}',
            dataset,strategy,'--exact-run-id'],check=True)


def run(plan_path):
    from scripts.paper_campaign import alive, atomic_json, identity
    plan=json.loads(plan_path.read_text());base=plan_path.parent
    jobs=plan['jobs'];states={j['id']:{'state':'pending'} for j in jobs};children={};active=[]
    for job in jobs:
        if job.get('adopt'):
            states[job['id']]={'state':'running','identity':job['adopt']};active.append(job)
    while True:
        for job in list(active):
            key=job['id'];status=states[key]
            if job.get('adopt'):
                if alive(job['adopt']):continue
                path=Path(job['completion'])
                evidence=json.loads(path.read_text()) if path.exists() else {}
                code=0 if evidence.get('state',evidence.get('status')) in {'completed','complete','completed_unadmitted','index_complete','admitted'} or (job.get('completion_kind')=='direct_inputs' and evidence.get('queries') and not evidence.get('failed_rows')) else 1
            else:
                proc,handle=children[key];code=proc.poll()
                if code is None:continue
                handle.close();del children[key]
            status.update(state='completed' if code==0 else 'failed',exit_code=code,finished_at=time.time())
            active.remove(job)
        for job in jobs:
            if states[job['id']]['state']=='pending' and any(states[d]['state'] in {'failed','dependency_failed'} for d in job.get('after',[])):
                states[job['id']]={'state':'dependency_failed'}
        for job in ready_jobs(jobs,states,active):
            env=os.environ.copy()
            for k in ('RAG_INDEX_REUSE_LINK','RAG_INDEX_STATS_PATH','RAG_BENCHMARK_RESUME','RAG_ABLATION_DIRECT_INPUTS','RAG_CONNECTION_TIMING_MODE','RAG_CONNECTION_TIMING_STORE','RAG_PREHOP_ABLATION_PROFILE'):
                env.pop(k,None)
            env.update(job.get('environment',{}))
            env['RAG_RUN_ID']=job['id'];env['PYTHONUNBUFFERED']='1'
            handle=(base/(job['id']+'.log')).open('x')
            proc=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            children[job['id']]=(proc,handle);active.append(job)
            states[job['id']]={'state':'running','identity':identity(proc.pid),'started_at':time.time()}
        unfinished=any(s['state'] in {'pending','running'} for s in states.values())
        atomic_json(base/'status.json',{'state':'running' if unfinished else ('completed_with_failures' if any(s['state']!='completed' for s in states.values()) else 'completed'),
            'controller':identity(os.getpid()),'max_active':2,'active':[j['id'] for j in active],
            'tasks':states,'updated_at':time.time()})
        if not unfinished:return
        time.sleep(5)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    r=sub.add_parser('run');r.add_argument('plan',type=Path)
    r=sub.add_parser('primary');r.add_argument('campaign');r.add_argument('phase',choices=['index','benchmark']);r.add_argument('strategy');r.add_argument('dataset')
    a=p.parse_args()
    if a.mode=='run':run(a.plan)
    else:primary(a.campaign,a.phase,a.strategy,a.dataset)


if __name__=='__main__':main()
