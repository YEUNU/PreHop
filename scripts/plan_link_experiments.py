"""Write primary index/benchmark and HOP/NEXT expansion experiment tasks."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def make_plan(campaign,*,multihoprag_reference,adopt=(),updated_reference=None,updated_after=()):
    from core.execution_profile import resolved_execution_environment
    from core.strategy_registry import PRIMARY_STRATEGIES
    base=ROOT/'data/results'/campaign;python=str(Path(sys.executable).absolute())
    jobs=[]
    execution = resolved_execution_environment()
    job_environment = {key: execution[key] for key in
                       ('RAG_BENCHMARK_SEEDS', 'RAG_BENCHMARK_CONCURRENCY', 'RAG_EXECUTION_PROFILE') if key in execution}
    def add(key,command,after=(),priority=5,exclusive=False,environment=None):
        item={'id':key,'command':[str(c) for c in command],'after':list(after),'priority':priority,'exclusive':exclusive,
              'environment':{**job_environment,**(environment or {})}}
        jobs.append(item);return key
    def primary(strategy,phase,after=()):
        return add(f'hp-{strategy}-{phase}',[python,'scripts/link_experiment_campaign.py','primary',campaign,phase,strategy,'hotpotqa'],after,3 if phase=='benchmark' else 8)
    hp_index=primary('prehop','index')
    hp_benchmark=primary('prehop','benchmark',[hp_index])
    for tag,short in [('multihoprag','mhr'),('hotpotqa','hp')]:
        queries=ROOT/f'data/{tag}_queries.json'
        reference_result=(Path(multihoprag_reference) if short=='mhr' else
            ROOT/f'data/results/{campaign}-hotpotqa-prehop/prehop/hotpotqa/seed_42/prehop_hotpotqa.json')
        component_output=base/f'{short}-primary-without-hop'
        component_args=['--reference',reference_result,'--queries',queries,'--output',component_output]
        component_prepare=add(f'{short}-primary-hop-inputs',
            [python,'scripts/run_primary_hop_ablation.py','prepare',*component_args],
            [] if short=='mhr' else [hp_benchmark],0)
        add(f'{short}-primary-without-hop',
            [python,'scripts/run_primary_hop_ablation.py','supervise',*component_args],
            [component_prepare],0)
        for expansion, label in [('hop_only', 'hop-only'), ('none', 'direct-only')]:
            output=base/f'{short}-primary-{label}'
            args=['--reference',reference_result,'--queries',queries,'--output',output,'--expansion',expansion]
            prepared=add(f'{short}-primary-{label}-inputs',
                [python,'scripts/run_primary_hop_ablation.py','prepare',*args],
                [] if short=='mhr' else [hp_benchmark],0)
            add(f'{short}-primary-{label}',
                [python,'scripts/run_primary_hop_ablation.py','supervise',*args],[prepared],0)
        from scripts.run_primary_hop_ablation import comparison_metrics
        factorial=[python,'scripts/analyze_expansion_factorial.py','--full',reference_result]
        for option,label in [('next-only','without-hop'),('hop-only','hop-only'),('none','direct-only')]:
            factorial += ['--'+option,base/f'{short}-primary-{label}'/f'prehop/{tag}/seed_42/prehop_{tag}.json']
        add(f'{short}-primary-factorial',[*factorial,'--queries',queries,'--metrics',*comparison_metrics(tag),
            '--output',base/f'{short}-primary-factorial.json'],
            [f'{short}-primary-{label}' for label in ('without-hop','hop-only','direct-only')],1)
    for strategy in PRIMARY_STRATEGIES:
        if strategy=='prehop':continue
        index=primary(strategy,'index');primary(strategy,'benchmark',[index])
    for job in jobs:
        suffix=job['id'].split('-',1)[-1]
        if 'primary-' in suffix:
            job['experiment_role']='primary_anchored_ablation'
        if job['id']=='hp-prehop-index':job['priority']=1
    if updated_reference:
        from copy import deepcopy
        originals=[j for j in jobs if j['id'].startswith('mhr-') and
                   j['id'][4:].startswith('primary-')]
        mapping={j['id']:j['id'].replace('mhr-', 'mhr-updated-', 1) for j in originals}
        for original in originals:
            job=deepcopy(original)
            job['id']=mapping[original['id']]
            job['after']=[mapping.get(d,d) for d in original['after']] or list(updated_after)
            job['command']=[str(updated_reference) if c==str(multihoprag_reference)
                            else c.replace(str(base/'mhr-'),str(base/'mhr-updated-'))
                            for c in original['command']]
            job['reference_policy']='all-start/body-only'
            jobs.append(job)
    for row in adopt:
        existing=next((j for j in jobs if j['id']==row['id']),None)
        if existing:existing.update(row)
        else:jobs.append(row)
    return {'contract':'prehop-paper-experiments-v6','campaign':campaign,'max_active':1,
        'hotpotqa_protocol':'hotpotqa-hipporag-v1-1000','measurement_policy':'fixed-start expansion comparisons; primary Prehop and Naive runs',
        'multihoprag_reference_result':str(multihoprag_reference),
        'primary_endpoints':{'multihoprag_without_hop':'official_map@10','hotpotqa_without_hop':'hotpot_sp_f1'},
        'jobs':jobs}


def main():
    from scripts.runner_environment import _load_runner_environment
    _load_runner_environment()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--adopt',type=Path);p.add_argument('--multihoprag-reference',type=Path,required=True)
    p.add_argument('--updated-reference',type=Path);p.add_argument('--updated-after',action='append',default=[])
    a=p.parse_args();plan=make_plan(a.campaign,multihoprag_reference=a.multihoprag_reference,adopt=json.loads(a.adopt.read_text()) if a.adopt else (),updated_reference=a.updated_reference,updated_after=a.updated_after)
    path=ROOT/'data/results'/a.campaign/'plan.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plan,indent=2));print(json.dumps({'plan':str(path),'jobs':len(plan['jobs'])}))


if __name__=='__main__':main()
