"""Write primary component, matched timing, and supplemental experiment tasks."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def make_plan(campaign,mhr_stats,*,multihoprag_reference,adopt=(),updated_reference=None,updated_after=()):
    from core.index_namespace import index_namespace
    from core.strategy_registry import PRIMARY_STRATEGIES
    base=ROOT/'data/results'/campaign;python=str(Path(sys.executable).absolute())
    jobs=[];mhr=json.loads(Path(mhr_stats).read_text())
    def add(key,command,after=(),priority=5,exclusive=False,environment=None):
        item={'id':key,'command':[str(c) for c in command],'after':list(after),'priority':priority,'exclusive':exclusive,
              'environment':{'RAG_BENCHMARK_SEEDS':'42','RAG_BENCHMARK_CONCURRENCY':'8',**(environment or {})}}
        jobs.append(item);return key
    def primary(strategy,phase,after=()):
        return add(f'hp-{strategy}-{phase}',[python,'scripts/link_experiment_campaign.py','primary',campaign,phase,strategy,'hotpotqa'],after,3 if phase=='benchmark' else 8)
    hp_index=primary('prehop','index')
    hp_benchmark=primary('prehop','benchmark',[hp_index])
    hp_run=f'{campaign}-index-hotpotqa-prehop'
    old=os.environ.get('RAG_INDEX_NAMESPACE');os.environ['RAG_INDEX_NAMESPACE']='hotpotqa_'+hp_run
    hp_ns=index_namespace('hotpotqa')
    if old is None:os.environ.pop('RAG_INDEX_NAMESPACE')
    else:os.environ['RAG_INDEX_NAMESPACE']=old
    hp_stats=ROOT/f'data/index_stats/prehop_hotpotqa_{hp_run}.json'
    for tag,short,stats,namespace,dependencies in [
        ('multihoprag','mhr',Path(mhr_stats),mhr['index_policy']['index_namespace'],[]),
        ('hotpotqa','hp',hp_stats,hp_ns,[hp_index])]:
        corpus=ROOT/f'data/{tag}_corpus';queries=ROOT/f'data/{tag}_queries.json'
        inputs={};arms={};arm_results={}
        for profile,suffix in [('question_body','body'),('question_full','full')]:
            directory=base/f'{short}-direct-{suffix}';key=f'{short}-direct-{suffix}'
            add(key,[python,'scripts/prepare_ablation_inputs.py','--index-stats',stats,'--queries',queries,'--profile',profile,'--output',directory],dependencies,1)
            inputs[suffix]=(key,directory)
        c_namespace=f'ablation_{campaign.replace("-","_")}_{short}_body'
        reference=base/f'{short}-body-reference.json';c_run=f'{campaign}-{short}-body-index'
        c_stats=ROOT/f'data/index_stats/prehop_{tag}_{c_run}.json'
        clone=add(f'{short}-body-clone',[python,'scripts/prehop_ablation.py','--mode','index','--profile','body_body',
            '--clone-body-from',stats,'--namespace',c_namespace,'--run-id',c_run,'--corpus-tag',tag,'--dataset',corpus,
            '--queries',queries,'--reference',reference,'--execute'],dependencies,2)
        def bench(arm,profile,ns,source,after,*,direct=None,short=short,tag=tag,corpus=corpus,queries=queries,reference=reference):
            run=f'{campaign}-{short}-{arm}'
            cmd=[python,'scripts/prehop_ablation.py','--mode','benchmark','--profile',profile,'--namespace',ns,
                '--run-id',run,'--corpus-tag',tag,'--dataset',corpus,'--queries',queries,'--index-stats',source,'--execute']
            if profile!='body_body':cmd+=['--reuse-existing-index']
            else:cmd+=['--reference',reference]
            if direct:cmd+=['--direct-inputs',direct]
            key=add(f'{short}-{arm}',cmd,after,0)
            result=ROOT/f'data/results/ablations/{run}/{profile}/prehop/{tag}/seed_42/prehop_{tag}.json'
            return key,result
        for arm,profile,suffix,ns,source,extra in [('A','question_full','full',namespace,stats,[]),
            ('B','question_body','body',namespace,stats,[]),('C','body_body','body',c_namespace,c_stats,[clone])]:
            arms[arm],arm_results[arm]=bench(arm,profile,ns,source,[inputs[suffix][0],*extra],direct=inputs[suffix][1])
            add(f'{short}-{arm}-utility',[python,'scripts/analyze_ablation_links.py','--result',arm_results[arm],
                '--output',base/f'{short}-{arm}-utility.json'],[arms[arm]],2)
        metrics=['official_map@10','all_facts@10','official_qa_accuracy'] if tag=='multihoprag' else ['hotpot_sp_f1','hotpot_joint_f1','hotpot_f1','hotpot_em','hotpot_sp_em','hotpot_joint_em']
        for left,right in [('B','C'),('A','B'),('A','C')]:
            add(f'{short}-{left}-{right}-comparison',[python,'scripts/ablation_statistics.py','--left',arm_results[left],
                '--right',arm_results[right],'--queries',queries,'--metrics',*metrics,'--output',base/f'{short}-{left}-{right}-comparison.json'],[arms[left],arms[right]],2)
        for arm,ns,after in [('B',namespace,dependencies),('C',c_namespace,[clone])]:
            add(f'{short}-{arm}-connectivity',[python,'scripts/analyze_evidence_connections.py','--namespace',ns,
                '--dataset',tag,'--queries',queries,'--output',base/f'{short}-{arm}-connectivity.json'],after,4)
        add(f'{short}-connectivity-comparison',[python,'scripts/ablation_statistics.py','--left',base/f'{short}-B-connectivity.json',
            '--right',base/f'{short}-C-connectivity.json','--queries',queries,'--metrics','any_inter_evidence_hop',
            'directed_group_pair_coverage','oracle_start_one_hop_group_coverage','oracle_start_all_groups_rate',
            '--output',base/f'{short}-connectivity-comparison.json'],[f'{short}-B-connectivity',f'{short}-C-connectivity'],4)
        store=base/f'{short}-timing-store.json'
        build=add(f'{short}-timing-build',[python,'scripts/prehop_connection_timing.py','--mode','build','--index-stats',stats,
            '--store',store,'--output',base/f'{short}-timing-build.json','--execute'],dependencies,1)
        add(f'{short}-timing-replay',[python,'scripts/prehop_connection_timing.py','--mode','replay','--index-stats',stats,
            '--store',store,'--activations',inputs['full'][1]/'activations.json','--queries',queries,'--repetitions','1','--warmups','0',
            '--output',base/f'{short}-timing-replay.json','--execute'],[build,inputs['full'][0]],0,True)
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
        reference_inputs=base/f'{short}-reference-starts.json'
        reference_job=add(f'{short}-reference-starts',[python,'scripts/prepare_reference_timing.py',
            '--reference-result',reference_result,'--output',reference_inputs],
            [] if short=='mhr' else [hp_benchmark],4)
        build_job=next(j for j in jobs if j['id']==build)
        build_job['command'] += ['--reference-inputs', str(reference_inputs)]
        build_job['after'] = [reference_job]
        reference_replay=add(f'{short}-reference-timing',[python,'scripts/verify_primary_timing.py','--index-stats',stats,'--store',store,'--reference-inputs',reference_inputs,
            '--queries',queries,'--output',base/f'{short}-reference-timing.json'],[build,reference_job],0,True)
        add(f'{short}-estimated-total',[python,'scripts/estimate_connection_total.py',
            '--reference-result',reference_result,'--timing',base/f'{short}-reference-timing.json',
            '--queries',queries,'--output',base/f'{short}-estimated-total.json'],[reference_replay],2)
    for strategy in PRIMARY_STRATEGIES:
        if strategy=='prehop':continue
        index=primary(strategy,'index');primary(strategy,'benchmark',[index])
    for job in jobs:
        suffix=job['id'].split('-',1)[-1]
        if suffix in {'A','B','C'} or suffix.startswith(('A-','B-','C-','direct-','body-clone','connectivity-','timing-replay')):
            job['experiment_role']='supplemental_representation_comparison'
            job['priority']=12
        elif 'primary-' in suffix or suffix in {'reference-starts','reference-timing','estimated-total','timing-build'}:
            job['experiment_role']='primary_anchored_ablation'
        if job['id']=='hp-prehop-index':job['priority']=1
    if updated_reference:
        from copy import deepcopy
        suffixes={'reference-starts','reference-timing','estimated-total','timing-build'}
        originals=[j for j in jobs if j['id'].startswith('mhr-') and
                   (j['id'][4:].startswith('primary-') or j['id'][4:] in suffixes)]
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
    return {'contract':'prehop-link-experiments-v5','campaign':campaign,'max_active':2,'max_hoprag_active':1,
        'hotpotqa_protocol':'hotpotqa-hipporag-v1-1000','measurement_policy':'fixed-start timing only; no natural end-to-end reruns; separate reference-based estimates; queue quota 120',
        'multihoprag_reference_result':str(multihoprag_reference),
        'primary_endpoints':{'timing':'paired connection-stage saving','multihoprag_without_hop':'official_map@10','hotpotqa_without_hop':'hotpot_sp_f1'},
        'jobs':jobs}


def reconcile_pending(existing, states, latest, completed=None):
    """Replace pending definitions while preserving active and completed evidence."""
    from copy import deepcopy
    plan=deepcopy(existing)
    previous={j['id']:j for j in existing['jobs']}
    jobs=[]
    updated=deepcopy(states)
    for job in latest['jobs']:
        key=job['id']
        state=updated.setdefault(key, {'state':'pending'})
        jobs.append(deepcopy(previous[key] if state['state'] in {'running','completed'} and key in previous else job))
    keys={j['id'] for j in jobs}
    jobs.extend(deepcopy(j) for j in existing['jobs'] if j['id'] not in keys)
    for key,evidence in (completed or {}).items():
        if updated.get(key,{}).get('state') != 'running':
            updated[key]={'state':'completed','reused_evidence':str(evidence)}
    plan.update({k:deepcopy(v) for k,v in latest.items() if k!='jobs'})
    plan['jobs']=jobs
    plan.pop('execution_filter',None)
    return plan,updated


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--multihoprag-index-stats',type=Path,required=True);p.add_argument('--adopt',type=Path);p.add_argument('--multihoprag-reference',type=Path,required=True)
    p.add_argument('--updated-reference',type=Path);p.add_argument('--updated-after',action='append',default=[])
    a=p.parse_args();plan=make_plan(a.campaign,a.multihoprag_index_stats,multihoprag_reference=a.multihoprag_reference,adopt=json.loads(a.adopt.read_text()) if a.adopt else (),updated_reference=a.updated_reference,updated_after=a.updated_after)
    path=ROOT/'data/results'/a.campaign/'plan.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plan,indent=2));print(json.dumps({'plan':str(path),'jobs':len(plan['jobs'])}))


if __name__=='__main__':main()
