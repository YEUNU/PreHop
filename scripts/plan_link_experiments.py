"""Write the reproducible A/B/C, timing and co-evidence experiment task graph."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def make_plan(campaign,mhr_stats,*,adopt=()):
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
    primary('prehop','benchmark',[hp_index])
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
        def bench(arm,profile,ns,source,after,*,direct=None,timing=None,exclusive=False,short=short,tag=tag,corpus=corpus,queries=queries,reference=reference):
            run=f'{campaign}-{short}-{arm}'
            cmd=[python,'scripts/prehop_ablation.py','--mode','benchmark','--profile',profile,'--namespace',ns,
                '--run-id',run,'--corpus-tag',tag,'--dataset',corpus,'--queries',queries,'--index-stats',source,'--execute']
            if profile!='body_body':cmd+=['--reuse-existing-index']
            else:cmd+=['--reference',reference]
            if direct:cmd+=['--direct-inputs',direct]
            if timing:cmd+=['--connection-timing',timing,'--timing-store',base/f'{short}-timing-store.json']
            key=add(f'{short}-{arm}',cmd,after,0,exclusive,{'RAG_BENCHMARK_CONCURRENCY':'1'} if exclusive else {})
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
            '--store',store,'--output',base/f'{short}-timing-build.json','--execute'],list(arms.values()),4)
        replay=add(f'{short}-timing-replay',[python,'scripts/prehop_connection_timing.py','--mode','replay','--index-stats',stats,
            '--store',store,'--activations',inputs['full'][1]/'activations.json','--queries',queries,'--repetitions','5','--warmups','1',
            '--output',base/f'{short}-timing-replay.json','--execute'],[build,inputs['full'][0]],0,True)
        # Natural end-to-end timing is supplemental. Reverse block order on repeat 2.
        previous=replay;e2e={}
        for rep,order in [(1,['precomputed','online']),(2,['online','precomputed'])]:
            for arm in order:
                previous,result=bench(f'T-{arm}-{rep}','question_full',namespace,stats,[previous],timing=arm,exclusive=True)
                e2e[(rep,arm)]=(previous,result)
            add(f'{short}-T-comparison-{rep}',[python,'scripts/compare_timing_runs.py','--online',e2e[(rep,'online')][1],
                '--precomputed',e2e[(rep,'precomputed')][1],'--queries',queries,
                '--output',base/f'{short}-T-comparison-{rep}.json'],[e2e[(rep,a)][0] for a in order],2)
    for strategy in PRIMARY_STRATEGIES:
        if strategy=='prehop':continue
        index=primary(strategy,'index');primary(strategy,'benchmark',[index])
    for row in adopt:
        existing=next((j for j in jobs if j['id']==row['id']),None)
        if existing:existing.update(row)
        else:jobs.append(row)
    return {'contract':'prehop-link-experiments-v2','campaign':campaign,'max_active':2,
        'hotpotqa_protocol':'hotpotqa-hipporag-v1-1000','measurement_policy':'timing jobs run without other campaign jobs; queue quota remains 120',
        'primary_endpoints':{'timing':'paired connection-stage saving','multihoprag_B_C':'official_map@10','hotpotqa_B_C':'hotpot_sp_f1'},
        'jobs':jobs}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--multihoprag-index-stats',type=Path,required=True);p.add_argument('--adopt',type=Path)
    a=p.parse_args();plan=make_plan(a.campaign,a.multihoprag_index_stats,adopt=json.loads(a.adopt.read_text()) if a.adopt else ())
    path=ROOT/'data/results'/a.campaign/'plan.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plan,indent=2));print(json.dumps({'plan':str(path),'jobs':len(plan['jobs'])}))


if __name__=='__main__':main()
