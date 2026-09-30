import json
import sqlite3

import pytest

from models.prehop.ablation_inputs import read_input, write_input
from scripts.ablation_statistics import cluster_interval
from scripts.datasets.prepare_hotpotqa_hipporag import prepare


def test_release_duplicates_preserve_occurrences_and_original_sentences(tmp_path):
    raw=tmp_path/'raw';raw.mkdir()
    row={'_id':'original','question':'q','answer':'a','type':'bridge','supporting_facts':[['One',0],['Two',0]]}
    (raw/'hotpotqa.json').write_text(json.dumps([row,row]))
    (raw/'hotpotqa_corpus.json').write_text(json.dumps({'One':['First.',' Second.'],'Two':['Target.']}))
    manifest=prepare(raw,tmp_path/'corpus',tmp_path/'queries.json')
    queries=json.loads((tmp_path/'queries.json').read_text())
    assert len(queries)==2 and queries[0]['_id']!=queries[1]['_id']
    assert queries[0]['original_query_id']==queries[1]['original_query_id']=='original'
    assert manifest['query_count']==2 and manifest['unique_original_queries']==1
    assert manifest['official_fullwiki'] is False
    with sqlite3.connect(tmp_path/'corpus/sentences.sqlite3') as db:
        assert json.loads(db.execute("SELECT sentences FROM paragraphs WHERE title='One'").fetchone()[0])==['First.',' Second.']


def test_frozen_inputs_are_independent_mutable_copies(tmp_path):
    write_input(tmp_path,'q',{'query_embedding':[1,2],'base_candidates':[{'id':'a','score':1}]})
    first=read_input(tmp_path,'q');first['base_candidates'][0]['score']=99
    assert read_input(tmp_path,'q')['base_candidates'][0]['score']==1


def test_duplicate_rows_are_resampled_as_original_question_clusters():
    report=cluster_interval([1,1,-1],['a','a','b'],repeats=1000)
    assert report['rows']==3 and report['clusters']==2
    assert report['mean']==pytest.approx(1/3)
    assert report['unique_question_macro_mean']==0


def test_timing_waits_without_stopping_other_ready_work():
    from scripts.link_experiment_campaign import ready_jobs
    jobs=[{'id':'timing','exclusive':True,'priority':0},{'id':'bench','priority':1}]
    states={j['id']:{'state':'pending'} for j in jobs}
    assert ready_jobs(jobs,states,[{'id':'adopted'}],max_active=2)==[jobs[1]]
    assert ready_jobs(jobs,states,[])==[jobs[0]]
    assert ready_jobs(jobs,states,[jobs[0]])==[]


def test_paired_quality_keeps_terminal_failures_in_denominator():
    from scripts.ablation_statistics import compare
    left = {'details': [{'query_id': 'q', 'error': 'timeout', 'official_map@10': 1}]}
    right = {'details': [{'query_id': 'q', 'official_map@10': .5}]}
    report = compare(left, right, ['official_map@10'], [{'_id': 'q'}])
    assert report['metrics']['official_map@10']['mean'] == -.5
    assert report['metrics']['official_map@10']['rows'] == 1
    assert report['left_failures'] == 1


def test_only_one_hoprag_job_can_use_the_two_campaign_slots():
    from scripts.link_experiment_campaign import ready_jobs
    jobs=[{'id':'hp-hoprag-index','priority':8},{'id':'mhr-hoprag-index','priority':8},{'id':'ordinary','priority':0}]
    states={j['id']:{'state':'pending'} for j in jobs}
    picked=ready_jobs(jobs,states,[],max_active=2)
    assert len(picked)==2 and sum('hoprag' in j['id'] for j in picked)==1
    assert ready_jobs(jobs,states,[{'id':'adopted-hoprag-mhr'}],max_active=2)==[jobs[2]]


def test_campaign_resume_does_not_repeat_completed_adopted_work(tmp_path):
    from scripts.link_experiment_campaign import run
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps({'jobs':[{'id':'done','adopt':{'pid':-1},'command':['must-not-run']}]}))
    (tmp_path/'status.json').write_text(json.dumps({'tasks':{'done':{'state':'completed','exit_code':0}}}))
    run(plan,resume=True)
    state=json.loads((tmp_path/'status.json').read_text())
    assert state['state']=='completed' and state['tasks']['done']['state']=='completed'
    assert not (tmp_path/'done.log').exists()


def test_hoprag_only_mode_runs_one_task_and_leaves_other_models_pending():
    from scripts.link_experiment_campaign import ready_jobs
    jobs=[{'id':'hp-hoprag-index'},{'id':'mhr-hoprag-index'},{'id':'hp-lightrag-index','priority':0}]
    states={j['id']:{'state':'pending'} for j in jobs}
    assert ready_jobs(jobs,states,[],max_active=1,strategy_filter='hoprag')==[jobs[0]]
    assert ready_jobs(jobs,states,[jobs[0]],max_active=1,strategy_filter='hoprag')==[]


def test_primary_hotpot_jobs_do_not_depend_on_supplemental_arms(tmp_path):
    from scripts.plan_link_experiments import make_plan
    source=tmp_path/'stats.json'
    source.write_text(json.dumps({'index_policy':{'index_namespace':'source'},'run_id':'source'}))
    plan=make_plan('test',source,multihoprag_reference=tmp_path/'reference.json')
    jobs={j['id']:j for j in plan['jobs']}
    assert plan['max_active']==1 and plan['max_hoprag_active']==1
    assert jobs['hp-primary-hop-inputs']['after']==['hp-prehop-benchmark']
    assert jobs['hp-primary-without-hop']['after']==['hp-primary-hop-inputs']
    for key in ['hp-A','hp-B']:
        assert jobs[key]['experiment_role']=='supplemental_representation_comparison'
        assert jobs[key]['priority']>jobs['hp-primary-without-hop']['priority']


def test_updated_policy_suite_has_separate_evidence_and_reference_dependency(tmp_path):
    from scripts.plan_link_experiments import make_plan
    source=tmp_path/'stats.json'
    source.write_text(json.dumps({'index_policy':{'index_namespace':'source'},'run_id':'source'}))
    old=tmp_path/'historical.json';new=tmp_path/'updated.json'
    plan=make_plan('suite',source,multihoprag_reference=old,updated_reference=new,
                   updated_after=['new-reference'],adopt=[{'id':'new-reference','command':[], 'after':[]}])
    jobs={j['id']:j for j in plan['jobs']}
    for short in ('hp','mhr-updated'):
        for label,expansion in [('hop-only','hop_only'),('direct-only','none')]:
            job=jobs[f'{short}-primary-{label}-inputs']
            assert job['command'][job['command'].index('--expansion')+1]==expansion
    for key,job in jobs.items():
        if key.startswith('mhr-updated-'):
            assert str(old) not in job['command']
            assert all(not d.startswith('mhr-') or d.startswith('mhr-updated-') for d in job['after'])
            assert job['after']
    assert jobs['mhr-updated-primary-hop-inputs']['after']==['new-reference']
    assert str(new) in jobs['mhr-updated-primary-hop-inputs']['command']
    assert str(old) in jobs['mhr-primary-hop-inputs']['command']
    assert not any('natural' in j['id'] for j in plan['jobs'])


def test_factorial_uses_paired_occurrences_and_keeps_cluster_units():
    from scripts.analyze_expansion_factorial import analyze
    queries=[{'_id':'a','original_query_id':'same'}, {'_id':'b','original_query_id':'same'}, {'_id':'c'}]
    results={arm:{'details':[{'query_id':q['_id'],'official_map@10':v} for q in queries]}
             for arm,v in [('full',.8),('next_only',.5),('hop_only',.4),('none',.2)]}
    report=analyze(results,queries,['official_map@10'])
    metric=report['interaction']['official_map@10']
    assert metric['mean']==pytest.approx(.1)
    assert metric['rows']==3 and metric['clusters']==2
    assert len(report['contrasts'])==6
    assert report['contrasts']['full-minus-next_only']['metrics']['official_map@10']['mean']==pytest.approx(.3)
