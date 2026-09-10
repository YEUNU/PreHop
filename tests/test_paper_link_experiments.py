import json
import sqlite3

import pytest

from models.prehop.ablation_inputs import read_input, write_input
from scripts.ablation_statistics import cluster_interval
from scripts.analyze_evidence_connections import connectivity, random_edges
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


def test_connectivity_does_not_turn_missing_or_one_way_evidence_into_complete_graph():
    groups={'A':{'a'},'B':{'b'},'C':set()}
    observed=connectivity(groups,[('a','b')])
    assert observed['any_inter_evidence_hop']==1
    assert observed['directed_group_pair_coverage']==pytest.approx(1/6)
    assert observed['oracle_start_all_groups_rate']==0
    assert observed['unmapped_groups']==1
    assert connectivity({'A':{'a'}},[])['any_inter_evidence_hop'] is None


def test_null_preserves_each_degree_and_excludes_source_document():
    nodes=[{'id':str(i),'source':str(i//2)} for i in range(10)]
    edges=[('0','2'),('0','4'),('1','8')]
    result=random_edges(nodes,edges,42)
    assert result==random_edges(nodes,edges,42)
    assert sum(s=='0' for s,t in result)==2 and sum(s=='1' for s,t in result)==1
    assert len(result)==len(set(result))==3
    assert all(nodes[int(s)]['source']!=nodes[int(t)]['source'] for s,t in result)


def test_duplicate_rows_are_resampled_as_original_question_clusters():
    report=cluster_interval([1,1,-1],['a','a','b'],repeats=1000)
    assert report['rows']==3 and report['clusters']==2
    assert report['mean']==pytest.approx(1/3)
    assert report['unique_question_macro_mean']==0


def test_timing_waits_without_stopping_other_ready_work():
    from scripts.link_experiment_campaign import ready_jobs
    jobs=[{'id':'timing','exclusive':True,'priority':0},{'id':'bench','priority':1}]
    states={j['id']:{'state':'pending'} for j in jobs}
    assert ready_jobs(jobs,states,[{'id':'adopted'}])==[jobs[1]]
    assert ready_jobs(jobs,states,[])==[jobs[0]]
    assert ready_jobs(jobs,states,[jobs[0]])==[]


def test_plan_reuses_b_prefix_for_c_and_keeps_timing_native(tmp_path):
    from scripts.plan_link_experiments import make_plan
    source=tmp_path/'stats.json';source.write_text(json.dumps({'index_policy':{'index_namespace':'source'},'run_id':'source'}))
    jobs=make_plan('test-campaign',source)['jobs'];byid={j['id']:j for j in jobs}
    assert len(byid)==len(jobs)
    for job in jobs:assert all(d in byid for d in job['after'])
    b=byid['mhr-B']['command'];c=byid['mhr-C']['command']
    assert b[b.index('--direct-inputs')+1]==c[c.index('--direct-inputs')+1]
    t=byid['mhr-T-online-1'];assert '--direct-inputs' not in t['command'] and t['exclusive']
    assert byid['mhr-T-precomputed-2']['after']==['mhr-T-online-2']
    seen=set()
    while len(seen)<len(jobs):
        ready={j['id'] for j in jobs if j['id'] not in seen and set(j['after'])<=seen}
        assert ready,'dependency cycle'
        seen|=ready


def test_natural_timing_comparison_reports_unpaired_and_different_inputs():
    from scripts.compare_timing_runs import agreement
    row = {'starts': ['a'], 'destinations': {'a': ['b']}, 'connection_ms': 2}
    online = {'starts': ['c'], 'destinations': {'c': ['d']}, 'connection_ms': 7}
    report = agreement({'q': row, 'stored-only': row}, {'q': online, 'online-only': online})
    assert report['paired_event_queries'] == 1
    assert report['missing_online'] == ['stored-only']
    assert report['missing_precomputed'] == ['online-only']
    assert report['identical_start_query_fraction'] == 0
    assert report['identical_destination_query_fraction'] == 0
    assert report['details'][0]['connection_saving_ms'] == 5


def test_paired_quality_keeps_terminal_failures_in_denominator():
    from scripts.ablation_statistics import compare
    left = {'details': [{'query_id': 'q', 'error': 'timeout', 'official_map@10': 1}]}
    right = {'details': [{'query_id': 'q', 'official_map@10': .5}]}
    report = compare(left, right, ['official_map@10'], [{'_id': 'q'}])
    assert report['metrics']['official_map@10']['mean'] == -.5
    assert report['metrics']['official_map@10']['rows'] == 1
    assert report['left_failures'] == 1
