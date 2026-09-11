import copy
import gzip
import json

import pytest

from scripts.estimate_connection_total import estimate, reference_settings
from scripts.prepare_reference_timing import prepare, recorded_starts


def fixture():
    reference={'dataset':'MultiHop-RAG','models':{'default':'model'},'ablation':{},
               'execution_profile':{'benchmark_concurrency':8},
               'details':[{'query_id':'a','latency':10},{'query_id':'b','latency':20},{'query_id':'c','latency':30}]}
    queries=[{'_id':'a','original_query_id':'duplicate'},{'_id':'b','original_query_id':'duplicate'},
             {'_id':'c','original_query_id':'other'}]
    timing={'measurement_scope':'fixed_start_connection_replay','reference_settings':reference_settings(reference),
            'reference_starts':{k:['start'] for k in ['a','b','c']},'details':[]}
    for qid,delta in [('a',2000),('b',-1000),('c',3000)]:
        for rep in range(2):
            timing['details'].append({'query_id':qid,'repetition':rep,'arms':{
                'precomputed':{'starts':['start'],'destinations':{'start':['target']},'connection_ms':4000},
                'online':{'starts':['start'],'destinations':{'start':['target']},'connection_ms':4000+delta}}})
    return reference,timing,queries


def test_per_query_delta_units_sign_and_both_estimands_are_correct():
    reference,timing,queries=fixture();before=copy.deepcopy(reference)
    out=estimate(reference,timing,queries)
    assert reference==before
    assert out['measurement_scope']=='estimated_end_to_end'
    assert out['full_population_estimate_available']
    assert [r['estimated_online_total_seconds'] for r in out['details']]==[12,19,33]
    summary=out['full_population_summary']['estimated_online_total_seconds']
    assert summary['release_row_mean']==pytest.approx(64/3)
    assert summary['unique_question_macro_mean']==pytest.approx((15.5+33)/2)
    assert summary['p95_seconds']==pytest.approx(31.6)
    assert out['full_population_summary']['measured_reference_total_seconds']['mean']==20
    assert 'online answer quality' in out['not_measured']


def test_missing_timing_never_becomes_a_full_population_estimate():
    reference,timing,queries=fixture();timing['details']=[r for r in timing['details'] if r['query_id']!='b']
    out=estimate(reference,timing,queries)
    assert out['estimated_queries']==2
    assert out['full_population_summary'] is None
    assert 'connection_measurement_missing' in out['details'][1]['unavailable_reasons']


def test_partial_reference_keeps_expected_population_visible():
    reference,timing,queries=fixture();reference['details']=reference['details'][:2]
    out=estimate(reference,timing,queries)
    assert not out['full_population_estimate_available']
    assert out['missing_reference_queries']==['c']


@pytest.mark.parametrize('change', ['frozen','settings','starts'])
def test_incompatible_inputs_are_observations_not_fabricated_estimates(change):
    reference,timing,queries=fixture()
    if change=='frozen':reference['latency_scope']='frozen_prefix_downstream_only'
    elif change=='settings':timing['reference_settings']=dict(timing['reference_settings'],models={'default':'other'})
    else:timing['reference_starts']={k:['different'] for k in ['a','b','c']}
    out=estimate(reference,timing,queries)
    assert out['estimated_queries']==0 and out['full_population_summary'] is None
    assert all(r['unavailable_reasons'] for r in out['details'])


def test_destination_disagreement_is_exposed_without_predicting_quality():
    reference,timing,queries=fixture()
    timing['details'][0]['arms']['online']['destinations']={'start':['other']}
    out=estimate(reference,timing,queries)
    assert out['details'][0]['stored_online_destinations_identical'] is False
    assert out['details'][0]['estimated_online_total_seconds']==12
    assert 'online answer quality' in out['not_measured']


def test_recorded_primary_owner_starts_do_not_use_all_ablation_candidates():
    candidates=[{'id':'owner','dependency_seed':True,'matched_qplus_ids':['q']},
                {'id':'body','dependency_seed':False,'matched_qplus_ids':[]},
                {'id':'only','role_body_owner_only':True,'dependency_seed':True,'matched_qplus_ids':['q']}]
    assert recorded_starts(candidates,{})==['owner']
    assert recorded_starts(candidates,{'hop_seed_policy':'all'})==['body','owner']
    assert recorded_starts(candidates,{},['owner'])==[]


def test_preparation_recovers_starts_and_destinations_from_original_traces(tmp_path):
    events=tmp_path/'events.jsonl';lines=[]
    def emit(name,payload,qid=None):
        path=tmp_path/f'{len(lines)}.json.gz';path.write_bytes(gzip.compress(json.dumps(payload).encode()))
        lines.append(json.dumps({'event':name,'identity':{'query_id':qid} if qid else {},'payload':path.name}))
    emit('session_start',{'namespace':'existing'})
    emit('TraversalMixin.graph_search.start',{'depth':1},'q')
    emit('RetrieveMixin._retrieve_with_candidate_pool.result',[[],[{'id':'s','dependency_seed':True,'matched_qplus_ids':['question']}]],'q')
    emit('SimilarityScoringMixin._score_and_select.start',{'candidates':[{'id':'t','retrieval_paths':[{'kind':'hop','source_chunk_id':'s'}]}]},'q')
    events.write_text('\n'.join(lines))
    reference={'ablation':{'question_schema':'legacy','qplus_hop_activation':'owner','hop_edge_filter':'none'},
               'details':[{'query_id':'q','prehop_trace':{'events_path':str(events)}}]}
    report=prepare(reference)
    assert report['activations']=={'q':['s']} and report['historical_destinations']=={'q':{'s':['t']}}
    assert report['reference_issues']==[] and report['namespaces']==['existing']
    assert report['mapped_queries']==1
