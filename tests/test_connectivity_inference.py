"""Scientific denominators and the separation of graph and query uncertainty."""
import json
import subprocess
import sys

import pytest

from scripts.ablation_statistics import cluster_interval, compare
from scripts.analyze_evidence_connections import analyze_graph, connectivity, summarize


def test_other_document_recall_excludes_all_source_chunks_and_retains_missing_gold():
    groups={'A':{'a1','a2'},'B':{'b'},'missing':set()}
    sources={'a1':'docA','a2':'docA','a3':'docA','b':'docB'}
    # Only one of three oracle starts reaches another gold group; one of two
    # other required documents is recovered there. Sibling a3 adds no credit.
    report=connectivity(groups,[('a1','a2'),('a1','a3'),('a1','b')],sources)
    assert report['oracle_start_other_document_recall']==pytest.approx(1/6)
    assert report['oracle_start_one_hop_group_coverage']>report['oracle_start_other_document_recall']
    assert report['unmapped_groups']==1
    assert connectivity(groups,[('a1','a2'),('a1','a3')],sources)['oracle_start_other_document_recall']==0
    assert connectivity({'A':set(),'B':set()},[],sources)['oracle_start_other_document_recall']==0
    assert connectivity({'A':{'a1'}},[],sources)['oracle_start_other_document_recall'] is None


def test_macro_and_release_means_have_separate_cluster_intervals_and_stable_order():
    a=cluster_interval([1,1,0],['same','same','other'],repeats=400)
    b=cluster_interval([0,1,1],['other','same','same'],repeats=400)
    assert a==b
    assert a['release_row_mean']==pytest.approx(2/3)
    assert a['unique_question_macro_mean']==.5
    assert a['unique_question_macro_ci95']==[0,1]
    rows=[{'query_id':'a','original_query_id':'same','oracle_start_other_document_recall':1},
          {'query_id':'b','original_query_id':'same','oracle_start_other_document_recall':1},
          {'query_id':'c','original_query_id':'other','oracle_start_other_document_recall':0},
          {'query_id':'d','original_query_id':'empty','oracle_start_other_document_recall':None}]
    summary=summarize(rows,resamples=400)['oracle_start_other_document_recall']
    assert summary['total_rows']==4 and summary['rows']==3 and summary['clusters']==2
    assert summary['ineligible_rows']==1
    assert summary['release_row_mean']==a['release_row_mean']


def graph_fixture():
    nodes=[{'id':x,'source':x} for x in ['a','b','c','d']]
    queries=[{'_id':'q1','original_query_id':'original','question_type':'bridge'},
             {'_id':'q2','original_query_id':'original','question_type':'bridge'},
             {'_id':'q3','original_query_id':'other','question_type':'comparison'}]
    mappings=[{'A':{'a'},'B':{'b'}},{'A':{'a'},'B':{'b'}},{'C':{'c'},'D':{'d'}}]
    return nodes,queries,mappings


def test_twenty_random_graphs_and_paired_question_inference_are_distinct():
    nodes,queries,mappings=graph_fixture()
    report=analyze_graph(nodes,{'HOP_ANSWER':[('a','b'),('b','a')],'NEXT':[]},queries,mappings,resamples=100)
    assert len(report['random_degree_matched_null'])==20
    assert report['random_policy']['realizations']==20
    key='oracle_start_other_document_recall'
    stat=report['observed_vs_random']['hop_only'][key]
    assert stat['random_graph_variation']['release_row_mean']['realizations']==20
    assert stat['random_graph_variation']['release_row_mean']['sd_across_graphs']>0
    assert 'ci95' not in stat['random_graph_variation']['release_row_mean']
    assert stat['observed_minus_random_question_inference']['clusters']==2
    assert stat['observed_minus_random_question_inference']['resamples']==100
    # No independent resampling of the observed and random arms.
    for i,row in enumerate(report['details']):
        mean=sum(g['details'][i]['hop_only'][key] for g in report['random_degree_matched_null'])/20
        assert row['observed_minus_random']['hop_only'][key]==pytest.approx(row['hop_only'][key]-mean)
    assert stat['observed_minus_random_question_inference']['release_row_mean']==pytest.approx(
        sum(r['observed_minus_random']['hop_only'][key] for r in report['details'])/3)
    json.dumps(report,allow_nan=False)


def test_existing_pending_comparison_command_acquires_new_metrics_without_restart(tmp_path):
    key='oracle_start_other_document_recall'
    signed='observed_minus_random.hop_only.'+key
    metrics=['hop_only.'+key,signed]
    left={'analysis_contract':'co-evidence-connectivity-v3','comparison_metrics':metrics,
          'details':[{'query_id':'a','hop_only':{key:1},'observed_minus_random':{'hop_only':{key:-.25}}},
                     {'query_id':'b','hop_only':{key:1},'observed_minus_random':{'hop_only':{key:-.25}}},
                     {'query_id':'c','hop_only':{key:0},'observed_minus_random':{'hop_only':{key:-.5}}}]}
    right={'analysis_contract':left['analysis_contract'],'comparison_metrics':metrics,
           'details':[{'query_id':r['query_id'],'hop_only':{key:.5},'observed_minus_random':{'hop_only':{key:0}}} for r in left['details']]}
    queries=[{'_id':'a','original_query_id':'same'},{'_id':'b','original_query_id':'same'},{'_id':'c','original_query_id':'other'}]
    expected=compare(left,right,[],queries)
    assert expected['metrics'][metrics[0]]['release_row_mean']==pytest.approx(1/6)
    assert expected['metrics'][metrics[0]]['unique_question_macro_mean']==0
    assert expected['metrics'][signed]['mean']==pytest.approx(-1/3)
    for name,obj in [('left',left),('right',right),('queries',queries)]:
        (tmp_path/f'{name}.json').write_text(json.dumps(obj))
    subprocess.run([sys.executable,'scripts/ablation_statistics.py','--left',str(tmp_path/'left.json'),
                    '--right',str(tmp_path/'right.json'),'--queries',str(tmp_path/'queries.json'),
                    '--metrics','directed_group_pair_coverage','--output',str(tmp_path/'out.json')],check=True)
    actual=json.loads((tmp_path/'out.json').read_text())
    assert actual['metrics'][signed]==expected['metrics'][signed]
    assert actual['analysis_contract']=='paired-connectivity-v3'


def test_actual_query_utility_uses_clusters_and_keeps_failures_separate():
    from scripts.analyze_ablation_links import summarize_utility
    rows=[{'query_id':'a','original_query_id':'same','added_gold_coverage':1},
          {'query_id':'b','original_query_id':'same','added_gold_coverage':1},
          {'query_id':'c','original_query_id':'other','added_gold_coverage':0},
          {'query_id':'d','original_query_id':'failure','failed':True}]
    result=summarize_utility(rows)['added_gold_coverage']
    assert result['release_row_mean']==pytest.approx(2/3)
    assert result['unique_question_macro_mean']==.5
    assert result['total_rows']==4 and result['eligible_queries']==3
    assert result['clusters']==2


@pytest.mark.asyncio
async def test_analyzer_writes_complete_posthoc_artifact_without_mutating_graph(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from scripts import analyze_evidence_connections as module
    nodes=[{'id':'a','source':'docA','title':'A','text':'Fact A.'},
           {'id':'b','source':'docB','title':'B','text':'Fact B.'},
           {'id':'c','source':'docC','title':'C','text':'Other.'}]
    queries=[{'_id':'q1','original_query_id':'same','question_type':'bridge',
              'evidence_docs':['A','B'],'evidence_facts':['Fact A.','Fact B.']},
             {'_id':'q2','original_query_id':'same','question_type':'bridge',
              'evidence_docs':['A','B'],'evidence_facts':['Fact A.','Fact B.']},
             {'_id':'empty','original_query_id':'empty','question_type':'null_query',
              'evidence_docs':[],'evidence_facts':[]}]
    service=SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr('core.neo4j_service.Neo4jService',lambda:service)
    export=AsyncMock(return_value=(nodes,{'HOP_ANSWER':[('a','b')],'NEXT':[]}))
    monkeypatch.setattr(module,'export_graph',export)
    source=tmp_path/'queries.json';source.write_text(json.dumps(queries))
    output=tmp_path/'report.json'
    await module.run(SimpleNamespace(namespace='fixture',queries=source,dataset='multihoprag',
                                     sentence_store=None,random_repeats=20,resamples=100,bootstrap_seed=42,output=output))
    report=json.loads(output.read_text())
    assert report['analysis_contract']=='co-evidence-connectivity-v3'
    summary=report['summary']['oracle_start_other_document_recall']
    assert summary['mean']==.5 and summary['clusters']==1 and summary['rows']==2
    assert summary['total_rows']==3 and summary['ineligible_rows']==1
    assert len(report['random_degree_matched_null'])==20
    export.assert_awaited_once_with(service,'fixture')
    service.close.assert_awaited_once()


def test_random_baseline_uses_title_when_source_property_is_absent():
    from scripts.analyze_evidence_connections import random_edges
    nodes=[{'id':'a','title':'same'},{'id':'b','title':'same'},{'id':'c','title':'other'}]
    assert random_edges(nodes,[('a','c')],0)==[('a','c')]
