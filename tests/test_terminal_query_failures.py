from pathlib import Path

import pytest

from cli.benchmark import _apply_judge_label, _assert_benchmark_complete, _recompute_aggregates, _update_summary_status
from core.amortized_cost import query_cost
from core.benchmark_failures import POLICY, QUALITY_METRICS
from core.campaign_outcomes import index_outcomes
from scripts.verify_submission_consistency import _validate_primary_row


def test_terminal_failure_keeps_denominator_latency_and_completion():
    good = {'answer':'A','answer_em':1.,'primary_answer_score':1., 'latency':2.,'category':'hop'}
    bad = {'answer':'ERROR','error':'timeout', 'failure_scope':'query','latency':8.,'category':'hop',
           **{key:0. for key in QUALITY_METRICS},'llm_judge_score':-1.,'hallucination':-1.}
    for row in [good,bad]:_apply_judge_label(row)
    s={'details':[good,bad],'total_queries':2,'judge_enabled':False}
    _recompute_aggregates(s);_update_summary_status(s)
    assert s['avg_answer_em']==.5 and s['eligible_answer_em_count']==2
    assert s['avg_latency']==5. and s['query_failure_rate']==.5
    assert s['correct_rate']==.5 and bad['answer_label']=='Incorrect Answer'
    assert s['status']=='completed_unadmitted' and s['failure_policy']==POLICY
    _assert_benchmark_complete(s,Path('synthetic.json'))
    assert query_cost(10,2,complete=True)['continuous_run_eligible']


def test_target_integrity_failure_still_blocks_completion():
    s={'details':[{'error':'foreign source','failure_scope':'target'}],'total_queries':1}
    _update_summary_status(s)
    assert s['status']=='failed'
    with pytest.raises(RuntimeError,match='integrity'):_assert_benchmark_complete(s,Path('synthetic.json'))


def test_failed_query_verification_requires_zero_primary_metrics():
    expected={'ground_truth':'answer','evidence_facts':['p']}
    row={'answer':'ERROR answer', 'error':'timeout','retrieved_sources':[],
         'expected_sources':{'docs':[],'facts':['p']},**{key:0. for key in QUALITY_METRICS}}
    assert _validate_primary_row(row,expected,'multihoprag')==[]
    row['official_mrr@10']=1.
    assert _validate_primary_row(row,expected,'multihoprag')


def test_failed_index_reports_unavailable_quality_and_attempt_time():
    result=index_outcomes({'targets':{'hotpotqa/gfm_rag':{'state':'index_failed','started_at':10,'finished_at':40,'index_exit_code':1}}})['targets'][0]
    assert result['quality_score'] is None and result['quality_evaluation']=='unavailable'
    assert result['attempt_wall_seconds']==30 and result['attempt_phase']=='index'


@pytest.mark.asyncio
@pytest.mark.parametrize('fatal', [False, True])
async def test_live_query_loop_isolates_failure_without_zeroing_success(monkeypatch, tmp_path, fatal):
    import json

    import cli.benchmark as bench
    import core.execution_profile as profile
    from core.benchmark_failures import BenchmarkIntegrityError
    from scripts import recovery_checkpoint
    calls=[]
    class Engine:
        async def run_workflow(self, query, history):
            calls.append(query)
            if query=='bad':
                if fatal:raise BenchmarkIntegrityError('foreign source')
                raise RuntimeError('native query failure')
            return 'correct', [], []
        def close(self):pass
    async def verify(*a, **k):return {'status':'matched'}
    async def evaluate(**k):return {'answer_em':1.,'primary_answer_score':1.,'doc_match':1.,'llm_judge_score':-1.}
    async def barrier(*a,**k):pass
    monkeypatch.setattr(profile,'require_queue',lambda *a:None)
    monkeypatch.setenv('RAG_PAPER_MODE','false')
    monkeypatch.setenv('RAG_BENCHMARK_CONCURRENCY','1')
    monkeypatch.delenv('RAG_INDEX_REUSE_LINK',raising=False)
    monkeypatch.delenv('RAG_BENCHMARK_RESUME',raising=False)
    monkeypatch.setattr(bench.RAGConfig,'JUDGE_ENABLED',False)
    monkeypatch.setattr(bench,'NaiveRAG',lambda **k:Engine())
    monkeypatch.setattr(bench,'_validate_benchmark_data',lambda data,*a:data)
    monkeypatch.setattr(bench,'_load_benchmark_corpus_manifest',lambda *a:{})
    monkeypatch.setattr(bench,'_latest_index_manifest_metadata',lambda *a:{})
    monkeypatch.setattr(bench,'_validate_corpus_index_fingerprint',lambda *a:'matched')
    monkeypatch.setattr(bench,'_verify_active_index_snapshot',verify)
    monkeypatch.setattr(bench,'evaluate_multihoprag_response',evaluate)
    monkeypatch.setattr(bench,'current_post_query_inventory',lambda *a:{})
    monkeypatch.setattr(bench,'_write_model_report_artifacts',lambda *a:None)
    monkeypatch.setattr(bench,'_write_slim_main',lambda *a:None)
    monkeypatch.setattr(recovery_checkpoint,'checkpoint_barrier',barrier)
    path=tmp_path/'queries.json'
    path.write_text(json.dumps([{'_id':str(i),'query':q,'dataset':'multihoprag'} for i,q in enumerate(['good','bad','after'])]))
    run=getattr(bench.run_benchmark,'__wrapped__',bench.run_benchmark)
    s=await run(str(path),'naive','synthetic',corpus_tag='synthetic',output_dir=tmp_path/'results')
    assert s['details'][0]['answer_em']==1.
    assert s['details'][1]['answer_em']==0. and s['query_failure_count']==1
    if fatal:
        assert calls==['good','bad'] and s['status']=='failed'
        assert not s['amortized_query_cost']['continuous_run_eligible']
    else:
        assert calls==['good','bad','after'] and s['avg_answer_em']==pytest.approx(2/3)
        assert s['status']=='completed_unadmitted'
        assert s['amortized_query_cost']['continuous_run_eligible']


@pytest.mark.asyncio
async def test_external_adapter_keeps_native_empty_answer_and_no_evidence():
    from types import SimpleNamespace

    from models.external_research.adapter import ExternalResearchAdapter
    adapter=object.__new__(ExternalResearchAdapter)
    adapter.strategy='gfm_rag'
    adapter._batcher=None
    adapter._worker=SimpleNamespace(request=lambda payload:{'documents':[],'answer':''})
    answer,sources,trace=await adapter.run_workflow('query')
    assert 'Insufficient evidence' not in answer
    assert sources==[] and trace[0]['retrieved']==0
