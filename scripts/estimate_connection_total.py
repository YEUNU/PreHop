"""Estimate full-query latency from an existing run and matched connection deltas.

This script never invokes retrieval, generation, or a service. Existing result
files are read-only; estimates and comparability observations have their own file.
"""
import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.ablation_statistics import cluster_interval


def reference_settings(result):
    return {key:result.get(key) for key in ('dataset','corpus_manifest_fingerprint',
            'models','ablation','execution_profile','index_manifest_stats_path')}


def estimate(reference,timing,queries):
    """Add each occurrence's online-minus-stored delta, never a grand mean delta."""
    issues=[]
    if reference.get('latency_scope') not in (None,'end_to_end'):
        issues.append('reference_is_not_measured_end_to_end')
    if reference.get('ablation',{}).get('latency_scope')=='frozen_prefix_downstream_only':
        issues.append('frozen_prefix_is_not_a_full_query_baseline')
    if reference.get('ablation',{}).get('connection_timing_arm') not in (None,'','precomputed'):
        issues.append('reference_is_not_a_stored_connection_run')
    if timing.get('reference_settings')!=reference_settings(reference):
        issues.append('reference_settings_differ_or_are_missing')
    if timing.get('measurement_scope')!='fixed_start_connection_replay':
        issues.append('connection_measurement_scope_differs')
    issues.extend(timing.get('reference_issues',[]))
    groups={q['_id']:q.get('original_query_id',q['_id']) for q in queries}
    measurements=defaultdict(list)
    for row in timing.get('details',[]):measurements[row['query_id']].append(row)
    details=[]
    expected_starts=timing.get('reference_starts',{})
    for row in reference['details']:
        qid=row['query_id'];observed=row.get('latency');reps=measurements.get(qid,[])
        reasons=list(issues)
        if row.get('error'):reasons.append('reference_query_failed')
        if not isinstance(observed,(int,float)) or not math.isfinite(observed) or observed<0:
            reasons.append('reference_latency_unavailable')
        if not reps:reasons.append('connection_measurement_missing')
        if qid not in expected_starts:reasons.append('recorded_reference_starts_missing')
        valid=[];agreements=[]
        for rep in reps:
            arms=rep['arms'];a=arms['precomputed'];b=arms['online']
            if set(a['starts'])!=set(b['starts']) or set(a['starts'])!=set(expected_starts.get(qid,[])):
                reasons.append('starts_do_not_match_reference')
            delta=(b['connection_ms']-a['connection_ms'])/1000
            if not math.isfinite(delta):reasons.append('nonfinite_connection_delta')
            valid.append(delta);agreements.append(a['destinations']==b['destinations'])
        delta=statistics.mean(valid) if valid else None
        predicted=observed+delta if not reasons else None
        if predicted is not None and predicted<0:
            reasons.append('negative_additive_prediction');predicted=None
        details.append({'query_id':qid,'original_query_id':groups.get(qid,row.get('original_query_id',qid)),
            'measured_reference_total_seconds':observed,'connection_delta_seconds':delta,
            'estimated_online_total_seconds':predicted,'connection_repetitions':len(reps),
            'stored_online_destinations_identical':all(agreements) if agreements else None,
            'historical_destination_agreement':timing.get('historical_destination_agreement',{}).get(qid),
            'unavailable_reasons':sorted(set(reasons))})
    paired=[r for r in details if r['estimated_online_total_seconds'] is not None]
    def summary(key):
        values=[r[key] for r in paired];identities=[r['original_query_id'] for r in paired]
        return {**cluster_interval(values,identities),
                'p95_seconds':float(np.quantile(values,.95)) if values else None}
    complete=set(groups)=={r['query_id'] for r in paired} and len(paired)==len(details) and bool(details)
    cohort={key:summary(key) for key in ('measured_reference_total_seconds',
                                       'estimated_online_total_seconds','connection_delta_seconds')}
    return {'analysis_contract':'connection-adjusted-total-v1','measurement_scope':'estimated_end_to_end',
        'measured_reference_scope':'existing_stored_graph_end_to_end','measured_connection_scope':'fixed_start_connection_replay',
        'formula':'estimated_online_seconds[q] = measured_reference_seconds[q] + mean_r(online_ms[q,r] - stored_ms[q,r])/1000',
        'assumptions':['unchanged retrieval, generation, non-connection work and scheduling',
                       'connection delta transfers additively from paired replay to the recorded workload',
                       'destination differences do not induce modeled downstream cost changes'],
        'not_measured':['online end-to-end latency','online answer quality','online batch wall time or throughput'],
        'expected_queries':len(groups),'missing_reference_queries':sorted(set(groups)-{r['query_id'] for r in details}),
        'reference_queries':len(details),'estimated_queries':len(paired),
        'full_population_estimate_available':complete,
        'global_comparability_issues':sorted(set(issues)),'paired_subset_summary':cohort,
        'full_population_summary':cohort if complete else None,
        'unmatched_timing_queries':sorted(set(measurements)-{r['query_id'] for r in details}),
        'interval_scope':'question-cluster uncertainty conditional on reference outputs and timing repetitions; does not cover additive-model or server-load error',
        'details':details}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-result',type=Path,required=True);p.add_argument('--timing',type=Path,required=True)
    p.add_argument('--queries',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=estimate(json.loads(a.reference_result.read_text()),json.loads(a.timing.read_text()),json.loads(a.queries.read_text()))
    result.update(reference_result=str(a.reference_result.resolve()),timing_result=str(a.timing.resolve()))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({'output':str(a.output),'estimated_queries':result['estimated_queries'],
                      'full_population_estimate_available':result['full_population_estimate_available']}))


if __name__=='__main__':main()
