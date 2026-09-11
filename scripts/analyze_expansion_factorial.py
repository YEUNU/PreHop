"""Analyze saved HOP/NEXT conditions with paired original-question clusters."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.benchmark_failures import metric_value
from scripts.ablation_statistics import cluster_interval, compare


def analyze(results, queries, metrics):
    pairs = [('full', 'next_only'), ('hop_only', 'none'), ('full', 'hop_only'),
             ('next_only', 'none'), ('full', 'none'), ('hop_only', 'next_only')]
    rows = {arm: {r['query_id']: r for r in result['details']} for arm, result in results.items()}
    shared = sorted(set.intersection(*(set(r) for r in rows.values())))
    groups = {q['_id']: q.get('original_query_id', q['_id']) for q in queries}
    interactions = {}
    for metric in metrics:
        values, units = [], []
        for qid in shared:
            v = {arm: metric_value(data[qid], metric) for arm, data in rows.items()}
            if all(x is not None for x in v.values()):
                values.append(v['full'] - v['next_only'] - v['hop_only'] + v['none'])
                units.append(groups.get(qid, qid))
        interactions[metric] = cluster_interval(values, units)
    return {'contract': 'paired-expansion-factorial-v1', 'paired_rows': len(shared),
            'condition_rows': {arm: len(data) for arm, data in rows.items()},
            'interaction_formula': 'full - next_only - hop_only + none',
            'interaction': interactions,
            'contrasts': {f'{a}-minus-{b}': compare(results[a], results[b], metrics, queries) for a, b in pairs}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ('full', 'next_only', 'hop_only', 'none'):
        parser.add_argument('--' + arm.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--metrics', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {arm: getattr(args, arm) for arm in ('full', 'next_only', 'hop_only', 'none')}
    result = analyze({arm: json.loads(path.read_text()) for arm, path in paths.items()},
                     json.loads(args.queries.read_text()), args.metrics)
    result['sources'] = {arm: str(path) for arm, path in paths.items()}
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
