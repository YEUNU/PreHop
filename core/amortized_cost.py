"""Wall-time normalization, distinct from summed request latency and billing."""
from __future__ import annotations

import math


def normalized_cost(seconds, count, *, unit, complete, resumed=False, scope):
    valid_time = type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0
    valid_count = type(count) is int and count > 0
    eligible = bool(complete and not resumed and valid_time and valid_count)
    return {
        'version': 1, 'scope': scope, 'unit': unit, 'count': count,
        'wall_seconds': seconds, 'complete': bool(complete), 'resumed': bool(resumed),
        'continuous_run_eligible': eligible,
        'seconds_per_unit': seconds / count if eligible else None,
        'units_per_second': count / seconds if eligible else None,
    }


def indexing_cost(stats):
    return normalized_cost(stats.get('timing_seconds', {}).get('total_elapsed_seconds'),
                           stats.get('corpus_manifest_paragraph_count'), unit='source_document',
                           complete=stats.get('status') == 'complete',
                           scope='index_pipeline_wall_including_internal_queue_and_retries')


def query_cost(seconds, count, *, complete, resumed=False):
    return normalized_cost(seconds, count, unit='query', complete=complete, resumed=resumed,
                           scope='query_batch_dispatch_to_last_answer_including_internal_queue_and_retries')


def validate_cost(value, expected):
    if not isinstance(value, dict) or value != expected:
        raise ValueError('Amortized cost differs from its wall-time/count/completion contract')
