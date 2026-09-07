"""Versioned scoring of terminal query failures, distinct from integrity failures."""
POLICY = 'terminal-query-failure-zero-v1'
QUALITY_METRICS = frozenset('answer_em answer_f1 answer_precision answer_recall official_answer_em official_answer_f1 official_qa_accuracy null_refusal doc_match official_mrr@10 official_map@10 official_hits@4 official_hits@10 evidence_fact_recall@4 evidence_fact_recall@10 evidence_doc_recall evidence_doc_precision evidence_doc_f1 paragraph_support_precision paragraph_support_recall paragraph_support_f1 primary_answer_score'.split())


class BenchmarkIntegrityError(RuntimeError):
    """The target cannot produce trustworthy, source-mapped benchmark rows."""


def metric_value(row, key):
    if row.get('error') and key in QUALITY_METRICS:
        return 0.0
    value = row.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return None
