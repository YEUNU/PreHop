"""Applied generation settings shared by adapter requests and paper identity."""
from __future__ import annotations


def structured_retry_profile() -> dict:
    from core.config import RAGConfig
    return {'profile': 'prehop-controlled-format-retry-v1',
            'max_total_attempts': RAGConfig.LLM_MAX_RETRIES,
            'budget_scope': 'transport-and-format-shared',
            'sdk_automatic_retries': 0,
            'eligible': ['raw_json_syntax', 'duplicate_property', 'nonfinite_constant', 'registered_schema'],
            'request_identity': 'unchanged', 'response_repair': False}


def request_settings(consumer: str) -> dict:
    from core.config import RAGConfig
    from core.strategy_registry import get_strategy
    if consumer in {'ms_completion', 'ms_extract', 'ms_report'}:
        ms = dict(get_strategy("ms_graphrag").paper_index_policy)
        key = {'ms_report': 'report_max_tokens', 'ms_extract': 'extract_max_tokens', 'ms_completion': 'query_max_tokens'}[consumer]
        cap = ms[key]
        # Pinned GraphRAG ModelConfig.call_args defaults to {}: omit the limit
        # instead of inventing a client cap (None is provenance, not a wire value).
        return {'max_tokens': cap} if cap is not None else {}
    settings = {
        'question_index': {'temperature': 0.0, 'max_tokens': RAGConfig.MAX_OUTPUT_TOKENS},
        'ranking': {'temperature': 0.0, 'max_tokens': 1024},
        'answer': {'temperature': 0.0, 'max_tokens': RAGConfig.SYNTHESIS_MAX_OUTPUT_TOKENS},
        'linear_native_qa': {'temperature': 0, 'max_tokens': 2000},
        'gfm_construction': {'temperature': 0.0},
    }
    return dict(settings[consumer])


def generation_profiles(strategy: str) -> dict:
    from core.strategy_registry import get_strategy
    spec = get_strategy(strategy)
    profiles = {}
    if strategy in {'prehop', 'naive'}:
        profiles = {key: request_settings(key) for key in ('question_index', 'ranking', 'answer')}
        if strategy == 'prehop':
            profiles['structured_format_retry'] = structured_retry_profile()
    elif strategy == 'hoprag':
        profiles = {'native_defaults': {'owner': 'pinned_upstream', 'revision': spec.revision, 'temperature': 0.1, 'max_tokens': 4096, 'frequency_penalty': 0.0, 'presence_penalty': 0.0, 'seed': None, 'response_parser': 'adapter-json-recovery-v2'}}
    elif strategy == 'ms_graphrag':
        profiles = {'default_completion_and_native_query': request_settings('ms_completion'),
                    'graph_extraction_and_gleaning': request_settings('ms_extract'),
                    'community_report': request_settings('ms_report'),
                    'native_search': dict(spec.paper_index_policy)}
    elif strategy == 'linear_rag':
        profiles['native_qa'] = request_settings('linear_native_qa')
    elif strategy == 'gfm_rag':
        profiles = {'construction': request_settings('gfm_construction'),
                    'answer': {'temperature': 0.0, 'max_tokens': None, 'owner': 'pinned_qa_inference'}}
    if spec.external:
        profiles['native_defaults'] = {'owner': 'pinned_upstream', 'revision': spec.revision,
                                      'registered_overrides': dict(spec.paper_index_policy)}
    return profiles
