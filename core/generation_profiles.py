"""Applied generation settings shared by adapter requests and paper identity."""
from __future__ import annotations


def request_settings(consumer: str) -> dict:
    import os

    from core.config import RAGConfig
    from core.strategy_registry import get_strategy
    if consumer in {'ms_completion', 'ms_report'}:
        ms = dict(get_strategy("ms_graphrag").paper_index_policy)
        cap = int(ms['extract_max_tokens']) if consumer == 'ms_completion' else int(os.environ.get('RAG_MS_REPORT_MAX_TOKENS', str(ms['report_max_tokens'])))
        return {'temperature': 0.0, 'max_tokens': cap}
    settings = {
        'question_index': {'temperature': 0.0, 'max_tokens': RAGConfig.MAX_OUTPUT_TOKENS},
        'rewrite': {'temperature': 0.0, 'max_tokens': 512},
        'refine': {'temperature': 0.0, 'max_tokens': 512},
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
        profiles = {key: request_settings(key) for key in ('question_index', 'rewrite', 'refine', 'ranking', 'answer')}
    elif strategy == 'ms_graphrag':
        profiles = {'default_completion_and_native_query': request_settings('ms_completion'),
                    'community_report': request_settings('ms_report'),
                    'native_search': dict(spec.paper_index_policy)}
    elif strategy == 'linear_rag':
        profiles['native_qa'] = request_settings('linear_native_qa')
    elif strategy == 'gfm_rag':
        profiles = {'construction': request_settings('gfm_construction'), 'answer': request_settings('answer')}
    if spec.external:
        profiles['native_defaults'] = {'owner': 'pinned_upstream', 'revision': spec.revision,
                                      'registered_overrides': dict(spec.paper_index_policy)}
    if strategy == 'youtu_graphrag':
        profiles['construction'] = {'temperature': 0.3, 'max_tokens': None, 'seed': None,
                                    'response_format': 'youtu-json-schema-v1'}
    return profiles
