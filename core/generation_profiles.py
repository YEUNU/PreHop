"""Applied generation settings shared by adapter requests and paper identity."""
from __future__ import annotations

from collections.abc import Mapping


def structured_retry_profile(environment: Mapping[str, str] | None = None) -> dict:
    from core.inference_transport import InferenceTransport
    return {'profile': 'prehop-native-json-retry-v3',
            'max_total_attempts': InferenceTransport.resolve('core', environment).retry_attempts,
            'budget_scope': 'transport-and-format-shared',
            'sdk_automatic_retries': 0,
            'eligible': ['json_syntax', 'invalid_raw_type'],
            'request_identity': 'unchanged', 'response_repair': False}


def request_settings(consumer: str) -> dict:
    from core.config import RAGConfig
    settings = {
        'question_index': {'temperature': 0.0, 'max_tokens': RAGConfig.MAX_OUTPUT_TOKENS},
        'ranking': {'temperature': 0.0, 'max_tokens': 1024},
        'answer': {'temperature': 0.0, 'max_tokens': RAGConfig.SYNTHESIS_MAX_OUTPUT_TOKENS},
        'prehop_answer': {'temperature': 0.0, 'max_tokens': RAGConfig.PREHOP_SYNTHESIS_MAX_OUTPUT_TOKENS},
    }
    return dict(settings[consumer])


def generation_profiles(strategy: str, environment: Mapping[str, str] | None = None) -> dict:
    from core.strategy_registry import get_strategy
    get_strategy(strategy)
    profiles = {key: request_settings(key) for key in ('question_index', 'ranking', 'answer')}
    if strategy == 'prehop':
        profiles['answer'] = request_settings('prehop_answer')
        profiles['structured_format_retry'] = structured_retry_profile(environment)
    return profiles
