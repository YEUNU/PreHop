"""Versioned model configuration compatibility, independent of project Git state.

Bump a method contract when behavior changes outside its materialized settings.
Git/source hashes describe execution provenance; they are not compatibility keys.
"""
from __future__ import annotations

import os
from typing import Any

COMPATIBILITY_VERSION = 'paper-config-v1'
EVIDENCE_VERSION = 'paper-evidence-v2'
METHOD_CONTRACT_VERSIONS = {name: 'paper-method-v1' for name in
    ('prehop', 'naive', 'ms_graphrag', 'lightrag', 'hipporag2', 'gfm_rag', 'linear_rag', 'youtu_graphrag')}


def runtime_compatibility(value: dict) -> dict:
    """Preserve the existing explicit runtime location and dependency contract."""
    from copy import deepcopy
    return deepcopy(value)


def prompt_configuration(strategy: str) -> dict[str, Any]:
    from utils.prompts import indexing, query_rewrite, shared
    result = {}
    if strategy in {'prehop', 'naive', 'gfm_rag'}:
        result['answer'] = shared.build_answer_prompt('{context}', '{query}')
    if strategy in {'prehop', 'naive'}:
        result['index'] = {key: value for key, value in vars(indexing).items()
                           if key.isupper() and isinstance(value, str)}
        result['rewrite'] = query_rewrite.build_role_aligned_query_prompt('{query}', 3)
        result['refine'] = query_rewrite.build_evidence_conditioned_query_prompt('{query}', '{evidence}', ['{attempted}'], 3)
        result['ranking'] = query_rewrite.build_evidence_ranking_prompt('{query}', [('C000', '{title}', '{metadata}', '{text}')], 1)
    return result


def target_configuration(strategy: str, dataset: str) -> dict[str, Any]:
    """Resolve the same canonical policies checked against actual produced stats."""
    from core.generation_profiles import generation_profiles
    from core.paper_policy import (
        canonical_operational_policy,
        canonical_query_policy,
        canonical_semantic_index_policy,
        configure_target_environment,
        validate_paper_semantic_environment,
    )
    prior = os.environ.copy()
    try:
        configure_target_environment(strategy, dataset, 'compatibility-resolution')
        validate_paper_semantic_environment(strategy, dataset)
        operational = canonical_operational_policy(strategy)
        return {'version': COMPATIBILITY_VERSION, 'evidence_version': EVIDENCE_VERSION,
                'method_contract': METHOD_CONTRACT_VERSIONS[strategy], 'strategy': strategy, 'dataset': dataset,
                'index': canonical_semantic_index_policy(strategy, dataset),
                'query': canonical_query_policy(strategy), 'prompts': prompt_configuration(strategy),
                'generation_profiles': generation_profiles(strategy),
                'operational': runtime_compatibility(operational)}
    finally:
        os.environ.clear()
        os.environ.update(prior)


def context_configuration() -> dict:
    from core.strategy_registry import PRIMARY_STRATEGIES
    from scripts.cold_canary_fixture import fixture_identity
    from scripts.paper_gate_ledger import STAGES
    return {'protocol': {'stages': list(STAGES), 'cold_fixture': fixture_identity()}, 'version': COMPATIBILITY_VERSION, 'evidence_version': EVIDENCE_VERSION,
            'targets': {f'{dataset}/{strategy}': target_configuration(strategy, dataset)
                        for strategy in PRIMARY_STRATEGIES for dataset in ('multihoprag', 'musique')}}


def without_realized_runtime(value: dict) -> dict:
    """Permit setup to realize dependencies, but never change method settings."""
    from copy import deepcopy
    result = deepcopy(value)
    result.pop('runtime', None)
    targets = result.get('targets', {})
    if isinstance(targets, dict):
        for target in targets.values():
            operational = target.get('operational', {})
            for key in ('main_runtime', 'runtime_freeze', 'constraints'):
                operational.pop(key, None)
    return result


def method_identity(strategy: str) -> dict[str, str]:
    if strategy not in METHOD_CONTRACT_VERSIONS:
        return {}  # Reserve/development methods do not acquire a primary paper contract.
    from core.admission import identity_sha256
    from core.generation_profiles import generation_profiles
    from core.runtime_requirements import load_runtime_requirements
    return {'method_contract': METHOD_CONTRACT_VERSIONS[strategy],
            'prompt_configuration_sha256': identity_sha256(prompt_configuration(strategy)),
            'generation_profiles_sha256': identity_sha256(generation_profiles(strategy)),
            'runtime_requirement_sha256': identity_sha256(load_runtime_requirements().get(strategy, {}))}
