"""Versioned model configuration compatibility, independent of project Git state.

Bump a method contract when behavior changes outside its materialized settings.
Git/source hashes describe execution provenance; they are not compatibility keys.
"""
from __future__ import annotations

import os
from typing import Any

COMPATIBILITY_VERSION = 'paper-config-v1'
EVIDENCE_VERSION = 'paper-evidence-v3'
METHOD_CONTRACT_VERSIONS = {name: 'paper-method-v1' for name in
    ('prehop', 'naive', 'ms_graphrag', 'lightrag', 'gfm_rag', 'linear_rag')}
METHOD_CONTRACT_VERSIONS['hoprag'] = 'paper-method-v1'
METHOD_CONTRACT_VERSIONS.update(prehop='paper-method-v4', naive='paper-method-v3')


def runtime_compatibility(value: dict) -> dict:
    """Preserve the existing explicit runtime location and dependency contract."""
    from copy import deepcopy
    return deepcopy(value)


def prompt_configuration(strategy: str) -> dict[str, Any]:
    from utils.prompts import evidence_ranking, indexing, shared
    result = {}
    if strategy in {'prehop', 'naive', 'gfm_rag'}:
        result['answer'] = shared.build_answer_prompt('{context}', '{query}')
    if strategy in {'prehop', 'naive'}:
        result['index'] = {key: value for key, value in vars(indexing).items()
                           if key.isupper() and isinstance(value, str)}
        result['ranking'] = evidence_ranking.build_evidence_ranking_prompt('{query}', [('C000', '{title}', '{metadata}', '{text}')], 1)
    return result


def index_method_identity(strategy):
    """Query-only changes do not change the construction identity."""
    identity = method_identity(strategy)
    if strategy in {"prehop", "naive"}:
        from core.admission import identity_sha256
        from core.generation_profiles import generation_profiles
        identity["prompt_configuration_sha256"] = identity_sha256({"index":prompt_configuration(strategy)["index"]})
        identity["generation_profiles_sha256"] = identity_sha256({"question_index":generation_profiles(strategy)["question_index"]})
    return identity


def preserve_legacy_core_index_identity(strategy, observed, expected):
    """Recognize the old combined index/query digest only for unchanged builds.

    The old digests were reproduced from the repository's pre-removal prompts
    and generation settings. Source records retain their original hashes.
    Unknown historical values or changed construction settings are rejected.
    """
    old_prompt = "dbefec38526fc3d7699a62e926af2c7141ea99a2c48ae53cfb78b67076ab6ef1"
    old_generation = {"prehop":"18145d0cb8623183ee57d10f10ebe11e510505779bb555e0e9c75344e365ca55",
                      "naive":"e0f5128007c709b4017691f9cdec7da18f9a285510d9b1eb4645fcf658e727bb"}
    if strategy not in old_generation:
        return
    if (observed.get("prompt_configuration_sha256"),observed.get("generation_profiles_sha256")) != (old_prompt,old_generation[strategy]):
        return
    if (expected.get("prompt_configuration_sha256"),expected.get("generation_profiles_sha256")) != (
        "58adc4825e977e8eb1d2bf78e8b9e609c465ab3548582a15503eafff9e5b46ac",
        "96c69065c7ee75ec703deefdc3efeb20da0dec769afc56b4735eec98b82f8b91"):
        return
    for key in ("prompt_configuration_sha256","generation_profiles_sha256"):
        expected[key] = observed[key]


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
                        for strategy in PRIMARY_STRATEGIES for dataset in ('multihoprag', 'hotpotqa')}}


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
