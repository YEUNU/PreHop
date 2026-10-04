"""Versioned model configuration compatibility, independent of project Git state.

Bump a method contract when behavior changes outside its materialized settings.
Git/source hashes describe execution provenance; they are not compatibility keys.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

COMPATIBILITY_VERSION = 'paper-config-v1'
EVIDENCE_VERSION = 'paper-evidence-v3'
METHOD_CONTRACT_VERSIONS = {'prehop': 'paper-method-v6', 'naive': 'paper-method-v3'}


def runtime_compatibility(value: dict) -> dict:
    """Preserve the existing explicit runtime location and dependency contract."""
    from copy import deepcopy
    return deepcopy(value)


def prompt_configuration(strategy: str) -> dict[str, Any]:
    from utils.prompts import evidence_ranking, indexing, prehop_answer, shared
    result = {}
    if strategy == 'prehop':
        result['answer'] = prehop_answer.build_answer_prompt('{context}', '{query}')
        result['answer_messages'] = prehop_answer.build_answer_messages('{context}', '{query}')
        result['answer_version'] = prehop_answer.SYNTHESIS_PROMPT_VERSION
    elif strategy == 'naive':
        result['answer'] = shared.build_answer_prompt('{context}', '{query}')
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    result['index'] = {key: value for key, value in vars(indexing).items()
                       if key.isupper() and isinstance(value, str)}
    result['ranking'] = evidence_ranking.build_evidence_ranking_prompt('{query}', [('C000', '{title}', '{metadata}', '{text}')], 1)
    return result


# Historical method bundles included unused indexing prompts. Alias only the
# exact retained prompt bundles; editing any active prompt still changes identity.
_HISTORICAL_PROMPT_IDENTITIES = {
    'fd4aa40f6915a5f5a99b999f78b5f2b4e2dbb60b5d5ebcd4bf952cd730caabda':
        'b9c0f31b92e0a71f18d8516a79a9f3493fe8feba2879a5dd7e02a288a0ebe5c5',
    'a1ced7308751cb25fd6a03ae5a5847de65abbea98a0d9907d332453714f705e2':
        'f5e094c8fe4a80cb11bde84a63feb0d972410113d422c6a503cc93d32ee4f0fa',
    '2ea79c00a5e70a70b5db8811aa607ec4f76252443d491b653f2a8210a603ea03':
        '58adc4825e977e8eb1d2bf78e8b9e609c465ab3548582a15503eafff9e5b46ac',
}


def _prompt_identity(configuration):
    from core.admission import identity_sha256
    digest = identity_sha256(configuration)
    return _HISTORICAL_PROMPT_IDENTITIES.get(digest, digest)


def index_method_identity(strategy, environment: Mapping[str, str] | None = None):
    """Query-only changes do not change the construction identity."""
    from core.admission import identity_sha256
    from core.generation_profiles import generation_profiles
    identity = method_identity(strategy, environment)
    identity["prompt_configuration_sha256"] = _prompt_identity({"index":prompt_configuration(strategy)["index"]})
    identity["generation_profiles_sha256"] = identity_sha256({"question_index":generation_profiles(strategy, environment)["question_index"]})
    return identity


def target_configuration(
    strategy: str, dataset: str, environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Describe a target without temporarily configuring the running process."""
    from core.generation_profiles import generation_profiles
    from core.paper_policy import (
        canonical_operational_policy,
        canonical_query_policy,
        canonical_semantic_index_policy,
        resolved_target_environment,
    )
    resolved = resolved_target_environment(strategy, dataset, 'compatibility-resolution', environment)
    operational = canonical_operational_policy(strategy, resolved)
    return {'version': COMPATIBILITY_VERSION, 'evidence_version': EVIDENCE_VERSION,
            'method_contract': METHOD_CONTRACT_VERSIONS[strategy], 'strategy': strategy, 'dataset': dataset,
            'index': canonical_semantic_index_policy(strategy, resolved),
            'query': canonical_query_policy(strategy, resolved), 'prompts': prompt_configuration(strategy),
            'generation_profiles': generation_profiles(strategy, resolved),
            'operational': runtime_compatibility(operational)}


def context_configuration() -> dict:
    from core.strategy_registry import PRIMARY_STRATEGIES
    from scripts.cold_canary_fixture import fixture_identity
    from scripts.paper_gate_ledger import STAGES
    return {'protocol': {'stages': list(STAGES), 'cold_fixture': fixture_identity()}, 'version': COMPATIBILITY_VERSION, 'evidence_version': EVIDENCE_VERSION,
            'targets': {f'{dataset}/{strategy}': target_configuration(strategy, dataset)
                        for strategy in PRIMARY_STRATEGIES for dataset in ('multihoprag', 'hotpotqa')}}


def method_identity(strategy: str, environment: Mapping[str, str] | None = None) -> dict[str, str]:
    from core.admission import identity_sha256
    from core.generation_profiles import generation_profiles
    return {'method_contract': METHOD_CONTRACT_VERSIONS[strategy],
            'prompt_configuration_sha256': _prompt_identity(prompt_configuration(strategy)),
            'generation_profiles_sha256': identity_sha256(generation_profiles(strategy, environment)),
            # Prehop and Naive never had a pinned native runtime requirement; the
            # digest of the empty requirement keeps saved identities comparable.
            'runtime_requirement_sha256': identity_sha256({})}
