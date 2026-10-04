"""Run-local labelling of explicit Prehop ablations. No defaults are changed.

``scripts/run_primary_hop_ablation.py`` replaces ``ablation_identity`` with its
component-ablation label before the benchmark module is imported.
"""

import os


def ablation_identity(config=None):
    if config is None:
        from core.config import RAGConfig

        config = RAGConfig
    if not config.PREHOP_ABLATION_PROFILE:
        return {}
    identity = {
        "representation_ablation": config.PREHOP_ABLATION_PROFILE,
        "method_contract": "prehop-representation-ablation-v1",
        "hop_seed_policy": config.HOP_SEED_POLICY,
        "hop_link_variant": config.HOP_LINK_VARIANT,
        "comparison_scope": "ablation_only",
    }
    if os.environ.get("RAG_ABLATION_DIRECT_INPUTS"):
        identity.update(direct_inputs=os.environ["RAG_ABLATION_DIRECT_INPUTS"],
                        latency_scope="frozen_prefix_downstream_only")
    return identity
