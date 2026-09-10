"""Explicit, non-primary representation ablations. No defaults are changed."""

import os
from pathlib import Path

PROFILES = {
    "question_full": {"HYPO_CHANNEL_VARIANT": "full", "HOP_LINK_VARIANT": "question"},
    "question_body": {"HYPO_CHANNEL_VARIANT": "body_only", "HOP_LINK_VARIANT": "question"},
    "body_body": {"HYPO_CHANNEL_VARIANT": "body_only", "HOP_LINK_VARIANT": "body"},
}
COMMON = {
    "HOP_SEED_POLICY": "all",
    "GRAPH_HOP_DEPTH": 1,
    "GRAPH_EDGE_VARIANT": "full",
    "GRAPH_PATH_DECAY": 0.5,
    "HOP_EDGE_FILTER": "none",
    "QPLUS_HOP_ACTIVATION": "owner",
    "HOP_SEMANTIC_VARIANT": "body_only",
    "CONTINUATION_EDGES_ENABLED": False,
    "QUESTION_SCHEMA": "legacy",
    "SOURCE_SELECTION_VARIANT": "role_body_list_ranking",
    "DEFAULT_TOP_K": 12,
    "FINAL_RANK_VARIANT": "fused",
    "CANDIDATE_POOL_MULTIPLIER": 1,
    "SENTENCE_CHANNEL_ENABLED": False,
}


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
    if getattr(config, "CONNECTION_TIMING_MODE", ""):
        from core.admission import sha256_file
        identity.update(connection_timing_contract="prehop-connection-timing-v2",
                        connection_timing_arm=config.CONNECTION_TIMING_MODE,
                        connection_timing_store_sha256=sha256_file(Path(config.CONNECTION_TIMING_STORE)),
                        destination_memoization=False)
    if config.HOP_LINK_VARIANT == "body":
        from core.admission import sha256_file

        identity["body_link_reference_sha256"] = sha256_file(Path(config.BODY_LINK_REFERENCE))
    return identity
