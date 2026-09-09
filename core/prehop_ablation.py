"""Explicit, non-primary representation ablations. No defaults are changed."""

import os
import re
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
    "QUERY_REWRITE_VARIANT": "none",
    "QUERY_REFINEMENT_MAX_ROUNDS": 0,
    "CONTINUATION_EDGES_ENABLED": False,
    "QUESTION_SCHEMA": "legacy",
    "SOURCE_SELECTION_VARIANT": "role_body_list_ranking",
    "DEFAULT_TOP_K": 12,
    "FINAL_RANK_VARIANT": "fused",
    "CANDIDATE_POOL_MULTIPLIER": 1,
    "SENTENCE_CHANNEL_ENABLED": False,
}


def validate_profile(config):
    profile = config.PREHOP_ABLATION_PROFILE
    if not profile:
        if config.HOP_LINK_VARIANT != "question" or config.HOP_SEED_POLICY != "qplus":
            raise ValueError("New representation controls require RAG_PREHOP_ABLATION_PROFILE")
        return
    if profile not in PROFILES:
        raise ValueError(f"Unknown Prehop ablation profile: {profile}")
    if os.environ.get("RAG_PAPER_MODE", "false").lower() not in {"false", "0", "no", "off"}:
        raise ValueError("Representation ablations require RAG_PAPER_MODE=false")
    namespace = os.environ.get("RAG_INDEX_NAMESPACE", "")
    reuse = os.environ.get("RAG_ABLATION_REUSE_EXISTING_INDEX") == "true"
    if reuse and profile not in {"question_full", "question_body"}:
        raise ValueError("Existing index reuse is only available for A/B")
    pattern = r"[A-Za-z0-9_]+" if reuse else r"ablation_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*"
    if not re.fullmatch(pattern, namespace):
        raise ValueError("Representation ablations require an isolated ablation_ namespace")
    expected = {
        **COMMON,
        **PROFILES[profile],
        "ABLATION_Q_MINUS": profile != "body_body",
        "ABLATION_Q_PLUS": profile != "body_body",
    }
    mismatches = {
        key: (getattr(config, key), value) for key, value in expected.items() if getattr(config, key) != value
    }
    if mismatches:
        raise ValueError(f"Ablation profile configuration mismatch: {mismatches}")
    if profile == "body_body":
        from models.prehop.indexing.body_links import load_reference

        reference = load_reference(config.BODY_LINK_REFERENCE)
        namespace = os.environ.get("RAG_INDEX_NAMESPACE", "")
        if not namespace.startswith("ablation_") or namespace == reference["namespace"]:
            raise ValueError("Body links require a separate ablation_ index namespace")


def ablation_identity(config=None):
    if config is None:
        from core.config import RAGConfig

        config = RAGConfig
    if not config.PREHOP_ABLATION_PROFILE:
        return {}
    validate_profile(config)
    identity = {
        "representation_ablation": config.PREHOP_ABLATION_PROFILE,
        "method_contract": "prehop-representation-ablation-v1",
        "hop_seed_policy": config.HOP_SEED_POLICY,
        "hop_link_variant": config.HOP_LINK_VARIANT,
        "comparison_scope": "ablation_only",
    }
    if config.HOP_LINK_VARIANT == "body":
        import hashlib

        identity["body_link_reference_sha256"] = hashlib.sha256(
            Path(config.BODY_LINK_REFERENCE).read_bytes()
        ).hexdigest()
    return identity


def validate_body_index_policy(policy, digest):
    """Validate an ablation index without admitting it to primary paper policy."""
    from core.config import RAGConfig
    from core.semantic_config import semantic_config_sha256

    validate_profile(RAGConfig)
    identity = ablation_identity()
    expected = {
        "strategy": "prehop",
        "hop_construction": "body_to_body",
        "body_link_reference_sha256": identity["body_link_reference_sha256"],
        "body_link_degree_policy": "reference_per_node",
        "q_minus_enabled": False,
        "q_plus_enabled": False,
        "question_schema": "legacy",
        "chunk_sentences": RAGConfig.CHUNK_SENTENCES,
        "sentence_channel_enabled": False,
        "embedding_model": RAGConfig.EMBEDDING_MODEL,
        "embedding_dimensions": RAGConfig.EMBEDDING_DIMENSIONS,
        "embedding_query_instruction": RAGConfig.EMBEDDING_QUERY_INSTRUCTION,
        "fulltext_analyzer": RAGConfig.FULLTEXT_ANALYZER,
    }
    if digest != semantic_config_sha256(policy) or any(policy.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Body ablation index policy/hash does not match the selected method")


def validate_ablation_index_policy(policy, digest, dataset):
    from cli.index import _resolved_index_policy
    from core.config import RAGConfig
    from core.semantic_config import semantic_config_sha256, semantic_index_policy

    validate_profile(RAGConfig)
    if not RAGConfig.PREHOP_ABLATION_PROFILE or policy.get("strategy") != "prehop":
        raise RuntimeError("Ablation requires a Prehop index")
    if digest != semantic_config_sha256(policy):
        raise RuntimeError("Ablation index policy digest mismatch")
    expected = semantic_index_policy(
        _resolved_index_policy("prehop", policy.get("indexing_model") or "default", dataset)
    )
    observed = semantic_index_policy(policy)
    # Index construction throughput and its historical sampling seed are
    # provenance, not query-time controls; retain them in the source artifact.
    for key in ("operational_config", "generation_seed"):
        observed.pop(key, None)
        expected.pop(key, None)
    if os.environ.get("RAG_ABLATION_REUSE_EXISTING_INDEX") == "true":
        # Historical primary indexes include additional provenance fields that
        # non-paper construction does not emit. Compare construction controls,
        # retaining the original full policy and its digest in the result.
        keys = {"strategy", "embedding_model", "embedding_dimensions",
                "embedding_query_instruction", "embedding_max_input_tokens",
                "fulltext_analyzer", "chunk_sentences", "questions_per_direction",
                "question_schema", "q_minus_enabled", "q_plus_enabled",
                "sentence_channel_enabled", "precompute_reciprocal_hops",
                "continuation_edges_materialized", "continuation_anchor_policy",
                "hop_construction"}
        observed = {key: observed.get(key) for key in keys}
        expected = {key: expected.get(key) for key in keys}
    if observed != expected:
        raise RuntimeError("Ablation index settings differ from the selected representation method")
    if RAGConfig.HOP_LINK_VARIANT == "body":
        validate_body_index_policy(policy, digest)


async def require_empty_ablation_namespace():
    from core.config import RAGConfig
    from core.neo4j_service import Neo4jService

    validate_profile(RAGConfig)
    if os.environ.get("RAG_ABLATION_REUSE_EXISTING_INDEX") == "true":
        raise ValueError("Existing ablation indexes are read-only")
    if RAGConfig.PREHOP_ABLATION_PROFILE == "question_body":
        raise ValueError("question_body must reuse the frozen question_full graph")
    namespace = os.environ["RAG_INDEX_NAMESPACE"]
    rows = await Neo4jService().execute_query(
        f"MATCH (n) WHERE any(label IN labels(n) WHERE label STARTS WITH 'PR_{namespace}_') RETURN count(n) AS count"
    )
    if not rows or rows[0]["count"]:
        raise RuntimeError("Ablation indexing requires a fresh empty namespace; existing data is preserved")
