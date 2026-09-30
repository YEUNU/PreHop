import os

from core.semantic_config import parse_strict_bool
from core.strategy_registry import get_strategy

_QUERY_FIELDS = {field: (name, default) for field, name, default in get_strategy("prehop").paper_query_policy}


def _query_setting(field: str):
    """Resolve the registry's query defaults using the existing core parsing rules."""
    name, default = _QUERY_FIELDS[field]
    raw = os.environ.get(name) if name else None
    if raw is None:
        return default
    if isinstance(default, bool):
        return parse_strict_bool(raw, name=name)
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if field == "fulltext_analyzer":
        return raw
    if field == "hop_seed_policy":
        return raw.strip()
    value = raw.strip().lower() or default
    supported = {
        "hypo_channel_variant": {"full", "body_only"},
        "source_selection_variant": {"role_body_list_ranking", "global"},
        "final_rank_variant": {"fused", "representation_only"},
        "hop_semantic_variant": {"body_only", "body_bridge_min"},
        "graph_edge_variant": {"full", "next_only", "hop_only", "none"},
    }
    if field in supported and value not in supported[field]:
        raise ValueError(f"Unsupported {name}: {value}")
    return value


class RAGConfig:
    PREHOP_ABLATION_PROFILE = os.environ.get("RAG_PREHOP_ABLATION_PROFILE", "").strip()
    HOP_SEED_POLICY = _query_setting("hop_seed_policy")
    HOP_LINK_VARIANT = "question"

    # --- Common Service Settings ---
    RETRY_COUNT = int(os.environ.get("RAG_RETRY_COUNT", "3"))
    RETRY_DELAY = float(os.environ.get("RAG_RETRY_DELAY", "2.0"))
    LLM_RETRY_DELAY = float(os.environ.get("LLM_RETRY_DELAY", "2.0"))
    # Capped below the configured context limit so input has output headroom.
    # Indexing prompts for Q-/Q+ generation rarely exceed 1–2K output;
    # 4K is comfortable headroom.
    MAX_OUTPUT_TOKENS = 4096
    # Reader-specific caps also reserve space during context fitting.
    SYNTHESIS_MAX_OUTPUT_TOKENS = 128
    # Prehop emits a brief evidence check before its labelled final answer.
    PREHOP_SYNTHESIS_MAX_OUTPUT_TOKENS = 256

    # --- RAG & Indexing Settings ---
    NEO4J_BATCH_SIZE = int(os.environ.get("NEO4J_BATCH_SIZE", "25"))

    # --- Search & Ranking ---
    # Query-time channels use unweighted reciprocal rank, 1 / (rank + 1).
    # There is no dataset-tuned fusion constant or modality preference.
    # --- Indexing Pipeline Settings ---
    # Each page is split into fixed CHUNK_SENTENCES windows, including the
    # final partial window.
    CHUNK_SENTENCES = _query_setting("chunk_sentences")
    QUESTIONS_PER_DIRECTION = _query_setting("questions_per_direction")
    # Index-time question output contract. ``legacy`` keeps string-only Q-/Q+;
    QUESTION_SCHEMA = _query_setting("question_schema")
    # HOP ANN sends high-dimensional vectors and candidate rows through bounded waves.
    HOP_GATHER_WAVE = int(os.environ.get("RAG_HOP_GATHER_WAVE") or "64")
    HOP_BUILD_CONCURRENCY = int(os.environ.get("RAG_HOP_BUILD_CONCURRENCY") or "4")
    DEFAULT_TOP_K = _query_setting("default_top_k")
    CANDIDATE_POOL_MULTIPLIER = _query_setting("candidate_pool_multiplier")
    FULLTEXT_ANALYZER = _query_setting("fulltext_analyzer")

    # Fixed one-step expansion; GRAPH_EDGE_VARIANT controls the HOP/NEXT ablation.
    GRAPH_HOP_DEPTH = _query_setting("graph_hop_depth")
    GRAPH_PATH_DECAY = _query_setting("graph_path_decay")
    GRAPH_EDGE_VARIANT = _query_setting("graph_edge_variant")
    # Fixed historical identity fields; retired experiments have no execution path.
    HOP_EDGE_FILTER = _query_setting("hop_edge_filter")
    QPLUS_HOP_ACTIVATION = _query_setting("qplus_hop_activation")
    CONTINUATION_EDGES_ENABLED = _query_setting("continuation_edges_enabled")
    CONTINUATION_ANCHOR_POLICY = _query_setting("continuation_anchor_policy")
    # Query-time semantic evidence for traversed HOP targets. The conservative
    # historical policy requires both query-to-body and query-to-source-Q+ similarity;
    # offline Q+->Q- edge has already selected the answering target.
    HOP_SEMANTIC_VARIANT = _query_setting("hop_semantic_variant")
    # Preserve the published index materialization and its measured cost.
    PRECOMPUTE_RECIPROCAL_HOPS = _query_setting("precompute_reciprocal_hops")
    # Historical identity fields: both question roles are always indexed.
    # Search-channel comparisons vary HYPO_CHANNEL_VARIANT on that same index.
    ABLATION_Q_MINUS = _query_setting("q_minus")
    ABLATION_Q_PLUS = _query_setting("q_plus")
    SENTENCE_CHANNEL_ENABLED = _query_setting("sentence_channel_enabled")

    # Select which Q-/Q+ representation channels retrieve.py queries.
    # Values:
    #   "body_only"       -> body direct evidence only.
    #   "full"            -> Q-/body direct evidence plus Q+
    #                        dependency seeds in one set union.
    # No re-indexing required; only retrieval-time channel selection changes.
    HYPO_CHANNEL_VARIANT = _query_setting("hypo_channel_variant")
    SOURCE_SELECTION_VARIANT = _query_setting("source_selection_variant")
    CANDIDATE_ORDER_INPUT_ORDER = _query_setting("candidate_order_input_order")
    CANDIDATE_ORDER_SHUFFLE_SEED = _query_setting("candidate_order_shuffle_seed")
    FINAL_RANK_VARIANT = _query_setting("final_rank_variant")
