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
    return raw.strip().lower() or default


class RAGConfig:
    PREHOP_ABLATION_PROFILE = os.environ.get("RAG_PREHOP_ABLATION_PROFILE", "").strip()
    HOP_SEED_POLICY = _query_setting("hop_seed_policy")
    HOP_LINK_VARIANT = os.environ.get("RAG_HOP_LINK_VARIANT", "question").strip()
    BODY_LINK_REFERENCE = os.environ.get("RAG_BODY_LINK_REFERENCE", "").strip()
    CONNECTION_TIMING_MODE = os.environ.get("RAG_CONNECTION_TIMING_MODE", "").strip()
    CONNECTION_TIMING_STORE = os.environ.get("RAG_CONNECTION_TIMING_STORE", "").strip()

    # --- Evaluation (LLM-as-a-judge) ---
    EVAL_MODEL = os.environ.get("EVAL_MODEL", "").strip()
    # LLM-as-a-judge is optional, supplemental analysis.  Deterministic and
    # official benchmark metrics must be runnable without an evaluator API.
    # Enable it explicitly for a separately labelled judge analysis.
    JUDGE_ENABLED = parse_strict_bool(os.environ.get("RAG_JUDGE_ENABLED", "false"), name="RAG_JUDGE_ENABLED")
    # Debug-only escape hatch. Paper artifacts must use an evaluator distinct
    # from the requested generation model.
    JUDGE_ALLOW_SELF = parse_strict_bool(
        os.environ.get("RAG_JUDGE_ALLOW_SELF", "false"), name="RAG_JUDGE_ALLOW_SELF"
    )
    # Retained for an explicit error when unsupported Batch judging is requested.
    JUDGE_BATCH = parse_strict_bool(os.environ.get("RAG_JUDGE_BATCH", "false"), name="RAG_JUDGE_BATCH")

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
    # ``grounded_v1`` requires source-verifiable structured provenance.
    QUESTION_SCHEMA = _query_setting("question_schema")
    # HOP ANN sends high-dimensional vectors and candidate rows through bounded waves.
    HOP_GATHER_WAVE = int(os.environ.get("RAG_HOP_GATHER_WAVE") or "64")
    HOP_BUILD_CONCURRENCY = int(os.environ.get("RAG_HOP_BUILD_CONCURRENCY") or "4")
    DEFAULT_TOP_K = _query_setting("default_top_k")
    CANDIDATE_POOL_MULTIPLIER = _query_setting("candidate_pool_multiplier")
    FULLTEXT_ANALYZER = _query_setting("fulltext_analyzer")

    # Zero disables graph expansion for ablation; one enables the fixed
    # bidirectional NEXT and outgoing HOP_ANSWER expansion.
    GRAPH_HOP_DEPTH = _query_setting("graph_hop_depth")
    GRAPH_PATH_DECAY = _query_setting("graph_path_decay")
    GRAPH_EDGE_VARIANT = _query_setting("graph_edge_variant")
    # Read-only traversal policy over the existing offline graph. The final
    # cross-dataset selection uses every materialized HOP edge; reciprocal
    # filtering remains available as an explicit ablation.
    HOP_EDGE_FILTER = _query_setting("hop_edge_filter")
    # ``exact`` activates only HOP provenance attached to query-matched Q+ IDs;
    # ``owner`` activates all provenance on a matched Q+ owner.
    QPLUS_HOP_ACTIVATION = _query_setting("qplus_hop_activation")
    # Query-time ablation over the separately indexed linked_v2 continuation
    # relations. Indexing always materializes them for that schema so on/off
    # comparisons share the exact same question and graph snapshot.
    CONTINUATION_EDGES_ENABLED = _query_setting("continuation_edges_enabled")
    # Index-time structural policy for linked_v2 answer anchors. ``named_only``
    # uses the generation contract's optional specific-entity marker;
    # ``all_grounded`` uses every complete source-verifiable Q- answer.
    CONTINUATION_ANCHOR_POLICY = _query_setting("continuation_anchor_policy")
    # Query-time semantic evidence for traversed HOP targets. The conservative
    # historical policy requires both query-to-body and query-to-source-Q+ similarity;
    # ``bridge_only`` is a parameter-free structural ablation because the
    # offline Q+->Q- edge has already selected the answering target.
    HOP_SEMANTIC_VARIANT = _query_setting("hop_semantic_variant")
    # Optional index-time materialization avoids reverse vector ANN on every
    # reciprocal-filtered query while preserving the same nearest-neighbour rule.
    PRECOMPUTE_RECIPROCAL_HOPS = _query_setting("precompute_reciprocal_hops")
    # --- Ablation & Experimental Toggles ---
    # Q-/Q+ channel ablations.
    # ABLATION_Q_MINUS / ABLATION_Q_PLUS gate whether the Q-/Q+ channels
    # participate in indexing (embedding storage) and retrieval (channel use).
    # Disabling Q+ also disables offline HOP edge construction, since HOP
    # selection is anchored on Q+ embeddings. The explicit body-link profile
    # instead constructs body-to-body edges with both question channels off.
    ABLATION_Q_MINUS = _query_setting("q_minus")
    ABLATION_Q_PLUS = _query_setting("q_plus")
    # Optional fine-grained retrieval representation. Sentence nodes are
    # deterministic children of the fixed output chunks and always collapse
    # back to those owners before ranking, so enabling this changes candidate
    # generation without changing the evidence unit or final top-k.
    SENTENCE_CHANNEL_ENABLED = _query_setting("sentence_channel_enabled")

    # Select which Q-/Q+ representation channels retrieve.py queries.
    # Values:
    #   "body_only"       -> body direct evidence only.
    #   "full"            -> Q-/body direct evidence plus Q+
    #                        dependency seeds in one set union.
    #   "qminus_only"     -> Q- only, direct evidence role.
    #   "qplus_only"      -> Q+ only, dependency-seed role.
    #   "single_combined" -> Q- and Q+ queried once and combined by set union,
    #                        with no body channel.
    # No re-indexing required; only retrieval-time channel selection changes.
    HYPO_CHANNEL_VARIANT = _query_setting("hypo_channel_variant")
    SOURCE_SELECTION_VARIANT = _query_setting("source_selection_variant")
    # Input-order control for the generation-model candidate-ordering call.
    # ``search`` preserves the primary system; alternatives are diagnostics
    # over the same frozen candidate pool.
    CANDIDATE_ORDER_INPUT_ORDER = _query_setting("candidate_order_input_order")
    CANDIDATE_ORDER_SHUFFLE_SEED = _query_setting("candidate_order_shuffle_seed")
    FINAL_RANK_VARIANT = _query_setting("final_rank_variant")
