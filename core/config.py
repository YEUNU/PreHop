import os

from core.semantic_config import parse_strict_bool


class RAGConfig:
    PREHOP_ABLATION_PROFILE = os.environ.get("RAG_PREHOP_ABLATION_PROFILE", "").strip()
    HOP_SEED_POLICY = os.environ.get("RAG_HOP_SEED_POLICY", "all").strip()
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
    # When the optional judge is enabled, Batch is preferred unless an
    # explicit synchronous debugging run requests otherwise.
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
    CHUNK_SENTENCES = 6
    QUESTIONS_PER_DIRECTION = 3
    # Index-time question output contract. ``legacy`` keeps string-only Q-/Q+;
    # ``grounded_v1`` requires source-verifiable structured provenance.
    QUESTION_SCHEMA = os.environ.get("RAG_QUESTION_SCHEMA", "legacy").strip().lower() or "legacy"
    # HOP ANN sends high-dimensional vectors and candidate rows through bounded waves.
    HOP_GATHER_WAVE = int(os.environ.get("RAG_HOP_GATHER_WAVE") or "64")
    HOP_BUILD_CONCURRENCY = int(os.environ.get("RAG_HOP_BUILD_CONCURRENCY") or "4")
    DEFAULT_TOP_K = 12
    CANDIDATE_POOL_MULTIPLIER = int(os.environ.get("RAG_CANDIDATE_POOL_MULTIPLIER", "1"))
    FULLTEXT_ANALYZER = os.environ.get("NEO4J_FULLTEXT_ANALYZER", "english")

    # Zero disables graph expansion for ablation; one enables the fixed
    # bidirectional NEXT and outgoing HOP_ANSWER expansion.
    GRAPH_HOP_DEPTH = int(os.environ.get("RAG_GRAPH_HOP_DEPTH", "1"))
    GRAPH_PATH_DECAY = float(os.environ.get("RAG_GRAPH_PATH_DECAY", "0.5"))
    GRAPH_EDGE_VARIANT = os.environ.get("RAG_GRAPH_EDGE_VARIANT", "full").strip().lower() or "full"
    # Read-only traversal policy over the existing offline graph. The final
    # cross-dataset selection uses every materialized HOP edge; reciprocal
    # filtering remains available as an explicit ablation.
    HOP_EDGE_FILTER = os.environ.get("RAG_HOP_EDGE_FILTER", "none").strip().lower() or "none"
    # ``exact`` activates only HOP provenance attached to query-matched Q+ IDs;
    # ``owner`` activates all provenance on a matched Q+ owner.
    QPLUS_HOP_ACTIVATION = os.environ.get("RAG_QPLUS_HOP_ACTIVATION", "owner").strip().lower() or "owner"
    # Query-time ablation over the separately indexed linked_v2 continuation
    # relations. Indexing always materializes them for that schema so on/off
    # comparisons share the exact same question and graph snapshot.
    CONTINUATION_EDGES_ENABLED = parse_strict_bool(
        os.environ.get("RAG_CONTINUATION_EDGES_ENABLED", "false"), name="RAG_CONTINUATION_EDGES_ENABLED"
    )
    # Index-time structural policy for linked_v2 answer anchors. ``named_only``
    # uses the generation contract's optional specific-entity marker;
    # ``all_grounded`` uses every complete source-verifiable Q- answer.
    CONTINUATION_ANCHOR_POLICY = (
        os.environ.get("RAG_CONTINUATION_ANCHOR_POLICY", "named_only").strip().lower() or "named_only"
    )
    # Query-time semantic evidence for traversed HOP targets. The conservative
    # historical policy requires both query-to-body and query-to-source-Q+ similarity;
    # ``bridge_only`` is a parameter-free structural ablation because the
    # offline Q+->Q- edge has already selected the answering target.
    HOP_SEMANTIC_VARIANT = (
        os.environ.get("RAG_HOP_SEMANTIC_VARIANT", "body_only").strip().lower() or "body_only"
    )
    # Optional index-time materialization avoids reverse vector ANN on every
    # reciprocal-filtered query while preserving the same nearest-neighbour rule.
    PRECOMPUTE_RECIPROCAL_HOPS = parse_strict_bool(
        os.environ.get("RAG_PRECOMPUTE_RECIPROCAL_HOPS", "true"), name="RAG_PRECOMPUTE_RECIPROCAL_HOPS"
    )
    # --- Ablation & Experimental Toggles ---
    # Q-/Q+ channel ablations.
    # ABLATION_Q_MINUS / ABLATION_Q_PLUS gate whether the Q-/Q+ channels
    # participate in indexing (embedding storage) and retrieval (channel use).
    # Disabling Q+ also disables offline HOP edge construction, since HOP
    # selection is anchored on Q+ embeddings. The explicit body-link profile
    # instead constructs body-to-body edges with both question channels off.
    ABLATION_Q_MINUS = parse_strict_bool(
        os.environ.get("RAG_ABLATION_Q_MINUS", "true"), name="RAG_ABLATION_Q_MINUS"
    )
    ABLATION_Q_PLUS = parse_strict_bool(
        os.environ.get("RAG_ABLATION_Q_PLUS", "true"), name="RAG_ABLATION_Q_PLUS"
    )
    # Optional fine-grained retrieval representation. Sentence nodes are
    # deterministic children of the fixed output chunks and always collapse
    # back to those owners before ranking, so enabling this changes candidate
    # generation without changing the evidence unit or final top-k.
    SENTENCE_CHANNEL_ENABLED = parse_strict_bool(
        os.environ.get("RAG_SENTENCE_CHANNEL_ENABLED", "false"), name="RAG_SENTENCE_CHANNEL_ENABLED"
    )

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
    HYPO_CHANNEL_VARIANT = os.environ.get("RAG_HYPO_CHANNEL_VARIANT", "full").strip().lower() or "full"
    SOURCE_SELECTION_VARIANT = (
        os.environ.get("RAG_SOURCE_SELECTION_VARIANT", "role_body_list_ranking").strip().lower()
        or "role_body_list_ranking"
    )
    # Input-order control for the generation-model candidate-ordering call.
    # ``search`` preserves the primary system; alternatives are diagnostics
    # over the same frozen candidate pool.
    CANDIDATE_ORDER_INPUT_ORDER = (
        os.environ.get("RAG_CANDIDATE_ORDER_INPUT_ORDER", "search").strip().lower() or "search"
    )
    CANDIDATE_ORDER_SHUFFLE_SEED = int(os.environ.get("RAG_CANDIDATE_ORDER_SHUFFLE_SEED", "0"))
    FINAL_RANK_VARIANT = os.environ.get("RAG_FINAL_RANK_VARIANT", "fused").strip().lower() or "fused"
