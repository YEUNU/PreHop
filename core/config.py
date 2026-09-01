import os


class RAGConfig:
    # --- Infrastructure (Actual ports identified) ---
    # Required external OpenAI-compatible endpoints. There is intentionally no
    # localhost fallback: missing configuration must fail before inference.
    VLLM_URL = os.environ.get("VLLM_URL", "").strip()
    VLLM_EMBED_URL = os.environ.get("VLLM_EMBED_URL", "").strip()

    # --- LLM Settings ---
    DEFAULT_MODEL = os.environ.get("VLLM_SERVED_MODEL_NAME", "generation-model")
    EMBEDDING_MODEL = os.environ.get("VLLM_SERVED_EMBED_MODEL_NAME", "embedding-model")

    # --- Evaluation (LLM-as-a-judge) ---
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
    EVAL_MODEL = os.environ.get("EVAL_MODEL", "").strip()
    # LLM-as-a-judge is optional, supplemental analysis.  Deterministic and
    # official benchmark metrics must be runnable without an evaluator API.
    # Enable it explicitly for a separately labelled judge analysis.
    JUDGE_ENABLED = os.environ.get("RAG_JUDGE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    # Debug-only escape hatch. Paper artifacts must use an evaluator distinct
    # from both the requested generation model and DEFAULT_MODEL.
    JUDGE_ALLOW_SELF = os.environ.get("RAG_JUDGE_ALLOW_SELF", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # When the optional judge is enabled, Batch is preferred unless an
    # explicit synchronous debugging run requests otherwise.
    JUDGE_BATCH = os.environ.get("RAG_JUDGE_BATCH", "true").strip().lower() in {"1", "true", "yes", "on"}
    JUDGE_BATCH_POLL_SECONDS = max(2, int(os.environ.get("RAG_JUDGE_BATCH_POLL_SECONDS", "15")))

    # --- Common Service Settings ---
    RETRY_COUNT = int(os.environ.get("RAG_RETRY_COUNT", "3"))
    RETRY_DELAY = float(os.environ.get("RAG_RETRY_DELAY", "2.0"))
    LLM_REQUEST_TIMEOUT = float(os.environ.get("LLM_REQUEST_TIMEOUT", "300"))
    LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "5"))
    LLM_RETRY_DELAY = float(os.environ.get("LLM_RETRY_DELAY", "2.0"))
    # Per-call sampling seed forwarded to external chat.completions when set
    # (multi-seed benchmarking). Empty/missing => no seed (engine default).
    _LLM_SEED_RAW = os.environ.get("RAG_LLM_SEED", "").strip()
    LLM_SEED = int(_LLM_SEED_RAW) if _LLM_SEED_RAW.lstrip("-").isdigit() else None
    MAX_CONTEXT_LENGTH = int(os.environ.get("RAG_MAX_CONTEXT_LENGTH", "262144"))
    # Capped below the configured context limit so input has output headroom.
    # Indexing prompts for Q-/Q+ generation rarely exceed 1–2K output;
    # 4K is comfortable headroom.
    MAX_OUTPUT_TOKENS = 4096
    # The benchmark contract requests a short final answer, not free-form
    # generation. This fixed cap reserves context space and bounds latency; it
    # is shared by controlled answer-synthesis paths and is not swept.
    SYNTHESIS_MAX_OUTPUT_TOKENS = 128
    MAX_EMBEDDING_LENGTH = int(os.environ.get("MAX_EMBEDDING_LENGTH", "32768"))
    EMBEDDING_QUERY_INSTRUCTION = os.environ.get(
        "EMBEDDING_QUERY_INSTRUCTION",
        "Given a web search query, retrieve relevant passages that answer the query",
    ).strip()

    # --- RAG & Indexing Settings ---
    MAX_CONCURRENT_LLM_CALLS = int(os.environ.get("MAX_CONCURRENT_LLM_CALLS", "30"))
    MAX_CONCURRENT_EMBEDDING_REQUESTS = int(os.environ.get("RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS", "1"))
    EMBEDDING_BATCH_SIZE = int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE", "512"))
    VLLM_MAX_NUM_SEQS = int(os.environ.get("VLLM_MAX_NUM_SEQS", "120"))
    EMBEDDING_MAX_NUM_SEQS = int(os.environ.get("EMBEDDING_MAX_NUM_SEQS", "512"))
    EMBEDDING_DIMENSIONS = int(os.environ.get("NEO4J_VECTOR_DIMENSIONS", "1024"))
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
    HOP_GATHER_WAVE = int(os.environ.get("RAG_HOP_GATHER_WAVE", "64"))
    HOP_BUILD_CONCURRENCY = int(os.environ.get("RAG_HOP_BUILD_CONCURRENCY", "4"))
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
    CONTINUATION_EDGES_ENABLED = os.environ.get("RAG_CONTINUATION_EDGES_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Index-time structural policy for linked_v2 answer anchors. ``named_only``
    # uses the generation contract's optional specific-entity marker;
    # ``all_grounded`` uses every complete source-verifiable Q- answer.
    CONTINUATION_ANCHOR_POLICY = (
        os.environ.get("RAG_CONTINUATION_ANCHOR_POLICY", "named_only").strip().lower() or "named_only"
    )
    # Query-time semantic evidence for traversed HOP targets. The conservative
    # default requires both query-to-body and query-to-source-Q+ similarity;
    # ``bridge_only`` is a parameter-free structural ablation because the
    # offline Q+->Q- edge has already selected the answering target.
    HOP_SEMANTIC_VARIANT = (
        os.environ.get("RAG_HOP_SEMANTIC_VARIANT", "body_bridge_min").strip().lower() or "body_bridge_min"
    )
    # Optional index-time materialization avoids reverse vector ANN on every
    # reciprocal-filtered query while preserving the same nearest-neighbour rule.
    PRECOMPUTE_RECIPROCAL_HOPS = os.environ.get("RAG_PRECOMPUTE_RECIPROCAL_HOPS", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    QUERY_REWRITE_VARIANT = (
        os.environ.get("RAG_QUERY_REWRITE_VARIANT", "role_aligned_evidence_iterative").strip().lower()
        or "role_aligned_evidence_iterative"
    )
    # Role rewriting is useful for compact compositional questions but can
    # perturb the explicit source and relation constraints already present in
    # long questions. Zero disables this input-length gate for ablations.
    QUERY_REWRITE_MAX_WORDS = int(os.environ.get("RAG_QUERY_REWRITE_MAX_WORDS", "32"))
    # Operational guard for evidence-conditioned rewrite calls. Zero keeps
    # the evidence-driven stopping rule; positive values cap refinement calls.
    QUERY_REFINEMENT_MAX_ROUNDS = int(os.environ.get("RAG_QUERY_REFINEMENT_MAX_ROUNDS", "0"))

    # --- Ablation & Experimental Toggles ---
    # Q-/Q+ channel ablations.
    # ABLATION_Q_MINUS / ABLATION_Q_PLUS gate whether the Q-/Q+ channels
    # participate in indexing (embedding storage) and retrieval (channel use).
    # Disabling Q+ also disables offline HOP edge construction, since HOP
    # selection is anchored on Q+ embeddings.
    ABLATION_Q_MINUS = os.environ.get("RAG_ABLATION_Q_MINUS", "True").lower() == "true"
    ABLATION_Q_PLUS = os.environ.get("RAG_ABLATION_Q_PLUS", "True").lower() == "true"
    # Optional fine-grained retrieval representation. Sentence nodes are
    # deterministic children of the fixed output chunks and always collapse
    # back to those owners before ranking, so enabling this changes candidate
    # generation without changing the evidence unit or final top-k.
    SENTENCE_CHANNEL_ENABLED = os.environ.get("RAG_SENTENCE_CHANNEL_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

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

    @classmethod
    def validate(cls) -> None:
        positive = {
            "RETRY_COUNT": cls.RETRY_COUNT,
            "MAX_CONCURRENT_LLM_CALLS": cls.MAX_CONCURRENT_LLM_CALLS,
            "MAX_CONCURRENT_EMBEDDING_REQUESTS": cls.MAX_CONCURRENT_EMBEDDING_REQUESTS,
            "EMBEDDING_BATCH_SIZE": cls.EMBEDDING_BATCH_SIZE,
            "VLLM_MAX_NUM_SEQS": cls.VLLM_MAX_NUM_SEQS,
            "EMBEDDING_MAX_NUM_SEQS": cls.EMBEDDING_MAX_NUM_SEQS,
            "EMBEDDING_DIMENSIONS": cls.EMBEDDING_DIMENSIONS,
            "NEO4J_BATCH_SIZE": cls.NEO4J_BATCH_SIZE,
            "CHUNK_SENTENCES": cls.CHUNK_SENTENCES,
            "QUESTIONS_PER_DIRECTION": cls.QUESTIONS_PER_DIRECTION,
            "HOP_GATHER_WAVE": cls.HOP_GATHER_WAVE,
            "HOP_BUILD_CONCURRENCY": cls.HOP_BUILD_CONCURRENCY,
            "DEFAULT_TOP_K": cls.DEFAULT_TOP_K,
            "CANDIDATE_POOL_MULTIPLIER": cls.CANDIDATE_POOL_MULTIPLIER,
        }
        invalid = {name: value for name, value in positive.items() if value < 1}
        if invalid:
            raise ValueError(f"RAG configuration values must be positive: {invalid}")
        if cls.EMBEDDING_BATCH_SIZE * cls.MAX_CONCURRENT_EMBEDDING_REQUESTS > cls.EMBEDDING_MAX_NUM_SEQS:
            raise ValueError(
                "Embedding client can exceed EMBEDDING_MAX_NUM_SEQS: "
                f"batch={cls.EMBEDDING_BATCH_SIZE} * concurrent_requests="
                f"{cls.MAX_CONCURRENT_EMBEDDING_REQUESTS} > {cls.EMBEDDING_MAX_NUM_SEQS}"
            )
        if cls.MAX_CONCURRENT_LLM_CALLS > cls.VLLM_MAX_NUM_SEQS:
            raise ValueError(
                "Generation client can exceed VLLM_MAX_NUM_SEQS: "
                f"concurrent_calls={cls.MAX_CONCURRENT_LLM_CALLS} > {cls.VLLM_MAX_NUM_SEQS}"
            )
        if cls.GRAPH_HOP_DEPTH not in {0, 1}:
            raise ValueError("RAG_GRAPH_HOP_DEPTH must be 0 or 1")
        if not 0.0 <= cls.GRAPH_PATH_DECAY <= 1.0:
            raise ValueError("RAG_GRAPH_PATH_DECAY must be between 0 and 1")
        if cls.GRAPH_EDGE_VARIANT not in {"full", "hop_only", "next_only"}:
            raise ValueError("RAG_GRAPH_EDGE_VARIANT must be full, hop_only, or next_only")
        if cls.HOP_EDGE_FILTER not in {"none", "reciprocal", "reciprocal_offline"}:
            raise ValueError("RAG_HOP_EDGE_FILTER must be none, reciprocal, or reciprocal_offline")
        if cls.QPLUS_HOP_ACTIVATION not in {"exact", "owner"}:
            raise ValueError("RAG_QPLUS_HOP_ACTIVATION must be exact or owner")
        if cls.CONTINUATION_EDGES_ENABLED and cls.QUESTION_SCHEMA != "linked_v2":
            raise ValueError("RAG_CONTINUATION_EDGES_ENABLED=true requires RAG_QUESTION_SCHEMA=linked_v2")
        if cls.CONTINUATION_EDGES_ENABLED and not cls.ABLATION_Q_MINUS:
            raise ValueError("RAG_CONTINUATION_EDGES_ENABLED=true requires Q- retrieval")
        if cls.CONTINUATION_ANCHOR_POLICY not in {"named_only", "all_grounded"}:
            raise ValueError("RAG_CONTINUATION_ANCHOR_POLICY must be named_only or all_grounded")
        if cls.CONTINUATION_ANCHOR_POLICY != "named_only" and cls.QUESTION_SCHEMA != "linked_v2":
            raise ValueError("RAG_CONTINUATION_ANCHOR_POLICY=all_grounded requires RAG_QUESTION_SCHEMA=linked_v2")
        if cls.HOP_SEMANTIC_VARIANT not in {"body_bridge_min", "body_only", "bridge_only"}:
            raise ValueError("RAG_HOP_SEMANTIC_VARIANT must be body_bridge_min, body_only, or bridge_only")
        if cls.HOP_EDGE_FILTER == "reciprocal_offline" and not cls.PRECOMPUTE_RECIPROCAL_HOPS:
            raise ValueError("RAG_HOP_EDGE_FILTER=reciprocal_offline requires RAG_PRECOMPUTE_RECIPROCAL_HOPS=true")
        if cls.QUESTION_SCHEMA not in {"legacy", "grounded_v1", "linked_v2"}:
            raise ValueError("RAG_QUESTION_SCHEMA must be legacy, grounded_v1, or linked_v2")
        if cls.QUERY_REWRITE_VARIANT not in {
            "none",
            "role_aligned",
            "role_aligned_additive",
            "role_aligned_evidence",
            "role_aligned_evidence_iterative",
        }:
            raise ValueError(
                "RAG_QUERY_REWRITE_VARIANT must be none, role_aligned, "
                "role_aligned_additive, role_aligned_evidence, or "
                "role_aligned_evidence_iterative"
            )
        if cls.QUERY_REWRITE_MAX_WORDS < 0:
            raise ValueError("RAG_QUERY_REWRITE_MAX_WORDS must be zero or positive")
        if cls.QUERY_REFINEMENT_MAX_ROUNDS < 0:
            raise ValueError("RAG_QUERY_REFINEMENT_MAX_ROUNDS must be zero or positive")
        if not cls.EMBEDDING_QUERY_INSTRUCTION:
            raise ValueError("EMBEDDING_QUERY_INSTRUCTION must not be empty")

        allowed_variants = {"body_only", "full", "qminus_only", "qplus_only", "single_combined"}
        if cls.HYPO_CHANNEL_VARIANT not in allowed_variants:
            raise ValueError(
                f"RAG_HYPO_CHANNEL_VARIANT={cls.HYPO_CHANNEL_VARIANT!r} is invalid; "
                f"expected one of {sorted(allowed_variants)}"
            )
        if cls.HYPO_CHANNEL_VARIANT == "qminus_only" and not cls.ABLATION_Q_MINUS:
            raise ValueError("qminus_only requires RAG_ABLATION_Q_MINUS=true")
        if cls.HYPO_CHANNEL_VARIANT == "qplus_only" and not cls.ABLATION_Q_PLUS:
            raise ValueError("qplus_only requires RAG_ABLATION_Q_PLUS=true")
        if cls.HYPO_CHANNEL_VARIANT == "single_combined" and not (cls.ABLATION_Q_MINUS and cls.ABLATION_Q_PLUS):
            raise ValueError("single_combined requires both Q- and Q+ channels")
        if cls.SOURCE_SELECTION_VARIANT not in {
            "graph_pairs",
            "source_balanced_graph_pairs",
            "source_balanced",
            "round_robin",
            "role_body_owners",
            "role_body_rounds",
            "role_body_list_ranking",
            "global",
        }:
            raise ValueError(
                "RAG_SOURCE_SELECTION_VARIANT must be graph_pairs, source_balanced, "
                "source_balanced_graph_pairs, round_robin, role_body_owners, "
                "role_body_rounds, role_body_list_ranking, or global"
            )
        if cls.CANDIDATE_ORDER_INPUT_ORDER not in {"search", "reverse", "hash_shuffle"}:
            raise ValueError("RAG_CANDIDATE_ORDER_INPUT_ORDER must be search, reverse, or hash_shuffle")
        if cls.FINAL_RANK_VARIANT not in {"fused", "semantic_only", "representation_only"}:
            raise ValueError("RAG_FINAL_RANK_VARIANT must be fused, semantic_only, or representation_only")
