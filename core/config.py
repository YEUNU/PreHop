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
    # OpenAI Batch is the paper/default path because it is cheaper than
    # synchronous judge calls. Set false only for an explicit debug run.
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
    MAX_CONTEXT_LENGTH = int(os.environ.get("RAG_MAX_CONTEXT_LENGTH", "16384"))
    # Capped below the configured context limit so input has output headroom.
    # Indexing prompts (chunking, Q-/Q+, summary) rarely exceed 1–2K output;
    # 4K is comfortable headroom.
    MAX_OUTPUT_TOKENS = 4096
    MAX_EMBEDDING_LENGTH = int(os.environ.get("MAX_EMBEDDING_LENGTH", "16384"))

    # --- RAG & Indexing Settings ---
    MAX_CONCURRENT_LLM_CALLS = int(os.environ.get("MAX_CONCURRENT_LLM_CALLS", "30"))
    MAX_CONCURRENT_EMBEDDING_REQUESTS = int(os.environ.get("RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS", "2"))
    EMBEDDING_BATCH_SIZE = int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE", "32"))
    VLLM_MAX_NUM_SEQS = int(os.environ.get("VLLM_MAX_NUM_SEQS", "128"))
    EMBEDDING_DIMENSIONS = int(os.environ.get("NEO4J_VECTOR_DIMENSIONS", "1024"))
    NEO4J_BATCH_SIZE = int(os.environ.get("NEO4J_BATCH_SIZE", "25"))

    # --- Search & Ranking (RRF) ---
    RRF_K_CONSTANT = int(os.environ.get("RAG_RRF_K", "60"))
    # The vector channel carries semantic signal from the shared bi-encoder;
    # keep it slightly above the full-text contribution in RRF.
    RRF_VECTOR_WEIGHT = float(os.environ.get("RAG_RRF_VECTOR_WEIGHT", "1.3"))
    RRF_TEXT_WEIGHT = float(os.environ.get("RAG_RRF_TEXT_WEIGHT", "1.0"))
    VECTOR_SEARCH_LIMIT = int(os.environ.get("RAG_VECTOR_SEARCH_LIMIT", "20"))
    TEXT_SEARCH_LIMIT = int(os.environ.get("RAG_TEXT_SEARCH_LIMIT", "20"))

    # --- Indexing Pipeline Settings ---
    # Fixed-size chunking (core-only rewrite — replaces adaptive/embedding-
    # similarity chunk splitting). Each page is windowed into chunks of
    # CHUNK_SENTENCES sentences; a trailing window shorter than
    # MIN_CHUNK_SENTENCES merges into the previous chunk instead of standing
    # alone.
    CHUNK_SENTENCES = int(os.environ.get("RAG_CHUNK_SENTENCES", "6"))
    MIN_CHUNK_SENTENCES = int(os.environ.get("RAG_MIN_CHUNK_SENTENCES", "2"))
    HOP_LINK_LIMIT = int(os.environ.get("RAG_HOP_LINK_LIMIT", "5"))
    # Each individual Q+ searches all three document-side representations.
    # Neo4j applies WHERE after ANN candidate generation, so the ANN pool is
    # intentionally larger than the retained cross-document candidate list.
    HOP_CANDIDATE_LIMIT = int(os.environ.get("RAG_HOP_CANDIDATE_LIMIT", "15"))
    HOP_ANN_POOL = int(os.environ.get("RAG_HOP_ANN_POOL", "50"))
    # HOP ANN sends high-dimensional vectors and candidate rows through Neo4j
    # transactions. Keep waves/channels bounded independently from file/LLM
    # concurrency to stay below the database transaction-memory pool.
    HOP_GATHER_WAVE = int(os.environ.get("RAG_HOP_GATHER_WAVE", "64"))
    HOP_CHANNEL_CONCURRENCY = int(os.environ.get("RAG_HOP_CHANNEL_CONCURRENCY", "2"))
    # Q+->Q+ is supporting evidence for a direct Q+->Q-/body match, not an
    # independently traversable document edge.
    HOP_SAME_NEED_WEIGHT = float(os.environ.get("RAG_HOP_SAME_NEED_WEIGHT", "0.5"))
    GRAPH_SEARCH_LIMIT = int(os.environ.get("RAG_GRAPH_SEARCH_LIMIT", "20"))
    DEFAULT_TOP_K = int(os.environ.get("RAG_DEFAULT_TOP_K", "12"))
    FULLTEXT_ANALYZER = os.environ.get("NEO4J_FULLTEXT_ANALYZER", "english")

    # Final top-k selection is otherwise pure global score order, so several
    # near-duplicate high-scoring chunks from one source document can occupy
    # most of the evidence slots and crowd out a lower-scoring chunk that is
    # the only path to a second gold document. A fixed fraction of top_k
    # (rounded down, minimum 1) caps how many chunks a single source can
    # contribute before the remaining slots open up to other sources; excess
    # same-source candidates still backfill by score if there are not enough
    # distinct sources to fill top_k. One rule for every dataset/strategy —
    # no per-dataset tuning.
    MAX_CHUNKS_PER_SOURCE_FRACTION = float(os.environ.get("RAG_MAX_CHUNKS_PER_SOURCE_FRACTION", "0.34"))

    # Graph traversal depth on the query path. depth=0 = pure `retrieve()`
    # (Stage 1+2 RRF + similarity ordering, no graph expansion) for ablation; depth>0 uses
    # `graph_search` — deterministic traversal over the
    # NEXT/HOP edges built during indexing (paper §3.1.4), no LLM continuation
    # check. depth=1 is the default (bidirectional NEXT plus outgoing
    # HOP_ANSWER).
    GRAPH_HOP_DEPTH = int(os.environ.get("RAG_GRAPH_HOP_DEPTH", "1"))

    # --- Ablation & Experimental Toggles ---
    # Predictive Knowledge Mapping channel ablations.
    # ABLATION_Q_MINUS / ABLATION_Q_PLUS gate whether the Q-/Q+ channels
    # participate in indexing (embedding storage) and retrieval (channel use).
    # Disabling Q+ also disables offline HOP edge construction, since HOP
    # selection is anchored on Q+ embeddings.
    ABLATION_Q_MINUS = os.environ.get("RAG_ABLATION_Q_MINUS", "True").lower() == "true"
    ABLATION_Q_PLUS = os.environ.get("RAG_ABLATION_Q_PLUS", "True").lower() == "true"

    # Direction-split ablation for the EMNLP rebuttal. Selects which Q-/Q+
    # channels Stage 1 of retrieve.py queries and whether Stage 2 fires.
    # Values:
    #   "full"            -> paper default (Q- 0.7 + body 0.3, followed by
    #                        Q+ 0.6 + Q- support 0.4 on every query).
    #   "qminus_only"     -> Stage 1: Q- 1.0, no body. Stage 2 disabled.
    #   "qplus_only"      -> Stage 1: Q+ 1.0, no body. Stage 2 disabled.
    #   "single_combined" -> Stage 1: Q- 0.5 + Q+ 0.5 (HopRAG-style single
    #                        hypothetical channel). Stage 2 disabled.
    # No re-indexing required; only retrieval-time channel selection changes.
    HYPO_CHANNEL_VARIANT = os.environ.get("RAG_HYPO_CHANNEL_VARIANT", "full").strip().lower() or "full"

    @classmethod
    def validate(cls) -> None:
        positive = {
            "RETRY_COUNT": cls.RETRY_COUNT,
            "MAX_CONCURRENT_LLM_CALLS": cls.MAX_CONCURRENT_LLM_CALLS,
            "MAX_CONCURRENT_EMBEDDING_REQUESTS": cls.MAX_CONCURRENT_EMBEDDING_REQUESTS,
            "EMBEDDING_BATCH_SIZE": cls.EMBEDDING_BATCH_SIZE,
            "VLLM_MAX_NUM_SEQS": cls.VLLM_MAX_NUM_SEQS,
            "EMBEDDING_DIMENSIONS": cls.EMBEDDING_DIMENSIONS,
            "NEO4J_BATCH_SIZE": cls.NEO4J_BATCH_SIZE,
            "RRF_K_CONSTANT": cls.RRF_K_CONSTANT,
            "VECTOR_SEARCH_LIMIT": cls.VECTOR_SEARCH_LIMIT,
            "TEXT_SEARCH_LIMIT": cls.TEXT_SEARCH_LIMIT,
            "CHUNK_SENTENCES": cls.CHUNK_SENTENCES,
            "MIN_CHUNK_SENTENCES": cls.MIN_CHUNK_SENTENCES,
            "HOP_LINK_LIMIT": cls.HOP_LINK_LIMIT,
            "HOP_CANDIDATE_LIMIT": cls.HOP_CANDIDATE_LIMIT,
            "HOP_ANN_POOL": cls.HOP_ANN_POOL,
            "HOP_GATHER_WAVE": cls.HOP_GATHER_WAVE,
            "HOP_CHANNEL_CONCURRENCY": cls.HOP_CHANNEL_CONCURRENCY,
            "GRAPH_SEARCH_LIMIT": cls.GRAPH_SEARCH_LIMIT,
            "DEFAULT_TOP_K": cls.DEFAULT_TOP_K,
        }
        invalid = {name: value for name, value in positive.items() if value < 1}
        if invalid:
            raise ValueError(f"RAG configuration values must be positive: {invalid}")
        if cls.EMBEDDING_BATCH_SIZE * cls.MAX_CONCURRENT_EMBEDDING_REQUESTS > cls.VLLM_MAX_NUM_SEQS:
            raise ValueError(
                "Embedding client can exceed VLLM_MAX_NUM_SEQS: "
                f"batch={cls.EMBEDDING_BATCH_SIZE} * concurrent_requests="
                f"{cls.MAX_CONCURRENT_EMBEDDING_REQUESTS} > {cls.VLLM_MAX_NUM_SEQS}"
            )
        if cls.MAX_CONCURRENT_LLM_CALLS > cls.VLLM_MAX_NUM_SEQS:
            raise ValueError(
                "Generation client can exceed VLLM_MAX_NUM_SEQS: "
                f"concurrent_calls={cls.MAX_CONCURRENT_LLM_CALLS} > {cls.VLLM_MAX_NUM_SEQS}"
            )
        if cls.HOP_SAME_NEED_WEIGHT < 0:
            raise ValueError("RAG_HOP_SAME_NEED_WEIGHT must be non-negative")
        if not (0.0 < cls.MAX_CHUNKS_PER_SOURCE_FRACTION <= 1.0):
            raise ValueError("RAG_MAX_CHUNKS_PER_SOURCE_FRACTION must be in (0, 1]")
        if cls.GRAPH_HOP_DEPTH < 0 or cls.GRAPH_HOP_DEPTH > 4:
            raise ValueError("RAG_GRAPH_HOP_DEPTH must be between 0 and 4")

        allowed_variants = {"full", "qminus_only", "qplus_only", "single_combined"}
        if cls.HYPO_CHANNEL_VARIANT not in allowed_variants:
            raise ValueError(
                f"RAG_HYPO_CHANNEL_VARIANT={cls.HYPO_CHANNEL_VARIANT!r} is invalid; "
                f"expected one of {sorted(allowed_variants)}"
            )
        if cls.HYPO_CHANNEL_VARIANT == "qminus_only" and not cls.ABLATION_Q_MINUS:
            raise ValueError("qminus_only requires RAG_ABLATION_Q_MINUS=true")
        if cls.HYPO_CHANNEL_VARIANT == "qplus_only" and not cls.ABLATION_Q_PLUS:
            raise ValueError("qplus_only requires RAG_ABLATION_Q_PLUS=true")
        if cls.HYPO_CHANNEL_VARIANT == "single_combined" and not (
            cls.ABLATION_Q_MINUS and cls.ABLATION_Q_PLUS
        ):
            raise ValueError("single_combined requires both Q- and Q+ channels")
