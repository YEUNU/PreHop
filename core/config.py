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
    EMBEDDING_BATCH_SIZE = int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE", "128"))
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

    # --- Offline graph construction & traversal ---
    HOP_THRESHOLD = float(os.environ.get("RAG_HOP_THRESHOLD", "0.82"))

    # --- Indexing Pipeline Settings ---
    # Fixed-size chunking (core-only rewrite — replaces adaptive/embedding-
    # similarity chunk splitting). Each page is windowed into chunks of
    # CHUNK_SENTENCES sentences; a trailing window shorter than
    # MIN_CHUNK_SENTENCES merges into the previous chunk instead of standing
    # alone.
    CHUNK_SENTENCES = int(os.environ.get("RAG_CHUNK_SENTENCES", "6"))
    MIN_CHUNK_SENTENCES = int(os.environ.get("RAG_MIN_CHUNK_SENTENCES", "2"))
    HOP_LINK_LIMIT = int(os.environ.get("RAG_HOP_LINK_LIMIT", "5"))
    GRAPH_SEARCH_LIMIT = int(os.environ.get("RAG_GRAPH_SEARCH_LIMIT", "10"))
    DEFAULT_TOP_K = int(os.environ.get("RAG_DEFAULT_TOP_K", "12"))
    FULLTEXT_ANALYZER = os.environ.get("NEO4J_FULLTEXT_ANALYZER", "english")

    # Graph traversal depth on the query path. depth=0 = pure `retrieve()`
    # (Stage 1+2 RRF + similarity ordering, no graph expansion) for ablation; depth>0 uses
    # `graph_search` — deterministic traversal over the
    # NEXT/HOP edges built during indexing (paper §3.1.4), no LLM continuation
    # check. depth=1 is the default (1-hop NEXT|HOP).
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
