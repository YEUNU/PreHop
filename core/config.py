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
    MAX_CONTEXT_LENGTH = int(os.environ.get("RAG_MAX_CONTEXT_LENGTH", "16384"))
    # Capped below the configured context limit so input has output headroom.
    # Indexing prompts for Q-/Q+ generation rarely exceed 1–2K output;
    # 4K is comfortable headroom.
    MAX_OUTPUT_TOKENS = 4096
    # The benchmark contract requests a short final answer, not free-form
    # generation. This fixed cap reserves context space and bounds latency; it
    # is shared by controlled answer-synthesis paths and is not swept.
    SYNTHESIS_MAX_OUTPUT_TOKENS = 128
    MAX_EMBEDDING_LENGTH = int(os.environ.get("MAX_EMBEDDING_LENGTH", "16384"))

    # --- RAG & Indexing Settings ---
    MAX_CONCURRENT_LLM_CALLS = int(os.environ.get("MAX_CONCURRENT_LLM_CALLS", "30"))
    MAX_CONCURRENT_EMBEDDING_REQUESTS = int(os.environ.get("RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS", "2"))
    EMBEDDING_BATCH_SIZE = int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE", "32"))
    VLLM_MAX_NUM_SEQS = int(os.environ.get("VLLM_MAX_NUM_SEQS", "120"))
    EMBEDDING_DIMENSIONS = int(os.environ.get("NEO4J_VECTOR_DIMENSIONS", "1024"))
    NEO4J_BATCH_SIZE = int(os.environ.get("NEO4J_BATCH_SIZE", "25"))

    # --- Search & Ranking ---
    # Query-time channels use unweighted reciprocal rank, 1 / (rank + 1).
    # There is no dataset-tuned fusion constant or modality preference.
    # --- Indexing Pipeline Settings ---
    # Fixed-size chunking (core-only rewrite — replaces adaptive/embedding-
    # similarity chunk splitting). Each page is windowed into chunks of
    # CHUNK_SENTENCES sentences, including the final partial window.
    CHUNK_SENTENCES = 6
    QUESTIONS_PER_DIRECTION = 3
    # HOP ANN sends high-dimensional vectors and candidate rows through bounded waves.
    HOP_GATHER_WAVE = int(os.environ.get("RAG_HOP_GATHER_WAVE", "64"))
    DEFAULT_TOP_K = 12
    FULLTEXT_ANALYZER = os.environ.get("NEO4J_FULLTEXT_ANALYZER", "english")

    # Zero disables graph expansion for ablation; one enables the fixed
    # bidirectional NEXT and outgoing HOP_ANSWER expansion.
    GRAPH_HOP_DEPTH = int(os.environ.get("RAG_GRAPH_HOP_DEPTH", "1"))

    # --- Ablation & Experimental Toggles ---
    # Q-/Q+ channel ablations.
    # ABLATION_Q_MINUS / ABLATION_Q_PLUS gate whether the Q-/Q+ channels
    # participate in indexing (embedding storage) and retrieval (channel use).
    # Disabling Q+ also disables offline HOP edge construction, since HOP
    # selection is anchored on Q+ embeddings.
    ABLATION_Q_MINUS = os.environ.get("RAG_ABLATION_Q_MINUS", "True").lower() == "true"
    ABLATION_Q_PLUS = os.environ.get("RAG_ABLATION_Q_PLUS", "True").lower() == "true"

    # Select which Q-/Q+ representation channels retrieve.py queries.
    # Values:
    #   "full"            -> Q-/body direct evidence plus Q+
    #                        dependency seeds in one set union.
    #   "qminus_only"     -> Q- only, direct evidence role.
    #   "qplus_only"      -> Q+ only, dependency-seed role.
    #   "single_combined" -> Q- and Q+ queried once and combined by set union,
    #                        with no body channel.
    # No re-indexing required; only retrieval-time channel selection changes.
    HYPO_CHANNEL_VARIANT = os.environ.get("RAG_HYPO_CHANNEL_VARIANT", "full").strip().lower() or "full"
    SOURCE_SELECTION_VARIANT = (
        os.environ.get("RAG_SOURCE_SELECTION_VARIANT", "round_robin").strip().lower() or "round_robin"
    )

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
            "CHUNK_SENTENCES": cls.CHUNK_SENTENCES,
            "QUESTIONS_PER_DIRECTION": cls.QUESTIONS_PER_DIRECTION,
            "HOP_GATHER_WAVE": cls.HOP_GATHER_WAVE,
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
        if cls.GRAPH_HOP_DEPTH not in {0, 1}:
            raise ValueError("RAG_GRAPH_HOP_DEPTH must be 0 or 1")

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
        if cls.SOURCE_SELECTION_VARIANT not in {"round_robin", "global"}:
            raise ValueError("RAG_SOURCE_SELECTION_VARIANT must be round_robin or global")
