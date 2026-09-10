"""Common operational embedding controls, separate from semantic model config."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingOperationalConfig:
    batch_size: int
    concurrency: int
    retry_attempts: int

    @classmethod
    def resolve(cls, strategy: str) -> EmbeddingOperationalConfig:
        from core.strategy_registry import EXTERNAL_STRATEGIES

        # In-repo clients currently consume the canonical global controls.
        # Isolated external workers receive normalized strategy overrides in
        # their child environment, so recorded and effective values agree.
        prefix = f"RAG_{strategy.upper()}_EMBEDDING_" if strategy in EXTERNAL_STRATEGIES else "__NO_STRATEGY_OVERRIDE__"
        values = cls(
            batch_size=int(
                os.environ.get(
                    prefix + "BATCH_SIZE", os.environ.get("RAG_EMBEDDING_BATCH_SIZE", "16")
                )
            ),
            concurrency=int(
                os.environ.get(
                    prefix + "CONCURRENCY",
                    os.environ.get(
                        "RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS", "1"
                    ),
                )
            ),
            retry_attempts=int(
                os.environ.get(prefix + "RETRY_ATTEMPTS", os.environ.get("RAG_INFERENCE_RETRY_ATTEMPTS", "5"))
            ),
        )
        return values

    def as_dict(self) -> dict[str, int]:
        return {
            "embedding_batch_size": self.batch_size,
            "embedding_concurrency": self.concurrency,
            "embedding_retry_attempts": self.retry_attempts,
        }
