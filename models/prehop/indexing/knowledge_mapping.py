"""Predictive Knowledge Mapping (paper §3.1.3).

Each chunk is annotated at indexing time with dual hypothetical queries:
- Q- (incoming): self-contained questions answerable from the chunk alone.
- Q+ (outgoing): questions the chunk only partially answers, pointing to its
  dependencies; later used as the ANN seed for HOP edge construction (§3.1.4).
"""

import asyncio
import logging
import random
from typing import Any

from core.config import RAGConfig
from models.prehop.llm_json import generate_json_or_raise
from utils.prompts import HOPRAG_FORMAT_INSTRUCTION, HOPRAG_PROMPT

logger = logging.getLogger(__name__)


class KnowledgeMappingMixin:
    @staticmethod
    def _validate_question_items(value: list[Any], channel: str, title: str) -> list[str]:
        if len(value) > 3:
            raise ValueError(f"{channel} generation returned more than 3 questions for title={title!r}")
        questions: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"{channel} generation returned a non-string/blank item "
                    f"at index={index} for title={title!r}"
                )
            questions.append(item.strip())
        return questions

    async def extract_hoprag_queries(self, chunk: str, title: str = "") -> dict[str, Any]:
        """Generate Q-/Q+ for a chunk without rolling context.

        A structurally invalid response (valid JSON, but missing/malformed
        q_minus, q_plus, or summary) is retried the same bounded number of
        times as other transient failures in this codebase, rather than
        failing the whole document on one flaky response. This is retrying
        the identical call until it validates, not a content-quality filter
        -- it does not change what a valid response looks like.
        """
        text_prompt = HOPRAG_PROMPT.format(chunk=chunk, global_context=f"Document Title: {title}")
        messages = [
            {"role": "user", "content": text_prompt},
            {"role": "user", "content": HOPRAG_FORMAT_INSTRUCTION},
        ]
        last_error: Exception | None = None
        for attempt in range(1, RAGConfig.RETRY_COUNT + 1):
            try:
                data = await generate_json_or_raise(
                    self.indexing_llm,
                    messages,
                    "Q-/Q+ generation",
                    f"title={title!r}",
                    required_fields={"q_minus": list, "q_plus": list, "summary": str},
                    apply_default_sampling=False,
                )
                summary = data["summary"].strip()
                if not summary:
                    raise ValueError(f"Q-/Q+ generation returned a blank summary for title={title!r}")
                return {
                    "q_minus": self._validate_question_items(data["q_minus"], "Q-", title),
                    "q_plus": self._validate_question_items(data["q_plus"], "Q+", title),
                    "summary": summary,
                }
            except (ValueError, TypeError) as exc:
                last_error = exc
                if attempt < RAGConfig.RETRY_COUNT:
                    logger.warning(
                        "Q-/Q+ generation validation failed (%d/%d) for title=%r: %s; retrying",
                        attempt,
                        RAGConfig.RETRY_COUNT,
                        title,
                        exc,
                    )
                    delay = (RAGConfig.RETRY_DELAY * (2 ** (attempt - 1))) + random.uniform(0, RAGConfig.RETRY_DELAY)
                    await asyncio.sleep(delay)
        raise last_error
