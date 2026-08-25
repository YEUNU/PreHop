"""Generate incoming and outgoing hypothetical questions for each chunk.

Each chunk is annotated at indexing time with dual hypothetical queries:
- Q- (incoming): self-contained questions answerable from the chunk alone.
- Q+ (outgoing): questions the chunk only partially answers, pointing to its
  dependencies and later seed cross-document evidence-edge construction.
"""

import asyncio
import logging
import random
import re
from typing import Any

from core.config import RAGConfig
from models.prehop.llm_json import generate_json_or_raise
from utils.prompts import HOPRAG_FORMAT_INSTRUCTION, HOPRAG_PROMPT

logger = logging.getLogger(__name__)

_SOURCE_RELATIVE_RE = re.compile(r"\b(?:the\s+)?(?:provided text|given text|this chunk|the passage)\b", re.IGNORECASE)


def _question_identity(question: str) -> str:
    return " ".join(question.casefold().split())


class KnowledgeMappingMixin:
    @staticmethod
    def _validate_question_items(value: list[Any], channel: str, title: str) -> list[str]:
        if len(value) > RAGConfig.QUESTIONS_PER_DIRECTION:
            raise ValueError(
                f"{channel} generation returned more than "
                f"{RAGConfig.QUESTIONS_PER_DIRECTION} questions for title={title!r}"
            )
        questions: list[str] = []
        seen: set[str] = set()
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"{channel} generation returned a non-string/blank item "
                    f"at index={index} for title={title!r}"
                )
            question = item.strip()
            identity = _question_identity(question)
            if identity in seen or _SOURCE_RELATIVE_RE.search(question):
                continue
            seen.add(identity)
            questions.append(question)
        return questions

    async def extract_hoprag_queries(self, chunk: str, title: str = "") -> dict[str, Any]:
        """Generate Q-/Q+ for a chunk without rolling context.

        A structurally invalid response (valid JSON, but missing/malformed
        q_minus or q_plus) is retried the same bounded number of
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
                    required_fields={"q_minus": list, "q_plus": list},
                    temperature=0.0,
                )
                q_minus = self._validate_question_items(data["q_minus"], "Q-", title)
                q_minus_identities = {_question_identity(question) for question in q_minus}
                q_plus = [
                    question
                    for question in self._validate_question_items(data["q_plus"], "Q+", title)
                    if _question_identity(question) not in q_minus_identities
                ]
                return {"q_minus": q_minus, "q_plus": q_plus}
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
