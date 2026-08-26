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
import unicodedata
from typing import Any

from core.config import RAGConfig
from models.prehop.llm_json import generate_json_or_raise
from utils.prompts import (
    GROUNDED_HOPRAG_FORMAT_INSTRUCTION,
    GROUNDED_HOPRAG_PROMPT,
    HOPRAG_FORMAT_INSTRUCTION,
    HOPRAG_PROMPT,
)

logger = logging.getLogger(__name__)

_SOURCE_RELATIVE_RE = re.compile(r"\b(?:the\s+)?(?:provided text|given text|this chunk|the passage)\b", re.IGNORECASE)


def _question_identity(question: str) -> str:
    return " ".join(question.casefold().split())


def _grounding_identity(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


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

    @staticmethod
    def _validate_grounded_items(
        value: list[Any],
        channel: str,
        chunk: str,
        title: str,
    ) -> list[dict[str, Any]]:
        if len(value) > RAGConfig.QUESTIONS_PER_DIRECTION:
            raise ValueError(
                f"{channel} generation returned more than "
                f"{RAGConfig.QUESTIONS_PER_DIRECTION} questions for title={title!r}"
            )
        chunk_identity = _grounding_identity(chunk)
        questions: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise TypeError(f"{channel} grounded item at index={index} must be an object")
            question = item.get("question")
            grounding_quote = item.get("grounding_quote")
            anchor_entities = item.get("anchor_entities")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{channel} grounded item at index={index} has no question")
            question = " ".join(question.split())
            if _SOURCE_RELATIVE_RE.search(question):
                continue
            identity = _question_identity(question)
            if identity in seen:
                continue
            if not isinstance(grounding_quote, str) or not grounding_quote.strip():
                raise ValueError(f"{channel} grounded item at index={index} has no grounding_quote")
            grounding_quote = " ".join(grounding_quote.split())
            quote_identity = _grounding_identity(grounding_quote)
            if quote_identity not in chunk_identity:
                raise ValueError(
                    f"{channel} grounding_quote is not present in source chunk for title={title!r}"
                )
            if (
                not isinstance(anchor_entities, list)
                or not anchor_entities
                or any(not isinstance(entity, str) or not entity.strip() for entity in anchor_entities)
            ):
                raise ValueError(f"{channel} grounded item at index={index} has invalid anchor_entities")
            anchors: list[str] = []
            anchor_seen: set[str] = set()
            for raw_anchor in anchor_entities:
                anchor = " ".join(raw_anchor.split())
                anchor_identity = _grounding_identity(anchor)
                if anchor_identity not in chunk_identity:
                    raise ValueError(
                        f"{channel} anchor_entity is not present in source chunk for title={title!r}"
                    )
                if anchor_identity not in anchor_seen:
                    anchor_seen.add(anchor_identity)
                    anchors.append(anchor)

            record: dict[str, Any] = {
                "text": question,
                "grounding_quote": grounding_quote,
                "anchor_entities": anchors,
                "question_schema": "grounded_v1",
            }
            if channel == "Q-":
                answer = item.get("answer")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError(f"Q- grounded item at index={index} has no answer")
                answer = " ".join(answer.split())
                if _grounding_identity(answer) not in quote_identity:
                    raise ValueError(f"Q- answer is not present in grounding_quote for title={title!r}")
                record["answer"] = answer
            else:
                missing_information = item.get("missing_information")
                if not isinstance(missing_information, str) or not missing_information.strip():
                    raise ValueError(f"Q+ grounded item at index={index} has no missing_information")
                record["missing_information"] = " ".join(missing_information.split())
            seen.add(identity)
            questions.append(record)
        return questions

    @classmethod
    def _filter_grounded_items(
        cls,
        value: list[Any],
        channel: str,
        chunk: str,
        title: str,
    ) -> list[dict[str, Any]]:
        """Keep valid grounded records without discarding their whole document."""
        if len(value) > RAGConfig.QUESTIONS_PER_DIRECTION:
            raise ValueError(
                f"{channel} generation returned more than "
                f"{RAGConfig.QUESTIONS_PER_DIRECTION} questions for title={title!r}"
            )
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, item in enumerate(value):
            try:
                validated = cls._validate_grounded_items([item], channel, chunk, title)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Dropping unverifiable %s record index=%d for title=%r: %s",
                    channel,
                    index,
                    title,
                    exc,
                )
                continue
            if not validated:
                continue
            record = validated[0]
            identity = _question_identity(record["text"])
            if identity not in seen:
                seen.add(identity)
                records.append(record)
        return records

    async def extract_hoprag_queries(self, chunk: str, title: str = "") -> dict[str, Any]:
        """Generate Q-/Q+ for a chunk without rolling context.

        A structurally invalid response (valid JSON, but missing/malformed
        q_minus or q_plus) is retried the same bounded number of
        times as other transient failures in this codebase, rather than
        failing the whole document on one flaky response. This is retrying
        the identical call until it validates, not a content-quality filter
        -- it does not change what a valid response looks like.
        """
        grounded = RAGConfig.QUESTION_SCHEMA == "grounded_v1"
        prompt = GROUNDED_HOPRAG_PROMPT if grounded else HOPRAG_PROMPT
        format_instruction = GROUNDED_HOPRAG_FORMAT_INSTRUCTION if grounded else HOPRAG_FORMAT_INSTRUCTION
        text_prompt = prompt.format(chunk=chunk, global_context=f"Document Title: {title}")
        messages = [
            {"role": "user", "content": text_prompt},
            {"role": "user", "content": format_instruction},
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
                if grounded:
                    q_minus = self._filter_grounded_items(data["q_minus"], "Q-", chunk, title)
                    q_minus_identities = {_question_identity(item["text"]) for item in q_minus}
                    q_plus = [
                        item
                        for item in self._filter_grounded_items(data["q_plus"], "Q+", chunk, title)
                        if _question_identity(item["text"]) not in q_minus_identities
                    ]
                else:
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
