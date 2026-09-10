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
from core.generation_profiles import request_settings
from core.structured_outputs import question_contract
from models.prehop.llm_json import generate_json_or_raise
from models.prehop.tracing import traced
from utils.prompts import (
    GROUNDED_HOPRAG_FORMAT_INSTRUCTION,
    GROUNDED_HOPRAG_PROMPT,
    HOPRAG_FORMAT_INSTRUCTION,
    HOPRAG_PROMPT,
    LINKED_HOPRAG_FORMAT_INSTRUCTION,
    LINKED_HOPRAG_PROMPT,
)

logger = logging.getLogger(__name__)

_SOURCE_RELATIVE_RE = re.compile(r"\b(?:the\s+)?(?:provided text|given text|this chunk|the passage)\b", re.IGNORECASE)


def _question_identity(question: str) -> str:
    return " ".join(question.casefold().split())


def _grounding_identity(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


class KnowledgeMappingMixin:
    @staticmethod
    def _question_items(value: list[Any], channel: str, title: str) -> list[str]:
        questions: list[str] = []
        seen: set[str] = set()
        for index, item in enumerate(value):
            question = item.strip()
            identity = _question_identity(question)
            if identity in seen or _SOURCE_RELATIVE_RE.search(question):
                continue
            seen.add(identity)
            questions.append(question)
        return questions

    @staticmethod
    def _grounded_items(
        value: list[Any],
        channel: str,
        chunk: str,
        title: str,
        question_schema: str = "grounded_v1",
    ) -> list[dict[str, Any]]:
        chunk_identity = _grounding_identity(chunk)
        questions: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, item in enumerate(value):
            question = item.get("question")
            grounding_quote = item.get("grounding_quote")
            anchor_entities = item.get("anchor_entities")
            question = " ".join(question.split())
            if _SOURCE_RELATIVE_RE.search(question):
                continue
            identity = _question_identity(question)
            if identity in seen:
                continue
            grounding_quote = " ".join(grounding_quote.split())
            _grounding_identity(grounding_quote)
            relaxed_anchors = question_schema == "linked_v2"
            if not isinstance(anchor_entities, list):
                if relaxed_anchors:
                    anchor_entities = []
                else:
                    raise ValueError(f"{channel} grounded item at index={index} has invalid anchor_entities")
            anchors: list[str] = []
            anchor_seen: set[str] = set()
            for raw_anchor in anchor_entities:
                if not isinstance(raw_anchor, str) or not raw_anchor.strip():
                    if relaxed_anchors:
                        continue
                    raise ValueError(f"{channel} grounded item at index={index} has invalid anchor_entities")
                anchor = " ".join(raw_anchor.split())
                anchor_identity = _grounding_identity(anchor)
                if anchor_identity not in chunk_identity:
                    if relaxed_anchors:
                        continue
                    raise ValueError(f"{channel} anchor_entity is not present in source chunk for title={title!r}")
                if anchor_identity not in anchor_seen:
                    anchor_seen.add(anchor_identity)
                    anchors.append(anchor)

            record: dict[str, Any] = {
                "text": question,
                "grounding_quote": grounding_quote,
                "anchor_entities": anchors,
                "question_schema": question_schema,
            }
            if channel == "Q-":
                answer = item.get("answer")
                answer = " ".join(answer.split())
                record["answer"] = answer
                if question_schema == "linked_v2":
                    continuation_anchor = item.get("continuation_anchor")
                    continuation_anchor = " ".join(continuation_anchor.split())
                    if continuation_anchor and _grounding_identity(continuation_anchor) != _grounding_identity(answer):
                        # The direct Q- record remains valid even when the
                        # optional continuation marker is over-specific or a
                        # partial name. Preserve retrieval evidence and simply
                        # decline to construct an edge from that marker.
                        continuation_anchor = ""
                    record["continuation_anchor"] = continuation_anchor
            else:
                missing_information = item.get("missing_information")
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
        question_schema: str = "grounded_v1",
    ) -> list[dict[str, Any]]:
        """Keep valid grounded records without discarding their whole document."""
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, item in enumerate(value):
            try:
                validated = cls._grounded_items([item], channel, chunk, title, question_schema=question_schema)
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

    @traced
    async def extract_hoprag_queries(self, chunk: str, title: str = "") -> dict[str, Any]:
        """Generate Q-/Q+ for a chunk without rolling context.

        A structurally invalid response (valid JSON, but missing/malformed
        q_minus or q_plus) is retried the same bounded number of
        times as other transient failures in this codebase, rather than
        failing the whole document on one flaky response. This is retrying
        the identical call until it validates, not a content-quality filter
        -- it does not change what a valid response looks like.
        """
        if RAGConfig.HOP_LINK_VARIANT == "body":
            return {"q_minus": [], "q_plus": []}
        question_schema = RAGConfig.QUESTION_SCHEMA
        grounded = question_schema in {"grounded_v1", "linked_v2"}
        if question_schema == "linked_v2":
            prompt = LINKED_HOPRAG_PROMPT
            format_instruction = LINKED_HOPRAG_FORMAT_INSTRUCTION
        elif grounded:
            prompt = GROUNDED_HOPRAG_PROMPT
            format_instruction = GROUNDED_HOPRAG_FORMAT_INSTRUCTION
        else:
            prompt = HOPRAG_PROMPT
            format_instruction = HOPRAG_FORMAT_INSTRUCTION
        text_prompt = prompt.format(chunk=chunk, global_context=f"Document Title: {title}")
        messages = [
            {"role": "user", "content": text_prompt},
            {"role": "user", "content": format_instruction.format()},
        ]
        last_error: Exception | None = None
        for attempt in range(1, RAGConfig.RETRY_COUNT + 1):
            try:
                data = await generate_json_or_raise(
                    self.indexing_llm,
                    messages,
                    "Q-/Q+ generation",
                    f"title={title!r}",
                    structured_contract=question_contract("index", question_schema, RAGConfig.QUESTIONS_PER_DIRECTION),
                    **request_settings("question_index"),
                )
                if grounded:
                    q_minus = self._filter_grounded_items(
                        data["q_minus"], "Q-", chunk, title, question_schema=question_schema
                    )
                    q_minus_identities = {_question_identity(item["text"]) for item in q_minus}
                    q_plus = [
                        item
                        for item in self._filter_grounded_items(
                            data["q_plus"], "Q+", chunk, title, question_schema=question_schema
                        )
                        if _question_identity(item["text"]) not in q_minus_identities
                    ]
                else:
                    q_minus = self._question_items(data["q_minus"], "Q-", title)
                    q_minus_identities = {_question_identity(question) for question in q_minus}
                    q_plus = [
                        question
                        for question in self._question_items(data["q_plus"], "Q+", title)
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
