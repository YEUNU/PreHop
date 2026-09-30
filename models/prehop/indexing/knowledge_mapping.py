"""Generate incoming and outgoing hypothetical questions for each chunk.

Each chunk is annotated at indexing time with dual hypothetical queries:
- Q- (incoming): self-contained questions answerable from the chunk alone.
- Q+ (outgoing): questions the chunk only partially answers, pointing to its
  dependencies and later seed cross-document evidence-edge construction.
"""

import logging
import re
from typing import Any

from core.config import RAGConfig
from core.generation_profiles import request_settings
from core.structured_outputs import question_contract
from models.prehop.tracing import traced
from utils.prompts import (
    HOPRAG_FORMAT_INSTRUCTION,
    HOPRAG_PROMPT,
)

logger = logging.getLogger(__name__)

_SOURCE_RELATIVE_RE = re.compile(r"\b(?:the\s+)?(?:provided text|given text|this chunk|the passage)\b", re.IGNORECASE)


def _question_identity(question: str) -> str:
    return " ".join(question.casefold().split())


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


    @traced
    async def extract_hoprag_queries(self, chunk: str, title: str = "") -> dict[str, Any]:
        """Generate questions once; the client owns transport and JSON retries."""
        question_schema = RAGConfig.QUESTION_SCHEMA
        prompt = HOPRAG_PROMPT
        format_instruction = HOPRAG_FORMAT_INSTRUCTION
        text_prompt = prompt.format(chunk=chunk, global_context=f"Document Title: {title}")
        messages = [
            {"role": "user", "content": text_prompt},
            {"role": "user", "content": format_instruction.format()},
        ]
        data = await self.indexing_llm.generate_json(
            messages,
            json_debug_label="Q-/Q+ generation",
            structured_contract=question_contract(question_schema, RAGConfig.QUESTIONS_PER_DIRECTION),
            **request_settings("question_index"),
        )
        q_minus = self._question_items(data["q_minus"], "Q-", title)
        q_minus_identities = {_question_identity(question) for question in q_minus}
        q_plus = [
            question
            for question in self._question_items(data["q_plus"], "Q+", title)
            if _question_identity(question) not in q_minus_identities
        ]
        return {"q_minus": q_minus, "q_plus": q_plus}
