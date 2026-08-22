"""Predictive Knowledge Mapping (paper §3.1.3).

Each chunk is annotated at indexing time with dual hypothetical queries:
- Q- (incoming): self-contained questions answerable from the chunk alone.
- Q+ (outgoing): questions the chunk only partially answers, pointing to its
  dependencies; later used as the ANN seed for HOP edge construction (§3.1.4).
"""

from typing import Any

from models.prehop.llm_json import generate_json_or_raise
from utils.prompts import HOPRAG_FORMAT_INSTRUCTION, HOPRAG_PROMPT


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
        """Generate Q-/Q+ for a chunk without rolling context."""
        text_prompt = HOPRAG_PROMPT.format(chunk=chunk, global_context=f"Document Title: {title}")
        messages = [
            {"role": "user", "content": text_prompt},
            {"role": "user", "content": HOPRAG_FORMAT_INSTRUCTION},
        ]
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
