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
        return {
            "q_minus": data.get("q_minus", []),
            "q_plus": data.get("q_plus", []),
            "summary": data.get("summary", ""),
        }
