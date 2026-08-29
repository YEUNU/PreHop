"""Dataset-neutral query rewriting for role-aligned Prehop retrieval."""


def build_role_aligned_query_prompt(query: str, max_per_role: int) -> str:
    """Request retrieval queries matching the indexed Q-/Q+ semantics."""
    return f"""Transform the user question into role-aligned retrieval questions. Do not answer it.

Definitions:
- q_minus: self-contained atomic questions that one evidence passage can answer directly.
- q_plus: dependency or relational questions that connect known entities, events, dates, or facts to a missing counterpart in another passage.

Rules:
1. Preserve all named entities, source distinctions, dates, comparison direction, temporal order, and negation from the user question.
2. Do not introduce facts or assumptions absent from the user question.
3. Produce up to {max_per_role} non-duplicate questions per role. Use an empty list when no faithful question exists for that role; the original user question remains available as the retrieval fallback.
4. Make every question stand alone; do not refer to "the user question" or "the text".
5. Return JSON only, using exactly this schema:
{{"q_minus": ["question"], "q_plus": ["question"]}}

USER QUESTION:
{query}
"""
