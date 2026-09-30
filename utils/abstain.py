"""Shared abstention-phrase detection for deterministic answer metrics."""

from __future__ import annotations

# Apply these phrases to the extracted final answer, not its rationale.
ABSTAIN_PHRASES: tuple[str, ...] = (
    "insufficient evidence",
    # MultiHop-RAG's null_query gold answer is literally "Insufficient
    # information." so an honest abstain must match it for correct Refusal
    # labeling on that dataset.
    "insufficient information",
    "i do not know",
    "i don't know",
    "do not know",
    "cannot be determined",
    "cannot determine",
    "unable to determine",
    "not determinable",
    "not provided in the context",
    "not specified in the context",
    "not stated in the context",
    "not mentioned in the context",
    "no information",
    "the context does not contain",
    "the context does not mention",
    "the context does not include",
    "the context does not provide",
    "context provides no",
    "context lacks",
    "unable to find relevant information",
    "unable to find",
    "no relevant information",
)


def is_abstain(text) -> bool:
    """True if `text` contains any recognized abstain phrase.

    Matches lowercased substring; safe to call on the FULL response or on
    an extracted-final-answer slice. CoT responses that include an abstain
    token mid-reasoning but conclude with a substantive answer should be
    checked against the extracted final answer (see
    `utils/metrics.py::extract_final_answer`), not the full text.
    """
    return any(p in str(text or "").lower() for p in ABSTAIN_PHRASES)
