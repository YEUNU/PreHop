"""Stage-aware JSON schema guard for Prehop indexing/retrieval LLM calls.

The shared VLLM client raises after JSON parse retries and propagates transport
errors. This wrapper adds the required-key/type checks specific to each Prehop
stage so a syntactically valid but structurally wrong object cannot silently
turn into empty Q-/Q+, rewrites, or decisions.
"""

from typing import Any


async def generate_json_or_raise(
    llm_client,
    messages,
    stage: str,
    context: str = "",
    required_fields: dict[str, type] | None = None,
    **kwargs,
) -> dict[str, Any]:
    kwargs.setdefault("json_debug_label", stage)
    data = await llm_client.generate_json(messages, **kwargs)
    if not isinstance(data, dict) or not data:
        suffix = f": {context}" if context else ""
        raise ValueError(f"{stage} returned no valid JSON after retries{suffix}")
    for field, expected_type in (required_fields or {}).items():
        if field not in data:
            suffix = f": {context}" if context else ""
            raise ValueError(f"{stage} JSON missing required field {field!r}{suffix}")
        if not isinstance(data[field], expected_type):
            suffix = f": {context}" if context else ""
            actual = type(data[field]).__name__
            raise TypeError(f"{stage} JSON field {field!r} must be {expected_type.__name__}, got {actual}{suffix}")
    return data
