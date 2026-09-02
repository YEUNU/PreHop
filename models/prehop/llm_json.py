"""Stage-aware JSON schema guard for Prehop indexing/retrieval LLM calls.

The shared VLLM client raises after JSON parse retries and propagates transport
errors. This wrapper adds the required-key/type checks specific to each Prehop
stage so a syntactically valid but structurally wrong object cannot silently
turn into empty Q-/Q+, rewrites, or decisions.
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _extract_json_payload(raw: str) -> dict[str, Any] | None:
    """Try to recover a JSON object from noisy model output."""
    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1)

    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _format_retry_hint(stage: str, context: str, attempt: int, reason: str, required_fields: dict[str, type] | None = None) -> str:
    if required_fields:
        req = ", ".join(sorted(required_fields.keys()))
        req_hint = f"Required fields: {req}. "
    else:
        req_hint = ""
    suffix = f" | context={context}" if context else ""
    return (
        f"Retry {attempt + 1} for {stage}. Previous output was invalid ({reason}). "
        f"Return ONLY one raw JSON object with double quotes, no markdown, no prose.{req_hint}"
        f"If it cannot be parsed, retry by outputting a valid JSON object only."
        f"{suffix}"
    )


async def generate_json_or_raise(
    llm_client,
    messages,
    stage: str,
    context: str = "",
    required_fields: dict[str, type] | None = None,
    **kwargs,
) -> dict[str, Any]:
    kwargs.setdefault("json_debug_label", stage)
    max_retries = int(kwargs.pop("max_parse_retries", 2))

    def _validate(candidate: Any, reason: str) -> tuple[dict[str, Any] | None, str]:
        if isinstance(candidate, dict):
            if not candidate:
                return None, "empty JSON object"
            for field, expected_type in (required_fields or {}).items():
                if field not in candidate:
                    return None, f"missing required field {field!r}"
                if not isinstance(candidate[field], expected_type):
                    actual = type(candidate[field]).__name__
                    return None, f"field {field!r} expected {expected_type.__name__}, got {actual}"
            return candidate, ""
        if isinstance(candidate, str):
            extracted = _extract_json_payload(candidate)
            if extracted:
                return _validate(extracted, reason)
            return None, "non-JSON string payload"
        return None, f"unexpected payload type {type(candidate).__name__}"

    last_error = ""
    for attempt in range(max_retries + 1):
        attempt_messages = list(messages)
        if attempt > 0:
            attempt_messages.append({
                "role": "user",
                "content": _format_retry_hint(stage, context, attempt, last_error, required_fields),
            })

        data = await llm_client.generate_json(attempt_messages, **kwargs)
        validated, reason = _validate(data, "")
        if validated is not None:
            return validated

        logger.warning(
            "JSON validation failed [stage=%s] attempt=%d/%d reason=%s",
            stage,
            attempt + 1,
            max_retries + 1,
            reason,
        )

        last_error = reason

    suffix = f": {context}" if context else ""
    raise ValueError(f"{stage} returned no valid JSON after retries{suffix}: {last_error}")
