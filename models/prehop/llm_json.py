"""Shared fail-loud wrapper around VLLMClient.generate_json for prehop's
own indexing/retrieval call sites (paper's own pipeline only).

core.vllm_client.generate_json swallows a persistent JSON-parse failure
after exhausting its own retries, returning {} rather than raising — that
file is shared by every strategy, so its own retry-exhaustion behavior is
intentionally left permissive (out of this fail-loud scope). Every
prehop-specific call site that depends on generate_json succeeding wraps it
with this helper instead, so a genuine failure surfaces as an exception
rather than silently degrading to empty/default content.
"""
from typing import Any


async def generate_json_or_raise(llm_client, messages, stage: str, context: str = "", **kwargs) -> dict[str, Any]:
    data = await llm_client.generate_json(messages, **kwargs)
    if not data:
        suffix = f": {context}" if context else ""
        raise ValueError(f"{stage} returned no valid JSON after retries{suffix}")
    return data
