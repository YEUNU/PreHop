"""Generate and decode Prehop JSON using the requested output schema."""
from typing import Any

from core.structured_outputs import StructuredContract


async def generate_json_or_raise(
    llm_client, messages, stage: str, context: str = "",
    structured_contract: StructuredContract | None = None, **kwargs,
) -> dict[str, Any]:
    kwargs.pop("max_parse_retries", None)
    kwargs.setdefault("json_debug_label", stage)
    value = await llm_client.generate_json(messages, structured_contract=structured_contract, **kwargs)
    return value
