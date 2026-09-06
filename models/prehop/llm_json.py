"""Prehop registered schema validation; no text extraction or format downgrade."""
from typing import Any

from core.structured_outputs import StructuredContract, StructuredOutputError


async def generate_json_or_raise(
    llm_client, messages, stage: str, context: str = "",
    required_fields: dict[str, type] | None = None,
    structured_contract: StructuredContract | None = None, **kwargs,
) -> dict[str, Any]:
    if structured_contract is None:
        raise StructuredOutputError(f"{stage} has no registered structured-output contract")
    kwargs.pop("max_parse_retries", None)
    kwargs.setdefault("json_debug_label", stage)
    value = await llm_client.generate_json(messages, structured_contract=structured_contract, **kwargs)
    return structured_contract.validate(value)
