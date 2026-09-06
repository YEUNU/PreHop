"""Frozen response shape for the controlled Youtu construction consumer only."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace
from typing import Any

YOUTU_STRUCTURED_PROFILE = "youtu-json-schema-v1"


def youtu_response_format() -> dict[str, Any]:
    strings = {"type": "array", "items": {"type": "string"}}
    schema = {
        "type": "object",
        "properties": {
            "attributes": {"type": "object", "additionalProperties": strings},
            "triples": {"type": "array", "items": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3}},
            "entity_types": {"type": "object", "additionalProperties": {"type": "string"}},
            "new_schema_types": {"type": "object", "properties": {key: strings for key in ("nodes", "relations", "attributes")},
                                 "required": ["nodes", "relations", "attributes"], "additionalProperties": False},
        },
        "required": ["attributes", "triples", "entity_types", "new_schema_types"],
        "additionalProperties": False,
    }
    return copy.deepcopy({"type": "json_schema", "json_schema": {
        "name": "youtu_native_extraction_contract_v1", "strict": True, "schema": schema,
    }})


def youtu_profile_sha256() -> str:
    return hashlib.sha256(json.dumps(youtu_response_format(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_youtu_raw_response(response: Any) -> None:
    """Observe raw conformance before the native cleaner/parser sees the response."""
    if len(response.choices) != 1 or response.choices[0].finish_reason != "stop":
        raise ValueError("Youtu structured extraction did not finish completely")
    message = response.choices[0].message
    if getattr(message, "refusal", None) or getattr(message, "tool_calls", None):
        raise ValueError("Youtu structured extraction returned refusal or tool output")
    if not isinstance(message.content, str) or not message.content.strip():
        raise ValueError("Youtu structured extraction has no raw content")
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise ValueError("duplicate extraction JSON property")
            output[key] = value
        return output
    def invalid_constant(_value):
        raise ValueError("non-finite extraction JSON constant")
    value = json.loads(message.content, object_pairs_hook=pairs, parse_constant=invalid_constant)
    def strings(items):
        return isinstance(items, list) and all(isinstance(item, str) for item in items)
    if not isinstance(value, dict) or set(value) != {"attributes", "triples", "entity_types", "new_schema_types"}:
        raise ValueError("Youtu structured extraction top-level schema mismatch")
    attributes, triples, types, evolved = (value[key] for key in ("attributes", "triples", "entity_types", "new_schema_types"))
    if not isinstance(attributes, dict) or not all(strings(items) for items in attributes.values()):
        raise ValueError("Youtu structured extraction attributes must contain string arrays")
    if not isinstance(triples, list) or not all(strings(item) and len(item) == 3 for item in triples):
        raise ValueError("Youtu structured extraction triples must contain exactly three strings")
    if not isinstance(types, dict) or not all(isinstance(item, str) for item in types.values()):
        raise ValueError("Youtu structured extraction entity_types must contain strings")
    if not isinstance(evolved, dict) or set(evolved) != {"nodes", "relations", "attributes"} or not all(strings(item) for item in evolved.values()):
        raise ValueError("Youtu structured extraction new_schema_types mismatch")


class YoutuConstructionClient:
    """Delegate the public SDK call unchanged except its extraction-only format."""

    def __init__(self, client: Any):
        if not callable(getattr(getattr(getattr(client, "chat", None), "completions", None), "create", None)):
            raise TypeError("Youtu construction requires the public chat completions SDK client")
        self._client = client
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        if any(key in kwargs for key in ("response_format", "extra_body", "structured_outputs")) or any(key.startswith("guided_") for key in kwargs):
            raise ValueError("Youtu construction structured format cannot be overridden")
        response = self._client.chat.completions.create(**kwargs, response_format=youtu_response_format())
        validate_youtu_raw_response(response)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
