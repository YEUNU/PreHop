"""One attempt selection contract for producers, validators and matrix assembly."""
from __future__ import annotations


def attempt_mapping(values: dict | None) -> dict[str, str]:
    return dict(values or {})


def selected_attempt(default: str, values: dict | None, step_id: str) -> str:
    return attempt_mapping(values).get(step_id, default)
