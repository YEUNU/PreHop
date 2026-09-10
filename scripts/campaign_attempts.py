"""One attempt selection contract for producers, validators and matrix assembly."""
from __future__ import annotations

import re


def validated_attempts(values: dict | None) -> dict[str, str]:
    from core.strategy_registry import PRIMARY_STRATEGIES
    values = {} if values is None else values
    allowed = {f'{phase}/{dataset}/{method}' for phase in ('cold', 'one-query')
               for dataset in ('multihoprag', 'hotpotqa') for method in PRIMARY_STRATEGIES}
    if not isinstance(values, dict) or set(values) - allowed:
        raise ValueError('Retry map contains an unsupported target step')
    if any(not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value)
           for value in values.values()):
        raise ValueError('Retry attempt must be a safe nonempty identifier')
    return dict(values)


def selected_attempt(default: str, values: dict | None, step_id: str) -> str:
    return validated_attempts(values).get(step_id, default)
