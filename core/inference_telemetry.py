"""Per-query inference accounting without leaking request content or credentials."""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any

_CURRENT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "prehop_inference_telemetry", default=None
)


_STRUCTURED_OBSERVER = contextvars.ContextVar('structured_attempt_observer', default=None)


@contextmanager
def observe_structured_attempts(observer):
    """Optional context-local diagnostics; unset for all uninstrumented clients."""
    token = _STRUCTURED_OBSERVER.set(observer)
    try:
        yield
    finally:
        _STRUCTURED_OBSERVER.reset(token)


def begin() -> contextvars.Token:
    return _CURRENT.set({
        "generation_calls": 0,
        "embedding_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reported_cost": 0.0,
        "token_usage_complete": True,
        "cost_complete": True,
    })


def record(kind: str, response: Any) -> None:
    state = _CURRENT.get()
    if state is None:
        return
    state[f"{kind}_calls"] += 1
    usage = getattr(response, "usage", None)
    if usage is None:
        state["token_usage_complete"] = False
    else:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, name, 0)
            if isinstance(value, int) and not isinstance(value, bool):
                state[name] += value
            elif name != "completion_tokens":
                state["token_usage_complete"] = False
    hidden = getattr(response, "_hidden_params", None)
    cost = hidden.get("response_cost") if isinstance(hidden, dict) else getattr(response, "response_cost", None)
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        state["reported_cost"] += float(cost)
    else:
        state["cost_complete"] = False


def finish(token: contextvars.Token, *, external_complete: bool = True) -> dict[str, Any]:
    state = dict(_CURRENT.get() or {})
    _CURRENT.reset(token)
    if not external_complete:
        state["token_usage_complete"] = False
        state["cost_complete"] = False
        state["external_worker_usage_complete"] = False
    return state


def snapshot() -> dict[str, Any]:
    """Read phase totals before persisting an index artifact; keep the scope alive."""
    from copy import deepcopy
    return deepcopy(_CURRENT.get() or {})


def record_generation_transport_failure(metadata: dict) -> None:
    state = _CURRENT.get()
    if state is not None:
        state['generation_calls'] += 1
        state['generation_transport_failures'] = state.get('generation_transport_failures', 0) + 1
        state.setdefault('generation_transport_failure_attempts', []).append(metadata)
        state['token_usage_complete'] = False
        state['cost_complete'] = False


def record_structured_contract(provenance: dict[str, str]) -> None:
    """Persist dynamic request schema identities without prompt or response text."""
    state = _CURRENT.get()
    if state is not None:
        state.setdefault("structured_output_contracts", []).append(dict(provenance))


def record_structured_attempt(metadata: dict, *, valid: bool) -> None:
    """Count all format attempts, retaining safe detail only for discarded outputs."""
    observer = _STRUCTURED_OBSERVER.get()
    if observer is not None:
        observer(metadata, valid=valid)
    state = _CURRENT.get()
    if state is None:
        return
    state['structured_attempt_count'] = state.get('structured_attempt_count', 0) + 1
    state['structured_elapsed_seconds'] = state.get('structured_elapsed_seconds', 0.0) + metadata['elapsed_seconds']
    if not valid:
        state.setdefault('structured_invalid_attempts', []).append(dict(metadata))
