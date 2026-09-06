"""Canonical separation of method semantics from paths and throughput controls."""

from __future__ import annotations

import hashlib
import json

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def parse_strict_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} has an invalid boolean value")

_NON_SEMANTIC_FIELDS = {
    "index_namespace",
    "schema_path",
    "gfm_checkpoint",
}


def semantic_index_policy(policy: dict | None) -> dict:
    return {key: value for key, value in dict(policy or {}).items() if key not in _NON_SEMANTIC_FIELDS}


def semantic_config_sha256(policy: dict | None) -> str:
    payload = json.dumps(semantic_index_policy(policy), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
