"""Registered Prehop JSON Schema contracts shared by requests and provenance."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, create_model

PREHOP_STRUCTURED_PROFILE = 'prehop-json-schema-v3'
# JSON Schema uses search semantics; grammar backends may use full matching.
# This accepts the same nonblank strings under both, including newlines.
NONBLANK_PATTERN = r'[\s\S]*\S[\s\S]*'
Nonempty = Annotated[StrictStr, Field(pattern=NONBLANK_PATTERN)]
_CONFIG = ConfigDict(extra='forbid', strict=True)


class StructuredOutputError(RuntimeError):
    """A constrained response failed; do not repair or downgrade its contract."""


@dataclass(frozen=True)
class StructuredContract:
    name: str
    model: type[BaseModel]

    def schema(self) -> dict[str, Any]:
        schema = self.model.model_json_schema()
        return schema

    def provenance(self) -> dict[str, str]:
        encoded = json.dumps(self.schema(), sort_keys=True, separators=(',', ':')).encode()
        return {'profile': PREHOP_STRUCTURED_PROFILE, 'schema_name': self.name,
                'schema_sha256': hashlib.sha256(encoded).hexdigest()}

    def response_format(self) -> dict[str, Any]:
        return {'type': 'json_schema', 'json_schema': {'name': self.name, 'strict': True, 'schema': self.schema()}}


def question_contract(question_schema: str = 'legacy', limit: int = 3) -> StructuredContract:
    name = f'prehop_index_{question_schema}_v1'
    minus: Any = Nonempty
    plus: Any = Nonempty
    if question_schema != 'legacy':
        raise ValueError(f'Unsupported question schema: {question_schema}')
    model = create_model(name, __config__=_CONFIG,
                         q_minus=(list[minus], Field(..., max_length=limit)),
                         q_plus=(list[plus], Field(..., max_length=limit)))
    return StructuredContract(name, model)


def ranking_contract(candidate_ids: list[str], top_k: int) -> StructuredContract:
    count = min(top_k, len(candidate_ids))
    identifier = Literal[tuple(candidate_ids)]
    model = create_model('prehop_ranking_v1', __config__=_CONFIG,
                         ranking=(list[identifier], Field(..., min_length=count, max_length=count)))
    return StructuredContract('prehop_ranking_v1', model)


def structured_bundle_sha256() -> str:
    """Bind the materialized request schemas."""
    schemas = [question_contract(mode).schema() for mode in ('legacy',)]
    schemas.append(ranking_contract(['C000', 'C001'], 1).schema())
    bundle = {'profile': PREHOP_STRUCTURED_PROFILE, 'schemas': schemas}
    digest = hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return _HISTORICAL_BUNDLE_IDENTITIES.get(digest, digest)


def structured_index_bundle_sha256() -> str:
    schemas = [question_contract(mode).schema() for mode in ('legacy',)]
    bundle = {'profile': PREHOP_STRUCTURED_PROFILE, 'schemas': schemas}
    digest = hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return _HISTORICAL_BUNDLE_IDENTITIES.get(digest, digest)


# The published v3 bundle also hashed two unused experimental schemas. The
# remaining requests are byte-identical, so preserve their index/cache identity.
# A change to the profile or any retained schema produces a new digest instead.
_HISTORICAL_BUNDLE_IDENTITIES = {
    '01e0468f3ace65579d34dee4b9cecbfdaddccc5ca5d6c4b2471605468171ff51': '766611f7c1524bebe36ca767a567ff5179eea6cd4d39952c539faa5d87d80218',
    '470781fcf1eafaa71bb724e1ce4f6e54e14d73e24e16a24bc9db0f0d97cf8d38': '00a2a48f12d74f5c8b2136ccecc8f26befdf7e34233582bcfd0a816cf0ba5010',
}
