"""Registered Prehop JSON Schema contracts shared by requests and provenance."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, create_model

PREHOP_STRUCTURED_PROFILE = 'prehop-json-schema-v2'
# JSON Schema uses search semantics; grammar backends may use full matching.
# This accepts the same nonblank strings under both, including newlines.
NONBLANK_PATTERN = r'[\s\S]*\S[\s\S]*'
Nonempty = Annotated[StrictStr, Field(min_length=1, pattern=NONBLANK_PATTERN)]
_CONFIG = ConfigDict(extra='forbid', strict=True)


class StructuredOutputError(RuntimeError):
    """A constrained response failed; do not repair or downgrade its contract."""


@dataclass(frozen=True)
class StructuredContract:
    name: str
    model: type[BaseModel]
    unique_ranking: bool = False

    def schema(self) -> dict[str, Any]:
        return self.model.model_json_schema()

    def provenance(self) -> dict[str, str]:
        encoded = json.dumps(self.schema(), sort_keys=True, separators=(',', ':')).encode()
        return {'profile': PREHOP_STRUCTURED_PROFILE, 'schema_name': self.name,
                'schema_sha256': hashlib.sha256(encoded).hexdigest()}

    def response_format(self) -> dict[str, Any]:
        return {'type': 'json_schema', 'json_schema': {'name': self.name, 'strict': True, 'schema': self.schema()}}

    def validate(self, value: Any) -> dict[str, Any]:
        try:
            output = self.model.model_validate(value, strict=True).model_dump()
            if self.unique_ranking and len(set(output['ranking'])) != len(output['ranking']):
                raise ValueError('ranking contains duplicate candidate IDs')
            return output
        except (ValueError, TypeError) as exc:
            raise StructuredOutputError(f'{self.name} failed its registered schema: {exc}') from exc


def question_contract(stage: str, question_schema: str = 'legacy', limit: int = 3) -> StructuredContract:
    if stage not in {'index', 'rewrite', 'refine'} or question_schema not in {'legacy', 'grounded_v1', 'linked_v2'}:
        raise ValueError('unknown structured question contract')
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError('question limit must be positive')
    if stage != 'index' and question_schema != 'legacy':
        raise ValueError('query rewrites require the string question contract')
    name = f'prehop_{stage}_{question_schema}_v1'
    minus: Any = Nonempty
    plus: Any = Nonempty
    if question_schema != 'legacy':
        anchors = Annotated[list[Nonempty], Field(min_length=0 if question_schema == 'linked_v2' else 1)]
        fields = {'question': (Nonempty, ...), 'grounding_quote': (Nonempty, ...), 'anchor_entities': (anchors, ...)}
        minus_fields = {**fields, 'answer': (Nonempty, ...)}
        if question_schema == 'linked_v2':
            minus_fields['continuation_anchor'] = (StrictStr, ...)
        minus = create_model(name + '_incoming', __config__=_CONFIG, **minus_fields)
        plus = create_model(name + '_outgoing', __config__=_CONFIG, **fields, missing_information=(Nonempty, ...))
    model = create_model(name, __config__=_CONFIG,
                         q_minus=(list[minus], Field(..., max_length=limit)),
                         q_plus=(list[plus], Field(..., max_length=limit)))
    return StructuredContract(name, model)


def ranking_contract(candidate_ids: list[str], top_k: int) -> StructuredContract:
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids) or any(not isinstance(v, str) or not v for v in candidate_ids):
        raise ValueError('ranking needs distinct nonempty candidate IDs')
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError('ranking top_k must be positive')
    count = min(top_k, len(candidate_ids))
    identifier = Literal[tuple(candidate_ids)]
    model = create_model('prehop_ranking_v1', __config__=_CONFIG,
                         ranking=(list[identifier], Field(..., min_length=count, max_length=count,
                                                         json_schema_extra={'uniqueItems': True})))
    return StructuredContract('prehop_ranking_v1', model, unique_ranking=True)


def structured_bundle_sha256() -> str:
    """Bind materialized schema contents and explicit validation semantics."""
    schemas = [question_contract('index', mode).schema() for mode in ('legacy', 'grounded_v1', 'linked_v2')]
    schemas += [question_contract(stage).schema() for stage in ('rewrite', 'refine')]
    schemas.append(ranking_contract(['C000', 'C001'], 1).schema())
    bundle = {'profile': PREHOP_STRUCTURED_PROFILE, 'schemas': schemas,
              'validation_contract': 'strict-json-schema-v1'}
    return hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
