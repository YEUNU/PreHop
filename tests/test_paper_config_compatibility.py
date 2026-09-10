"""Actual effective settings, rather than incidental Git state, govern reuse."""
from copy import deepcopy

import pytest
from test_paper_runtime_contract import _canonical_transport

from core import paper_compatibility as compatibility
from core.paper_policy import canonical_semantic_index_policy
from core.structured_outputs import structured_bundle_sha256


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    _canonical_transport(monkeypatch)


def test_resolved_matrix_is_config_only_and_preserves_environment(monkeypatch):
    import os
    before = dict(os.environ)
    context = compatibility.context_configuration()
    assert len(context['targets']) == 14
    assert dict(os.environ) == before
    monkeypatch.setattr('utils.provenance.code_provenance', lambda: {'revision': 'new', 'dirty': True})
    assert context == compatibility.context_configuration()
    assert 'code' not in context and 'verifier_sha256' not in context


def test_only_changed_method_contract_invalidates_target(monkeypatch):
    first = compatibility.target_configuration('prehop', 'hotpotqa')
    unrelated = compatibility.target_configuration('lightrag', 'hotpotqa')
    monkeypatch.setitem(compatibility.METHOD_CONTRACT_VERSIONS, 'prehop', 'paper-method-future')
    assert first != compatibility.target_configuration('prehop', 'hotpotqa')
    assert unrelated == compatibility.target_configuration('lightrag', 'hotpotqa')


def test_real_prompt_contents_change_policy_but_docstrings_do_not(monkeypatch):
    from utils.prompts import indexing, shared
    first = canonical_semantic_index_policy('prehop', 'hotpotqa')
    monkeypatch.setattr(shared.build_answer_prompt, '__doc__', 'Unrelated documentation')
    assert first == canonical_semantic_index_policy('prehop', 'hotpotqa')
    monkeypatch.setattr(indexing, 'HOPRAG_PROMPT', indexing.HOPRAG_PROMPT + '\nChanged instruction.')
    assert first != canonical_semantic_index_policy('prehop', 'hotpotqa')


def test_materialized_schema_change_invalidates_bundle(monkeypatch):
    from core import structured_outputs
    first = structured_bundle_sha256()
    original = structured_outputs.question_contract
    def changed(*args, **kwargs):
        return original(*args, **{**kwargs, 'limit': 2})
    monkeypatch.setattr(structured_outputs, 'question_contract', changed)
    assert first != structured_bundle_sha256()


def test_runtime_compatibility_preserves_explicit_location_and_content():
    first = {'main_runtime': {'prefix': '/old', 'python': '/old/python', 'installed_sha256': 'a'},
             'runtime_freeze': {'path': '/old/freeze', 'sha256': 'b', 'size': 2}}
    assert compatibility.runtime_compatibility(first) == first
    changed = deepcopy(first)
    changed['main_runtime']['installed_sha256'] = 'changed'
    assert compatibility.runtime_compatibility(first) != compatibility.runtime_compatibility(changed)


def test_actual_consumer_cap_changes_method_policy(monkeypatch):
    from core.config import RAGConfig
    from core.generation_profiles import request_settings
    before = canonical_semantic_index_policy('prehop', 'hotpotqa')
    query_before = compatibility.method_identity('prehop')
    monkeypatch.setattr(RAGConfig, 'SYNTHESIS_MAX_OUTPUT_TOKENS', 64)
    assert request_settings('answer')['max_tokens'] == 64
    assert before == canonical_semantic_index_policy('prehop', 'hotpotqa')
    assert query_before != compatibility.method_identity('prehop')




def test_protocol_and_fixture_are_in_static_review_scope(monkeypatch):
    from scripts import cold_canary_fixture, paper_gate_ledger
    context = compatibility.context_configuration()
    assert context['protocol']['cold_fixture'] == cold_canary_fixture.fixture_identity()
    monkeypatch.setattr(paper_gate_ledger, 'STAGES', (*paper_gate_ledger.STAGES, 'new_stage'))
    assert compatibility.without_realized_runtime(context) != compatibility.without_realized_runtime(compatibility.context_configuration())


def test_failed_probe_does_not_prevent_actual_checkpoint_resume_admission(tmp_path):
    from scripts.verify_paper_target import persist_admission
    output = tmp_path / 'admission.json'
    persist_admission(output, {'status': 'failed', 'errors': ['partial checkpoint']})
    assert not output.exists() and len(list(tmp_path.glob('admission.failed-validation-*.json'))) == 1
    persist_admission(output, {'status': 'admitted', 'errors': []})
    assert output.is_file()
