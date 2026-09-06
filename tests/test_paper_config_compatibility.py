"""Actual effective settings, rather than incidental Git state, govern reuse."""
from copy import deepcopy

import pytest
from test_paper_runtime_contract import _canonical_transport

from core import paper_compatibility as compatibility
from core.admission import identity_sha256
from core.paper_policy import canonical_semantic_index_policy, validate_canonical_index_policy
from core.structured_outputs import structured_bundle_sha256


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    _canonical_transport(monkeypatch)


def test_resolved_matrix_is_config_only_and_preserves_environment(monkeypatch):
    import os
    before = dict(os.environ)
    context = compatibility.context_configuration()
    assert len(context['targets']) == 16
    assert dict(os.environ) == before
    monkeypatch.setattr('utils.provenance.code_provenance', lambda: {'revision': 'new', 'dirty': True})
    assert context == compatibility.context_configuration()
    assert 'code' not in context and 'verifier_sha256' not in context


def test_only_changed_method_contract_invalidates_target(monkeypatch):
    first = compatibility.target_configuration('prehop', 'musique')
    unrelated = compatibility.target_configuration('hipporag2', 'musique')
    monkeypatch.setitem(compatibility.METHOD_CONTRACT_VERSIONS, 'prehop', 'paper-method-v2')
    assert first != compatibility.target_configuration('prehop', 'musique')
    assert unrelated == compatibility.target_configuration('hipporag2', 'musique')


def test_real_prompt_contents_change_policy_but_docstrings_do_not(monkeypatch):
    from utils.prompts import indexing, shared
    first = canonical_semantic_index_policy('prehop', 'musique')
    monkeypatch.setattr(shared.build_answer_prompt, '__doc__', 'Unrelated documentation')
    assert first == canonical_semantic_index_policy('prehop', 'musique')
    monkeypatch.setattr(indexing, 'HOPRAG_PROMPT', indexing.HOPRAG_PROMPT + '\nChanged instruction.')
    assert first != canonical_semantic_index_policy('prehop', 'musique')


def test_materialized_schema_change_invalidates_bundle(monkeypatch):
    from core import structured_outputs
    first = structured_bundle_sha256()
    original = structured_outputs.question_contract
    def changed(*args, **kwargs):
        return original(*args, **{**kwargs, 'limit': 2})
    monkeypatch.setattr(structured_outputs, 'question_contract', changed)
    assert first != structured_bundle_sha256()


@pytest.mark.parametrize('field,value', [('generation_revision', 'other-model'), ('generation_seed', 7),
    ('embedding_dimensions', 128), ('method_contract', 'unknown-version'), ('prompt_configuration_sha256', 'bad')])
def test_actual_index_validator_rejects_changed_semantics(field, value):
    from core.paper_policy import canonical_operational_policy
    policy = canonical_semantic_index_policy('prehop', 'musique')
    policy['operational_config'] = canonical_operational_policy('prehop')
    policy[field] = value
    with pytest.raises(RuntimeError, match='checked-in paper policy'):
        validate_canonical_index_policy('prehop', 'musique', policy)


def test_runtime_compatibility_preserves_explicit_location_and_content():
    first = {'main_runtime': {'prefix': '/old', 'python': '/old/python', 'installed_sha256': 'a'},
             'runtime_freeze': {'path': '/old/freeze', 'sha256': 'b', 'size': 2}}
    assert compatibility.runtime_compatibility(first) == first
    changed = deepcopy(first)
    changed['main_runtime']['installed_sha256'] = 'changed'
    assert compatibility.runtime_compatibility(first) != compatibility.runtime_compatibility(changed)


def test_setup_cannot_mask_config_change():
    first = compatibility.context_configuration()
    changed = deepcopy(first)
    target = changed['targets']['musique/prehop']
    target['operational']['runtime_freeze'] = {'sha256': 'installed'}
    assert compatibility.without_realized_runtime(first) == compatibility.without_realized_runtime(changed)
    target['index']['generation_seed'] = 7
    assert compatibility.without_realized_runtime(first) != compatibility.without_realized_runtime(changed)
    assert identity_sha256(first) != identity_sha256(changed)


def test_actual_consumer_cap_changes_method_policy(monkeypatch):
    from core.config import RAGConfig
    from core.generation_profiles import request_settings
    before = canonical_semantic_index_policy('prehop', 'musique')
    monkeypatch.setattr(RAGConfig, 'SYNTHESIS_MAX_OUTPUT_TOKENS', 64)
    assert request_settings('answer')['max_tokens'] == 64
    assert before != canonical_semantic_index_policy('prehop', 'musique')


def test_resolver_rejects_explicit_unknown_or_changed_semantics(monkeypatch):
    monkeypatch.setenv('RAG_GRAPH_HOP_DEPTH', '2')
    with pytest.raises(RuntimeError):
        compatibility.target_configuration('prehop', 'musique')


def test_protocol_and_fixture_are_in_static_review_scope(monkeypatch):
    from scripts import cold_canary_fixture, paper_gate_ledger
    context = compatibility.context_configuration()
    assert context['protocol']['cold_fixture'] == cold_canary_fixture.fixture_identity()
    monkeypatch.setattr(paper_gate_ledger, 'STAGES', (*paper_gate_ledger.STAGES, 'new_stage'))
    assert compatibility.without_realized_runtime(context) != compatibility.without_realized_runtime(compatibility.context_configuration())


def test_static_go_rejects_same_verdict_for_other_configuration(tmp_path, monkeypatch):
    import json

    from scripts import paper_gate_ledger as gate
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    context = {'version': 'test', 'targets': {'method': {'cap': 1}}}
    monkeypatch.setattr(gate, '_context', lambda: context)
    report = tmp_path / 'review.json'
    report.write_text(json.dumps({'verdict': 'GO', 'reviewed_configuration_sha256': 'wrong'}))
    evidence = {'status': 'canary_passed', 'kind': 'independent_static_review', 'reviewer': 'test',
                'report': {'path': report.name, 'sha256': gate.sha256_file(report)}}
    with pytest.raises(RuntimeError, match='different effective configuration'):
        gate._validate_evidence('static_go', tmp_path / 'evidence.json', evidence)
    report.write_text(json.dumps({'verdict': 'GO', 'reviewed_configuration_sha256': identity_sha256(compatibility.without_realized_runtime(context))}))
    evidence['report']['sha256'] = gate.sha256_file(report)
    gate._validate_evidence('static_go', tmp_path / 'evidence.json', evidence)
    context['targets']['method']['cap'] = 2
    with pytest.raises(RuntimeError, match='different effective configuration'):
        gate._validate_evidence('static_go', tmp_path / 'evidence.json', evidence)


def test_admission_revalidation_preserves_original_bytes_and_rejects_migration(tmp_path):
    import json

    from scripts.verify_paper_target import persist_admission
    path = tmp_path / 'admission.json'
    current = {'status': 'admitted', 'path': 'result', 'details_path': 'details', 'strategy': 'naive',
               'dataset': 'musique', 'bindings': {'configuration_sha256': 'same'}, 'errors': [],
               'verification_provenance': {'revision': 'old'}}
    persist_admission(path, current)
    before = path.read_bytes()
    persist_admission(path, {**current, 'verification_provenance': {'revision': 'new'}})
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError, match='fresh namespace'):
        persist_admission(path, {**current, 'bindings': {'configuration_sha256': 'different'}})
    with pytest.raises(RuntimeError, match='failed current validation'):
        persist_admission(path, {**current, 'status': 'failed', 'errors': ['forged result']})
    assert path.read_bytes() == before
    assert json.loads(before)['verification_provenance']['revision'] == 'old'


def test_failed_probe_does_not_prevent_actual_checkpoint_resume_admission(tmp_path):
    from scripts.verify_paper_target import persist_admission
    output = tmp_path / 'admission.json'
    persist_admission(output, {'status': 'failed', 'errors': ['partial checkpoint']})
    assert not output.exists() and len(list(tmp_path.glob('admission.failed-validation-*.json'))) == 1
    persist_admission(output, {'status': 'admitted', 'errors': []})
    assert output.is_file()
