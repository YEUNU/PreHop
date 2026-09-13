"""Native indexing completion keeps observations without retired approval hooks."""
import pytest


@pytest.mark.asyncio
async def test_old_extraction_profile_does_not_call_removed_validator(monkeypatch, tmp_path):
    from models.external_research import official_indexer as module
    written = []
    monkeypatch.setattr(module, 'stage_corpus', lambda *a: ([{'source_id': 'one'}], tmp_path))
    monkeypatch.setattr(module, 'run_index_worker', lambda *a: {'stats': {'native': 'kept'}})
    monkeypatch.setattr(module, 'artifact_inventory', lambda *a: {})
    monkeypatch.setattr(module, 'semantic_index_policy', lambda *a: {'extraction_validation_profile': 'retired'})
    monkeypatch.setattr(module, 'semantic_config_sha256', lambda *a: 'semantic')
    monkeypatch.setattr(module, 'configured_embedding_model', lambda: 'embed')
    monkeypatch.setattr(module, 'configured_embedding_revision', lambda: 'revision')
    monkeypatch.setattr(module, 'source_set_sha256', lambda *a: 'sources')
    monkeypatch.setattr(module, 'corpus_records_sha256', lambda *a: 'records')
    monkeypatch.setattr(module, 'snapshot_metadata_path', lambda *a: tmp_path/'snapshot.json')
    monkeypatch.setattr(module, '_write_json', lambda path, value: written.append(value))
    await module.run_official_index('gfm_rag', str(tmp_path), 'hotpotqa', {})
    assert written[0]['official_stats'] == {'native': 'kept'}
    assert written[0]['status'] == 'complete'
