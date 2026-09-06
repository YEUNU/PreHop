"""Preregistered two-source integration fixture; never a benchmark dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / 'configs/cold_canary/museum_rich_entities_v2.json'
FIXTURE_SHA256 = 'ce4cfea5eb2f6eb784afaa669c762ec82e5869c98144b9512ac85e3b6fcb841a'
DATASETS = ('multihoprag', 'musique')


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def lines_digest(values: list[str]) -> str:
    return hashlib.sha256('\n'.join(values).encode()).hexdigest()


def load_fixture() -> dict:
    raw = FIXTURE_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise RuntimeError('Preregistered cold fixture bytes changed')
    return json.loads(raw)


def fixture_identity() -> dict:
    fixture = load_fixture()
    return {key: fixture[key] for key in ('kind', 'version', 'fixture_id')} | {'sha256': FIXTURE_SHA256}


def query_record(dataset: str) -> dict:
    if dataset not in DATASETS:
        raise ValueError('Unsupported cold fixture dataset alias')
    fixture = load_fixture()
    row = fixture['query'] | {'dataset': dataset, 'cold_fixture': fixture_identity()}
    if dataset == 'musique':
        row['evidence_paragraph_ids'] = ['musique:synthetic_cold_v2_1', 'musique:synthetic_cold_v2_2']
    else:
        row['evidence_docs'] = ['Meridian collection register 1', 'Meridian collection register 2']
    return row


def stage_fixture(base: Path, dataset: str) -> tuple[Path, dict, dict]:
    """Atomically reserve a fresh destination before writing exact fixed bytes."""
    row = query_record(dataset)
    fixture = load_fixture()
    base.mkdir(parents=True, exist_ok=False)
    corpus = base / f'{dataset}_corpus'
    corpus.mkdir()
    for document in fixture['documents']:
        with (corpus / document['filename']).open('x', encoding='utf-8') as stream:
            stream.write(document['text'])
    files = sorted(document['filename'] for document in fixture['documents'])
    hashes = {name: hashlib.sha256((corpus / name).read_bytes()).hexdigest() for name in files}
    records = [{'source_id': Path(name).stem, 'filename': name, 'content_sha256': hashes[name]} for name in files]
    if dataset == 'musique':
        for record in records:
            record['paragraph_id'] = 'musique:' + record['source_id'].removeprefix('musique_')
    manifest = {'schema_version': 2, 'paragraph_count': 2,
                'source_ids_sha256': lines_digest([Path(name).stem for name in files]),
                'corpus_records_sha256': digest(records),
                'corpus_files_sha256': lines_digest([name + '\0' + hashes[name] for name in files]),
                'query_ids_sha256': lines_digest([row['_id']]),
                'query_records_sha256': lines_digest([json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'))])}
    if dataset == 'musique':
        manifest['paragraph_ids_sha256'] = lines_digest(sorted(record['paragraph_id'] for record in records))
    manifest['fingerprint'] = digest(manifest)
    with (corpus / 'corpus_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True)
        stream.write('\n')
    return corpus, manifest, row


def validate_fixture(corpus: Path, dataset: str, record: dict, identity: object) -> None:
    if identity != fixture_identity() or record != query_record(dataset):
        raise RuntimeError('Cold fixture/query is not the preregistered protocol')
    expected = {document['filename']: document['text'].encode() for document in load_fixture()['documents']}
    actual = {path.name: path.read_bytes() for path in corpus.iterdir() if path.is_file() and path.suffix in {'.txt', '.md'}}
    if actual != expected:
        raise RuntimeError('Cold fixture source bytes differ from preregistration')
