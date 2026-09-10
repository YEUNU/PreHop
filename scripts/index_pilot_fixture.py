"""Fixed source-only sample for adapter throughput pilots, never paper results."""
from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

from scripts.cold_canary_fixture import digest, lines_digest


def stage_index_pilot(base: Path, dataset: str, count: int = 32) -> Path:
    source = Path(__file__).resolve().parents[1] / 'data' / f'{dataset}_corpus'
    candidates = sorted(p.name for p in source.iterdir() if p.suffix in {'.txt', '.md'})
    names = sorted(random.Random(42).sample(candidates, min(count, len(candidates))))
    base.mkdir(parents=True, exist_ok=False)
    corpus = base / f'{dataset}_corpus'
    corpus.mkdir()
    for name in names:
        shutil.copyfile(source / name, corpus / name)
    hashes = {name: hashlib.sha256((corpus / name).read_bytes()).hexdigest() for name in names}
    records = ([{'source_id': Path(name).stem, 'filename': name, 'content_sha256': hashes[name]}
                for name in names])
    manifest = {'schema_version': 2, 'paragraph_count': len(names),
                'source_ids_sha256': lines_digest(sorted(Path(name).stem for name in names)),
                'corpus_records_sha256': digest(records),
                'corpus_files_sha256': lines_digest([name + '\0' + hashes[name] for name in names])}
    manifest['fingerprint'] = digest(manifest)
    (corpus / 'corpus_manifest.json').write_text(json.dumps(manifest, sort_keys=True) + '\n')
    metadata = source / 'source_metadata.json'
    if metadata.exists():
        value = json.loads(metadata.read_text())
        value['records'] = {k: v for k, v in value['records'].items() if k in names}
        (corpus / 'source_metadata.json').write_text(json.dumps(value) + '\n')
    (base / 'pilot.json').write_text(json.dumps({'kind': 'non_reportable_index_pilot', 'seed': 42,
        'source_manifest': json.loads((source / 'corpus_manifest.json').read_text())['fingerprint'],
        'sample_manifest': manifest['fingerprint'], 'count': len(names)}) + '\n')
    return corpus
