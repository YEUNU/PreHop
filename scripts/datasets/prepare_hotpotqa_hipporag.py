"""Prepare the original HippoRAG release, retaining every sentence and query."""
import argparse
import hashlib
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datasets.hotpotqa_common import annotation_coverage, canonical

REVISION = 'b144c46df14cabe5f5822d8caded4bec5f709461'
BASE = f'https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/{REVISION}/data'
PROTOCOL = 'hotpotqa-hipporag-v1-1000'


def prepare(raw, output, queries_output):
    raw, output, queries_output = map(Path, (raw, output, queries_output))
    corpus = json.loads((raw / 'hotpotqa_corpus.json').read_text())
    original = json.loads((raw / 'hotpotqa.json').read_text())
    queries = [{'_id': f"{row['_id']}__row_{i:04d}", 'original_query_id': row['_id'],
        'release_row': i, 'dataset': 'hotpotqa', 'query': row['question'],
        'ground_truth': row['answer'], 'question_type': row['type'], 'category': row['type'],
        'supporting_facts': row['supporting_facts'], 'evidence_docs': sorted({f[0] for f in row['supporting_facts']}),
        'evidence_facts': [], 'protocol': PROTOCOL} for i,row in enumerate(original)]
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with sqlite3.connect(output / 'sentences.sqlite3') as db:
        db.execute('CREATE TABLE paragraphs (source_id TEXT PRIMARY KEY, title TEXT UNIQUE NOT NULL, sentences TEXT NOT NULL, content_sha256 TEXT NOT NULL)')
        for title, sentences in sorted(corpus.items()):
            source = 'hotpotqa_' + hashlib.sha256(title.encode()).hexdigest()[:24]
            content = ('Title: ' + title + '\n\n' + ''.join(sentences) + '\n').encode()
            digest = hashlib.sha256(content).hexdigest()
            (output / (source + '.txt')).write_bytes(content)
            db.execute('INSERT INTO paragraphs VALUES (?,?,?,?)', (source, title, canonical(sentences), digest))
            records.append({'source_id': source, 'filename': source + '.txt', 'content_sha256': digest})
        coverage = annotation_coverage(db, queries)
    records.sort(key=lambda r:r['source_id'])
    sha = lambda value: hashlib.sha256(value.encode()).hexdigest()
    manifest = {'schema_version': 2, 'protocol': PROTOCOL, 'paragraph_count': len(corpus),
        'query_count': len(queries), 'unique_original_queries': len({q['original_query_id'] for q in queries}),
        'query_population': 'all released rows, including duplicate questions; cluster inference by original_query_id', 'official_fullwiki': False,
        'corpus_construction': 'unchanged released HippoRAG union of supporting and distractor contexts',
        'source_repository': 'https://github.com/OSU-NLP-Group/HippoRAG', 'source_revision': REVISION,
        'source_urls': [f'{BASE}/{name}' for name in ('hotpotqa.json','hotpotqa_corpus.json')],
        'source_files_sha256': {name:hashlib.sha256((raw/name).read_bytes()).hexdigest() for name in ('hotpotqa.json','hotpotqa_corpus.json')},
        'source_ids_sha256': sha('\n'.join(r['source_id'] for r in records)),
        'corpus_files_sha256': sha('\n'.join(r['filename']+'\0'+r['content_sha256'] for r in records)),
        'corpus_records_sha256': sha(canonical(records)),
        'query_ids_sha256': sha('\n'.join(sorted(q['_id'] for q in queries))),
        'query_records_sha256': sha('\n'.join(canonical(q) for q in sorted(queries,key=lambda q:q['_id']))),
        'annotation_coverage': coverage, 'license': 'CC-BY-SA-4.0',
        'sentence_projection': 'original released title and sentence index; gold never selects predictions'}
    manifest['fingerprint'] = sha(canonical(manifest))
    queries_output.parent.mkdir(parents=True, exist_ok=True)
    queries_output.write_text(json.dumps(queries, ensure_ascii=False, indent=2))
    (output/'corpus_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=ROOT/'data/hotpotqa_hipporag_raw')
    p.add_argument('--output',type=Path,default=ROOT/'data/hotpotqa_corpus')
    p.add_argument('--queries-output',type=Path,default=ROOT/'data/hotpotqa_queries.json')
    p.add_argument('--download',action='store_true')
    args=p.parse_args()
    if args.download:
        args.raw.mkdir(parents=True,exist_ok=True)
        for name in ('hotpotqa.json','hotpotqa_corpus.json'):
            with urllib.request.urlopen(f'{BASE}/{name}',timeout=60) as response:
                (args.raw/name).write_bytes(response.read())
    print(json.dumps(prepare(args.raw,args.output,args.queries_output),indent=2))


if __name__ == '__main__':
    main()
