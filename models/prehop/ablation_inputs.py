"""Frozen retrieval inputs for explicit ablations only; no primary-path cache."""
import gzip
import hashlib
import json
from pathlib import Path


def input_path(directory,query):
    return Path(directory)/(hashlib.sha256(query.encode()).hexdigest()+'.json.gz')


def write_input(directory,query,value):
    path=input_path(directory,query)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(value,ensure_ascii=False).encode(),compresslevel=1))


def read_input(directory,query):
    # Return a fresh object so downstream scoring cannot mutate another arm.
    occurrences = Path(directory) / 'occurrences'
    if occurrences.is_dir():
        from models.prehop.tracing import _IDENTITY
        query_id = (_IDENTITY.get() or {})['query_id']
        path = input_path(occurrences, query_id)
    else:
        path = input_path(directory, query)
    return json.loads(gzip.decompress(path.read_bytes()))
