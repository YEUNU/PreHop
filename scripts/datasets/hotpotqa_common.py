"""Serialization and annotation observations shared by paper preparation tools."""
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_file(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_coverage(db, queries):
    """Observe upstream label gaps without rewriting gold or injecting text."""
    missing, total = [], 0
    for q in queries:
        for title, idx in q["supporting_facts"]:
            total += 1
            row = db.execute("SELECT sentences FROM paragraphs WHERE title=?", (title,)).fetchone()
            count = len(json.loads(row[0])) if row else None
            if count is None or idx >= count:
                missing.append({"query_id":q["_id"], "title":title, "sentence_index":idx,
                                "corpus_sentences":count})
    return {"fact_occurrences":total, "available_fact_occurrences":total-len(missing),
            "affected_queries":len({r["query_id"] for r in missing}), "unavailable_facts":missing,
            "policy":"preserve_all_official_queries_and_gold_labels_without_repair"}
