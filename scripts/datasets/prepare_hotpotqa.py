"""Prepare the official HotpotQA fullwiki dev set and Wikipedia abstracts.

The archive is streamed, never extracted with tar.extractall. Gold context is
not a corpus source. Sentence boundaries and titles are preserved for evaluation.
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import sqlite3
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_URL = "https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
ARCHIVE_MD5 = "01edf64cd120ecc03a2745352779514c"
ARCHIVE_BYTES = 1553565403
DEV_URL = "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
DEV_QUERY_IDS_SHA256 = "8f1a1b80b352ff578988c4dfb320f44dc7c08c5e4d4b86ef132598271a0adb00"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_file(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paragraphs(archive):
    with tarfile.open(archive, "r|bz2") as tar:
        for member in tar:
            if not member.isfile():
                continue
            stream = tar.extractfile(member)
            if member.name.endswith(".bz2"):
                stream = bz2.BZ2File(stream)
            with stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)


def sentences_of(row):
    text = row["text"]
    if not isinstance(text, list) or not text:
        raise ValueError("Wikipedia abstract has no sentence list")
    # The abstract release uses a flat list; accept the equivalent singleton
    # paragraph container but never flatten full article paragraphs silently.
    if len(text) == 1 and isinstance(text[0], list):
        text = text[0]
    if any(not isinstance(s, str) for s in text):
        raise ValueError("Expected the official abstract archive, not full articles")
    return text


def prepare_queries(rows):
    queries = []
    for row in rows:
        facts = row["supporting_facts"]
        if not facts or any(not isinstance(f, list) or len(f) != 2 or
                            not isinstance(f[0], str) or type(f[1]) is not int or f[1] < 0 for f in facts):
            raise ValueError("Invalid official supporting facts")
        queries.append({"_id": row["_id"], "dataset": "hotpotqa", "query": row["question"],
                        "ground_truth": row["answer"], "question_type": row["type"],
                        "category": row["type"], "supporting_facts": facts,
                        "evidence_docs": sorted({f[0] for f in facts}),
                        "evidence_facts": [], "protocol": "hotpotqa-fullwiki-dev-v1"})
    ids = [q["_id"] for q in queries]
    if len(ids) != len(set(ids)) or any(not isinstance(q, str) or not q for q in ids):
        raise ValueError("Missing or duplicate query IDs")
    return queries


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


def build(archive, gold, output, queries_path, *, verify_official=True, finalize_existing=False):
    archive, gold, output, queries_path = map(Path, (archive, gold, output, queries_path))
    if queries_path.exists() or (output / "corpus_manifest.json").exists() or (output.exists() and not finalize_existing):
        raise FileExistsError("Preserve existing prepared data; choose new output paths")
    if finalize_existing and not (output / "sentences.sqlite3").is_file():
        raise FileNotFoundError("Finalization requires an existing paragraph store")
    if verify_official and (archive.stat().st_size != ARCHIVE_BYTES or digest_file(archive, "md5") != ARCHIVE_MD5):
        raise ValueError("Official archive size/MD5 mismatch")
    queries = prepare_queries(json.loads(gold.read_text()))
    if verify_official and len(queries) != 7405:
        raise ValueError("Official fullwiki dev must contain 7,405 queries")
    query_ids_digest = hashlib.sha256("\n".join(sorted(q["_id"] for q in queries)).encode()).hexdigest()
    if verify_official and query_ids_digest != DEV_QUERY_IDS_SHA256:
        raise ValueError("Official fullwiki dev query identities differ")
    provenance_path = gold.parent / "download_provenance.json"
    provenance = None
    if verify_official:
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("gold_sha256") != digest_file(gold) or provenance.get("archive_sha256") != digest_file(archive):
            raise ValueError("Download provenance does not match official input files")
    output.mkdir(parents=True, exist_ok=finalize_existing)
    db = (sqlite3.connect((output / "sentences.sqlite3").resolve().as_uri()+"?mode=ro", uri=True)
          if finalize_existing else sqlite3.connect(output / "sentences.sqlite3"))
    try:
        if not finalize_existing:
            db.execute("CREATE TABLE paragraphs (source_id TEXT PRIMARY KEY, title TEXT UNIQUE NOT NULL, sentences TEXT NOT NULL, content_sha256 TEXT NOT NULL)")
        count = 0
        for row in paragraphs(archive):
            title, sentences = row["title"], sentences_of(row)
            if not isinstance(title, str) or not title or not str(row["id"]).isdigit():
                raise ValueError("Invalid Wikipedia paragraph identity")
            source = "hotpotqa_" + str(row["id"])
            content = ("Title: " + title + "\n\n" + "".join(sentences) + "\n").encode()
            digest = hashlib.sha256(content).hexdigest()
            if finalize_existing:
                stored = db.execute("SELECT title,sentences,content_sha256 FROM paragraphs WHERE source_id=?", (source,)).fetchone()
                if stored != (title, canonical(sentences), digest):
                    raise ValueError(f"Existing paragraph store differs from source archive: {source}")
            else:
                db.execute("INSERT INTO paragraphs VALUES (?,?,?,?)", (source, title, canonical(sentences), digest))
                (output / (source + ".txt")).write_bytes(content)
            count += 1
            if count % 10000 == 0:
                if not finalize_existing:
                    db.commit()
                if not finalize_existing or count % 100000 == 0:
                    print(f"{'Verified archive against' if finalize_existing else 'Prepared'} {count:,} official Wikipedia paragraphs", flush=True)
        db.commit()
        if db.execute("SELECT count(*) FROM paragraphs").fetchone()[0] != count:
            raise ValueError("Paragraph store contains records outside the source archive")
        coverage = annotation_coverage(db, queries)
        ids_hash, files_hash, records_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        records_hash.update(b"[")
        for i, (source, content_sha) in enumerate(db.execute("SELECT source_id,content_sha256 FROM paragraphs ORDER BY source_id")):
            if i:
                ids_hash.update(b"\n"); files_hash.update(b"\n"); records_hash.update(b",")
            filename = source + ".txt"
            ids_hash.update(source.encode())
            files_hash.update(f"{filename}\0{content_sha}".encode())
            records_hash.update(canonical({"source_id": source, "filename": filename, "content_sha256": content_sha}).encode())
        records_hash.update(b"]")
    finally:
        db.close()
    manifest = {"schema_version": 2, "paragraph_count": count,
                "source_ids_sha256": ids_hash.hexdigest(), "corpus_files_sha256": files_hash.hexdigest(),
                "corpus_records_sha256": records_hash.hexdigest(),
                "query_ids_sha256": hashlib.sha256("\n".join(sorted(q["_id"] for q in queries)).encode()).hexdigest(),
                "query_records_sha256": hashlib.sha256("\n".join(canonical(q) for q in sorted(queries,key=lambda q:q["_id"])).encode()).hexdigest(),
                "protocol": "hotpotqa-fullwiki-dev-v1", "official_archive_verified": verify_official,
                "archive_url": ARCHIVE_URL, "archive_sha256": digest_file(archive),
                "gold_sha256": digest_file(gold), "sentence_store_sha256": digest_file(output / "sentences.sqlite3"),
                "license": "CC-BY-SA-4.0", "annotation_coverage":coverage,
                "gold_supporting_sentence_coverage":coverage["available_fact_occurrences"]/coverage["fact_occurrences"],
                "archive_record_binding":"streamed_archive_records_v1"}
    if provenance is not None:
        manifest["download_provenance"] = provenance
        manifest["download_provenance_sha256"] = digest_file(provenance_path)
    manifest["fingerprint"] = hashlib.sha256(canonical(manifest).encode()).hexdigest()
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    queries_path.write_text(json.dumps(queries, ensure_ascii=False, indent=2))
    (output / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=ROOT / "data/hotpotqa_raw/hotpot_dev_fullwiki_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/hotpotqa_corpus")
    parser.add_argument("--queries-output", type=Path, default=ROOT / "data/hotpotqa_queries.json")
    parser.add_argument("--verify-only", action="store_true", help="Verify all prepared files and source provenance without modifying them")
    parser.add_argument("--finalize-existing", action="store_true", help="Recheck the entire archive against an unpublished store, then publish metadata without rewriting paragraphs")
    args = parser.parse_args()
    if args.verify_only and args.finalize_existing:
        parser.error("Choose verification or finalization")
    if args.verify_only:
        result = verify_prepared(args.archive, args.gold, args.output, args.queries_output)
    else:
        result = build(args.archive, args.gold, args.output, args.queries_output, finalize_existing=args.finalize_existing)
    print(json.dumps(result, indent=2))


def verify_prepared(archive, gold, output, queries_path):
    """Audit the complete corpus once before launching expensive model indexes."""
    import sys
    sys.path.insert(0, str(ROOT))
    from cli.index import _load_corpus_manifest, _validate_staged_snapshot
    archive, gold, output, queries_path = map(Path, (archive, gold, output, queries_path))
    if archive.stat().st_size != ARCHIVE_BYTES or digest_file(archive, "md5") != ARCHIVE_MD5:
        raise ValueError("Official archive checksum mismatch")
    provenance_path = gold.parent / "download_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("gold_sha256") != digest_file(gold) or provenance.get("archive_sha256") != digest_file(archive):
        raise ValueError("Input provenance mismatch")
    queries = json.loads(queries_path.read_text())
    if queries != prepare_queries(json.loads(gold.read_text())):
        raise ValueError("Prepared queries differ from downloaded official dev records")
    query_digest = hashlib.sha256("\n".join(sorted(q["_id"] for q in queries)).encode()).hexdigest()
    if len(queries) != 7405 or query_digest != DEV_QUERY_IDS_SHA256:
        raise ValueError("Official fullwiki query split mismatch")
    manifest = json.loads((output / "corpus_manifest.json").read_text())
    if (manifest.get("official_archive_verified") is not True or
            manifest.get("archive_record_binding") != "streamed_archive_records_v1" or
            manifest.get("gold_sha256") != provenance["gold_sha256"] or
            manifest.get("archive_sha256") != provenance["archive_sha256"]):
        raise ValueError("Prepared manifest does not bind the official inputs")
    files = sorted(p.name for p in output.iterdir() if p.suffix in {".txt", ".md"})
    _validate_staged_snapshot(files, _load_corpus_manifest(output), output,
                              progress=lambda count: print(f"Verified {count:,}/{len(files):,} corpus files", file=sys.stderr, flush=True))
    with sqlite3.connect((output / "sentences.sqlite3").resolve().as_uri()+"?mode=ro", uri=True) as db:
        if db.execute("SELECT count(*) FROM paragraphs").fetchone()[0] != len(files):
            raise ValueError("Sentence store paragraph count differs")
        coverage = annotation_coverage(db, queries)
        if coverage != manifest.get("annotation_coverage"):
            raise ValueError("Recorded annotation coverage differs from the official labels and corpus")
    return {"status":"ready", "protocol":"hotpotqa-fullwiki-dev-v1", "documents":len(files),
            "queries":len(queries), "corpus_manifest_fingerprint":manifest["fingerprint"],
            "annotation_coverage":coverage,
            "queries_sha256":digest_file(queries_path), "download_provenance_sha256":digest_file(provenance_path),
            "verifier_source_sha256":{str(Path(__file__).relative_to(ROOT)):digest_file(__file__),
                                      "cli/index.py":digest_file(ROOT / "cli/index.py")}}


if __name__ == "__main__":
    main()
