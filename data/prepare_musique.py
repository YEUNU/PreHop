"""Prepare the answerable MuSiQue development corpus and benchmark queries."""

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import requests

QUERIES_URL = "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_dev.jsonl"

DATA_DIR = Path("data")
CORPUS_DIR = DATA_DIR / "musique_corpus"
QUERIES_PATH = DATA_DIR / "musique_queries.json"
RAW_QUERIES_PATH = DATA_DIR / "musique_ans_v1.0_dev.jsonl"
DEFAULT_LIMIT = 0  # Full official answerable dev split; sampling belongs in make_sample.py.
CORPUS_MANIFEST_FILENAME = "corpus_manifest.json"


_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def clean_wiki_markup(text: str) -> str:
    """Strip leftover MediaWiki markup and unescape HTML entities that can
    survive in Wikipedia-derived source text (found via a code-intent audit
    of hotpotqa_corpus, which shares the same underlying Wikipedia source as
    musique — see prepare_hotpotqa.py's identical helper for the full
    rationale, including the follow-up full-corpus scan that generalized
    tag-stripping beyond just <nowiki>/<br> and added citation-marker /
    malformed-wikilink-bracket cleanup)."""
    if not text:
        return text
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = _WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[\[|\]\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def paragraph_identity(title: str, body: str) -> str:
    """Return a corpus-wide stable identity for one MuSiQue paragraph.

    MuSiQue's ``idx`` is query-local.  A title/body digest survives rows and
    lets every retrieval backend report an unambiguous global paragraph unit.
    The original local ``idx`` is retained separately in query metadata.
    """
    digest = hashlib.sha256(f"{title}\0{body}".encode()).hexdigest()
    return f"musique:{digest}"


def query_ids_sha256(rows: list[dict]) -> str:
    """Digest sorted prepared ``_id`` values with newline separators."""
    query_ids: list[str] = []
    for row in rows:
        if row.get("answerable") is False:
            continue
        raw_id = str(row.get("id") or "").strip()
        if not raw_id:
            raise ValueError("Answerable MuSiQue row has no stable id")
        query_ids.append(f"musique_{raw_id}")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("MuSiQue query ids are not unique")
    return hashlib.sha256("\n".join(sorted(query_ids)).encode()).hexdigest()


def _download_jsonl(url: str, dest: Path, limit: int) -> list[dict]:
    if dest.exists():
        print(f"Already exists: {dest}")
    else:
        print(f"Downloading {url} ...")
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"Downloaded: {dest}")

    rows = []
    with open(dest, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _safe_corpus_target(target: Path) -> Path:
    """Validate the narrow, explicitly configured corpus replacement target."""
    resolved = target.resolve()
    if not resolved.name or resolved == resolved.parent:
        raise ValueError(f"Unsafe corpus target: {target}")
    if target.is_symlink():
        raise ValueError(f"Refusing to replace symlinked corpus target: {target}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _gold_supporting_paragraph_ids(rows: list[dict]) -> set[str]:
    """Return all answerable-query gold paragraphs that must exist in corpus."""
    gold_ids: set[str] = set()
    for row in rows:
        if row.get("answerable") is False:
            continue
        for para in row.get("paragraphs") or []:
            if not para.get("is_supporting"):
                continue
            title = clean_wiki_markup((para.get("title") or "").strip())
            body = clean_wiki_markup((para.get("paragraph_text") or "").strip())
            if not title or not body:
                raise ValueError("Answerable MuSiQue gold paragraph has no title or body")
            gold_ids.add(paragraph_identity(title, body))
    return gold_ids


def _all_paragraph_ids(rows: list[dict]) -> set[str]:
    """Return every valid source paragraph identity represented by the corpus."""
    identities: set[str] = set()
    for row in rows:
        for para in row.get("paragraphs") or []:
            title = clean_wiki_markup((para.get("title") or "").strip())
            body = clean_wiki_markup((para.get("paragraph_text") or "").strip())
            if title and body:
                identities.add(paragraph_identity(title, body))
    return identities


def build_corpus_integrity(rows: list[dict], identity_to_file: dict[str, str], corpus_dir: Path) -> dict:
    """Validate generated corpus coverage and return a reproducible manifest.

    No expected document count is hard-coded: every invariant is derived from
    the input rows.  The resulting fingerprint is stable for equivalent input
    rows and can be retained with an indexing run to identify its corpus.
    """
    corpus_ids = set(identity_to_file)
    if len(corpus_ids) != len(identity_to_file):  # defensive even though dict keys are unique
        raise ValueError("MuSiQue paragraph identity mapping is not unique")

    filenames = list(identity_to_file.values())
    if len(filenames) != len(set(filenames)):
        raise ValueError("MuSiQue paragraph filenames are not unique")
    expected_corpus_ids = _all_paragraph_ids(rows)
    if corpus_ids != expected_corpus_ids:
        raise ValueError("Generated MuSiQue paragraph identities do not match source rows")
    missing_files = [name for name in filenames if not (corpus_dir / f"{name}.txt").is_file()]
    if missing_files:
        raise ValueError(f"Generated MuSiQue corpus is missing {len(missing_files)} paragraph file(s)")

    gold_ids = _gold_supporting_paragraph_ids(rows)
    missing_gold = gold_ids - corpus_ids
    if missing_gold:
        raise ValueError(f"Generated MuSiQue corpus misses {len(missing_gold)} gold supporting paragraph(s)")

    payload = {
        "schema_version": 1,
        "paragraph_count": len(corpus_ids),
        "gold_supporting_paragraph_count": len(gold_ids),
        "gold_supporting_paragraph_coverage": 1.0,
        "paragraph_ids_sha256": hashlib.sha256("\n".join(sorted(corpus_ids)).encode()).hexdigest(),
        "query_ids_sha256": query_ids_sha256(rows),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**payload, "fingerprint": fingerprint}


def _replace_corpus_safely(temp_dir: Path, target: Path) -> None:
    """Atomically publish a validated sibling directory and preserve rollback."""
    target = _safe_corpus_target(target)
    if temp_dir.parent.resolve() != target.parent:
        raise ValueError("Temporary corpus must be a sibling of the target corpus")
    if not temp_dir.is_dir() or temp_dir.is_symlink():
        raise ValueError(f"Invalid temporary corpus directory: {temp_dir}")

    backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
    moved_existing = False
    try:
        if target.exists():
            if not target.is_dir():
                raise ValueError(f"Corpus target is not a directory: {target}")
            os.replace(target, backup)
            moved_existing = True
        os.replace(temp_dir, target)
    except Exception:
        if moved_existing and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def build_corpus(rows: list[dict]) -> dict[str, str]:
    """Build and validate corpus in a temporary sibling before safe publish.

    Returns the established ``{stable_paragraph_id: filename}`` mapping.  A
    deterministic ``corpus_manifest.json`` is published next to the generated
    paragraph files and supplies a reusable integrity fingerprint.
    """
    target = _safe_corpus_target(CORPUS_DIR)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    identity_to_file: dict[str, str] = {}
    try:
        _build_corpus_files(rows, temp_dir, identity_to_file)
        integrity = build_corpus_integrity(rows, identity_to_file, temp_dir)
        (temp_dir / CORPUS_MANIFEST_FILENAME).write_text(
            json.dumps(integrity, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _replace_corpus_safely(temp_dir, target)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    print(f"Created {len(identity_to_file)} corpus files in {target} (fingerprint={integrity['fingerprint'][:12]})")
    return identity_to_file


def _build_corpus_files(rows: list[dict], corpus_dir: Path, identity_to_file: dict[str, str]) -> None:
    """Materialize unique paragraphs only inside an unpublished temp directory."""
    for row in rows:
        for para in row.get("paragraphs") or []:
            title = clean_wiki_markup((para.get("title") or "").strip())
            body = clean_wiki_markup((para.get("paragraph_text") or "").strip())
            if not title or not body:
                continue
            identity = paragraph_identity(title, body)
            if identity in identity_to_file:
                continue
            # Keep original title user-facing; use the hash for the filename
            # rather than a title suffix so it remains machine-identifiable.
            name = f"musique_{identity.removeprefix('musique:')}"
            header = f"Title: {title}\nParagraph-ID: {identity}\n"
            (corpus_dir / f"{name}.txt").write_text(f"{header}\n{body}", encoding="utf-8")
            identity_to_file[identity] = name


def _hop_category(row_id: str) -> str:
    # ids look like "2hop__460946_294723", "3hop1__...", "4hop2__..." etc.
    prefix = str(row_id or "").split("__", 1)[0]
    match = re.match(r"(\d+hop)", prefix)
    return match.group(1) if match else (prefix or "unknown")


def build_queries(rows: list[dict]) -> list[dict]:
    """Convert MuSiQue rows to the shared benchmark query schema."""
    out = []
    for row in rows:
        if row.get("answerable") is False:
            continue  # The unanswerable subset is outside this benchmark scope.

        evidence_docs: list[str] = []
        evidence_facts: list[str] = []
        evidence_paragraph_ids: list[str] = []
        evidence_paragraph_indices: list[int | str] = []
        evidence_paragraphs: list[dict[str, int | str]] = []
        for para in row.get("paragraphs") or []:
            if not para.get("is_supporting"):
                continue
            title = clean_wiki_markup((para.get("title") or "").strip())
            body = clean_wiki_markup((para.get("paragraph_text") or "").strip())
            if title and title not in evidence_docs:
                evidence_docs.append(title)
            if body:
                evidence_facts.append(body)
                identity = paragraph_identity(title, body)
                if identity not in evidence_paragraph_ids:
                    evidence_paragraph_ids.append(identity)
                    # The official idx is query-local. Preserve it for audit,
                    # but evaluate retrieval with the global stable identity.
                    if para.get("idx") is not None:
                        evidence_paragraph_indices.append(para["idx"])
                        evidence_paragraphs.append({"idx": para["idx"], "paragraph_id": identity})

        answer = (row.get("answer") or "").strip()
        qtype = _hop_category(row.get("id", ""))
        out.append(
            {
                "_id": f"musique_{row.get('id', '')}",
                "query": (row.get("question") or "").strip(),
                "ground_truth": answer,
                "answer_aliases": row.get("answer_aliases") or [],
                "evidence_docs": evidence_docs,
                "evidence_facts": evidence_facts,
                "evidence_paragraph_ids": evidence_paragraph_ids,
                "evidence_paragraph_indices": evidence_paragraph_indices,
                "evidence_paragraphs": evidence_paragraphs,
                "evidence_doc": evidence_docs[0] if evidence_docs else "",
                "evidence_page": None,
                "evidence_text": evidence_facts[0] if evidence_facts else "",
                "category": qtype,
                "question_type": qtype,
                "dataset": "musique",
            }
        )

    with open(QUERIES_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"Created {len(out)} queries in {QUERIES_PATH}")
    return out


def print_stats(queries: list[dict]):
    print("\n=== MuSiQue Statistics ===")
    print(f"Total queries: {len(queries)}")
    type_counts: dict[str, int] = {}
    for q in queries:
        type_counts[q["question_type"]] = type_counts.get(q["question_type"], 0) + 1
    print("\nHop categories:")
    for qt, count in sorted(type_counts.items()):
        print(f"  - {qt}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Prepare the answerable MuSiQue development split")
    parser.add_argument("--skip-corpus", action="store_true", help="Update queries without rebuilding corpus files")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of development rows; zero loads the complete answerable split",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    rows = _download_jsonl(QUERIES_URL, RAW_QUERIES_PATH, args.limit)
    print(f"Loaded {len(rows)} rows (limit={args.limit or 'all'})")

    if not args.skip_corpus:
        build_corpus(rows)
    else:
        print("Corpus generation skipped (--skip-corpus).")

    queries = build_queries(rows)
    print_stats(queries)
    print("\nMuSiQue data preparation complete!")


if __name__ == "__main__":
    main()
