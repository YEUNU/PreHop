"""Prepare the MultiHop-RAG corpus and benchmark queries."""

import argparse
import html
import json
import re
from pathlib import Path

import requests


def _clean(text) -> str:
    """Decode HTML entities consistently in corpus and query titles."""
    return html.unescape((text or "").strip())


# Scraped article bodies contain two recurring newsletter UI patterns.
# Independent articles begin with a Mustache block ending at the second
# `{{ /verifyErrors }}` marker. Guardian articles contain a newsletter widget
# between accessibility skip markers. Remove only these fixed patterns.
_INDEPENDENT_BOILERPLATE_RE = re.compile(r"^.*?\{\{ /verifyErrors \}\}.*?\{\{ /verifyErrors \}\}\s*", re.DOTALL)
_GUARDIAN_BOILERPLATE_RE = re.compile(r"skip past newsletter promotion.*?after newsletter promotion\s*", re.DOTALL)

# Second audit pass (real content read of indexed Q-/Q+/chunks, not just
# pattern regression checks) found two more outlet-specific boilerplate
# families, both embedded mid-article rather than as one leading block:
#   - Fox News: standalone imperative CTA lines ("CLICK HERE TO SIGN UP FOR
#     OUR ... NEWSLETTER", "CLICK HERE TO GET THE FOX NEWS APP").
#   - Sporting News: a nav-widget header ("WEEK N PPR RANKINGS:" /
#     STANDARD RANKINGS / FANTASY ADVICE / DFS, or "MORE <TOPIC>:") followed
#     by a pipe-delimited link row ("QBs | RBs | WRs | TEs | D/STs |
#     Kickers") pointing at unrelated pages — both lines stripped together.
#     "APP USERS CLICK HERE" is the same family's single-line CTA variant.
_CLICK_HERE_RE = re.compile(r"(?m)^CLICK HERE TO .*\n?")
_APP_USERS_RE = re.compile(r"(?m)^APP USERS CLICK HERE\s*\n?")
_NAV_WIDGET_RE = re.compile(r"(?m)^(?:WEEK \d+ [A-Z ]+|MORE [A-Z0-9 '/.-]+):.*\n(?:[^\n]*\|[^\n]*\n)?")


def _strip_scraper_boilerplate(body: str) -> str:
    body = _INDEPENDENT_BOILERPLATE_RE.sub("", body, count=1)
    body = _GUARDIAN_BOILERPLATE_RE.sub("", body)
    body = _CLICK_HERE_RE.sub("", body)
    body = _APP_USERS_RE.sub("", body)
    body = _NAV_WIDGET_RE.sub("", body)
    return body.strip()


# Download from the dataset location referenced by the upstream README.
CORPUS_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json"
QUERIES_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json"

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CORPUS_DIR = DATA_DIR / "multihoprag_corpus"
QUERIES_PATH = DATA_DIR / "multihoprag_queries.json"
RAW_CORPUS_PATH = DATA_DIR / "multihoprag_corpus.json"
RAW_QUERIES_PATH = DATA_DIR / "MultiHopRAG.json"


def sanitize_filename(name: str) -> str:
    """Normalize a title into a bounded filename component."""
    cleaned = re.sub(r'[\\/*?:"<>|]', "_", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:150] or "untitled"


def _is_lfs_pointer(path: Path) -> bool:
    """Return whether a cached file contains a Git LFS pointer."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore").startswith("version https://git-lfs")
    except OSError:
        return False


def _download_json(url: str, dest: Path):
    """Download and cache JSON, replacing cached Git LFS pointers."""
    if dest.exists() and not _is_lfs_pointer(dest):
        print(f"Already exists: {dest}")
    else:
        if dest.exists():
            print(f"Cached file {dest} is an LFS pointer; re-downloading.")
        print(f"Downloading {url} ...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        dest.write_text(resp.text, encoding="utf-8")
        print(f"Downloaded: {dest}")
    with open(dest, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_corpus(corpus: list[dict]) -> dict[str, str]:
    """Write news articles to text files and return title-to-filename mappings."""
    if CORPUS_DIR.exists():
        import shutil

        shutil.rmtree(CORPUS_DIR)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    title_to_file: dict[str, str] = {}
    for article in corpus:
        title = _clean(article.get("title"))
        body = _strip_scraper_boilerplate(_clean(article.get("body")))
        if not body:
            continue

        base = sanitize_filename(title or article.get("url", "untitled"))
        name = base
        counter = 1
        while name in used_names:
            name = f"{base}_{counter}"
            counter += 1
        used_names.add(name)

        header = f"Title: {title}\n"
        (CORPUS_DIR / f"{name}.txt").write_text(f"{header}\n{body}", encoding="utf-8")
        if title:
            title_to_file[title] = name

    print(f"Created {len(used_names)} corpus files in {CORPUS_DIR}")
    return title_to_file


def build_queries(queries: list[dict]) -> list[dict]:
    """Convert MultiHopRAG.json to the shared benchmark query schema."""
    out = []
    for idx, q in enumerate(queries):
        evidence = q.get("evidence_list", []) or []
        evidence_docs, evidence_facts = [], []
        for ev in evidence:
            title = _clean(ev.get("title"))
            fact = _clean(ev.get("fact"))
            if title and title not in evidence_docs:
                evidence_docs.append(title)
            if fact:
                evidence_facts.append(fact)

        qtype = q.get("question_type", "unknown")
        out.append(
            {
                "_id": f"multihoprag_{idx:05d}",
                "query": _clean(q.get("query")),
                "ground_truth": _clean(q.get("answer")),
                # Ranking metrics use facts; document recall uses document titles.
                "evidence_docs": evidence_docs,
                "evidence_facts": evidence_facts,
                # Keep singular fields for report compatibility.
                "evidence_doc": evidence_docs[0] if evidence_docs else "",
                "evidence_page": None,
                "evidence_text": evidence_facts[0] if evidence_facts else "",
                # Preserve question type for category-level aggregation.
                "category": qtype,
                "question_type": qtype,
                "dataset": "multihoprag",
            }
        )

    with open(QUERIES_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"Created {len(out)} queries in {QUERIES_PATH}")
    return out


def print_stats(queries: list[dict]):
    print("\n=== MultiHop-RAG Statistics ===")
    print(f"Total queries: {len(queries)}")
    type_counts: dict[str, int] = {}
    hop_counts = []
    for q in queries:
        type_counts[q["question_type"]] = type_counts.get(q["question_type"], 0) + 1
        hop_counts.append(len(q["evidence_docs"]))
    print("\nQuestion types:")
    for qt, count in sorted(type_counts.items()):
        print(f"  - {qt}: {count}")
    if hop_counts:
        print(
            f"\nEvidence articles per query: "
            f"min={min(hop_counts)} max={max(hop_counts)} "
            f"avg={sum(hop_counts) / len(hop_counts):.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Prepare the MultiHop-RAG dataset")
    parser.add_argument("--skip-corpus", action="store_true", help="Update queries without rebuilding corpus files")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    corpus = _download_json(CORPUS_URL, RAW_CORPUS_PATH)
    queries_raw = _download_json(QUERIES_URL, RAW_QUERIES_PATH)
    print(f"Loaded {len(corpus)} articles, {len(queries_raw)} queries")

    if not args.skip_corpus:
        build_corpus(corpus)
    else:
        print("Corpus generation skipped (--skip-corpus).")

    queries = build_queries(queries_raw)
    print_stats(queries)
    print("\nMultiHop-RAG data preparation complete!")


if __name__ == "__main__":
    main()
