"""Prepare the official 2WikiMultiHopQA development split.

The matrix uses the same closed-corpus convention as the other paper datasets:
the indexed corpus is the union of paragraphs supplied in the selected split,
and each question keeps its supporting paragraph/sentence evidence.  This is
deliberately not the full Wikipedia paragraph archive, which is much larger
and would change the indexing-cost question being measured.
"""

import argparse
import html
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_SOURCE = DATA_DIR / "2wikimultihop_raw" / "dev.json"
CORPUS_DIR = DATA_DIR / "2wikimultihopqa_corpus"
QUERIES_PATH = DATA_DIR / "2wikimultihopqa_queries.json"
OFFICIAL_DATA_URL = "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip"
OFFICIAL_REPOSITORY = "https://github.com/Alab-NII/2wikimultihop"

_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]', "_", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:150] or "untitled"


def clean_wiki_markup(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text)
    text = _WIKILINK_RE.sub(lambda match: match.group(2) or match.group(1), text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[\[|\]\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_rows(source: Path) -> list[dict]:
    if not source.is_file() and source.resolve() == DEFAULT_SOURCE.resolve():
        DEFAULT_SOURCE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading official 2WikiMultiHopQA data from {OFFICIAL_DATA_URL} ...")
        with urllib.request.urlopen(f"{OFFICIAL_DATA_URL}?dl=1", timeout=180) as response:
            archive = io.BytesIO(response.read())
        with zipfile.ZipFile(archive) as handle, handle.open("dev.json") as source_handle:
            DEFAULT_SOURCE.write_bytes(source_handle.read())
        source = DEFAULT_SOURCE
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"2Wiki source must be a non-empty JSON list: {source}")
    return rows


def build_corpus(rows: list[dict]) -> int:
    if CORPUS_DIR.exists():
        import shutil

        shutil.rmtree(CORPUS_DIR)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    seen_titles: set[str] = set()
    for row in rows:
        for item in row.get("context") or []:
            if not isinstance(item, list) or len(item) != 2:
                continue
            title = clean_wiki_markup(item[0])
            sentences = item[1] if isinstance(item[1], list) else []
            if not title or title in seen_titles:
                continue
            body = " ".join(clean_wiki_markup(sentence) for sentence in sentences if sentence)
            if not body:
                continue

            base = sanitize_filename(title)
            filename = base
            counter = 1
            while filename in used_names:
                filename = f"{base}_{counter}"
                counter += 1
            used_names.add(filename)
            seen_titles.add(title)
            (CORPUS_DIR / f"{filename}.txt").write_text(
                f"Title: {title}\n\n{body}", encoding="utf-8"
            )
    return len(used_names)


def build_queries(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for row in rows:
        context = row.get("context") or []
        title_to_sentences = {
            clean_wiki_markup(item[0]): item[1]
            for item in context
            if isinstance(item, list) and len(item) == 2 and isinstance(item[1], list)
        }
        evidence_docs: list[str] = []
        evidence_facts: list[str] = []
        for fact in row.get("supporting_facts") or []:
            if not isinstance(fact, list) or len(fact) != 2:
                continue
            title = clean_wiki_markup(fact[0])
            sentence_id = fact[1]
            if title and title not in evidence_docs:
                evidence_docs.append(title)
            sentences = title_to_sentences.get(title) or []
            if isinstance(sentence_id, int) and 0 <= sentence_id < len(sentences):
                text = clean_wiki_markup(sentences[sentence_id])
                if text:
                    evidence_facts.append(text)

        question_type = str(row.get("type") or "unknown")
        output.append(
            {
                "_id": f"2wikimultihopqa_{row.get('_id', '')}",
                "query": str(row.get("question") or "").strip(),
                "ground_truth": str(row.get("answer") or "").strip(),
                "evidence_docs": evidence_docs,
                "evidence_facts": evidence_facts,
                "evidence_doc": evidence_docs[0] if evidence_docs else "",
                "evidence_page": None,
                "evidence_text": evidence_facts[0] if evidence_facts else "",
                "category": question_type,
                "question_type": question_type,
                "dataset": "2wikimultihopqa",
                "source_id": row.get("_id", ""),
            }
        )
    if not output or any(not item["query"] for item in output):
        raise ValueError("2Wiki source produced an empty query or query set")
    QUERIES_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 2WikiMultiHopQA dev data")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    rows = load_rows(source)
    corpus_count = build_corpus(rows)
    queries = build_queries(rows)
    type_counts: dict[str, int] = {}
    for query in queries:
        type_counts[query["question_type"]] = type_counts.get(query["question_type"], 0) + 1
    print(f"2WikiMultiHopQA dev rows: {len(rows)}")
    print(f"2WikiMultiHopQA corpus files: {corpus_count}")
    print(f"Question types: {type_counts}")
    print(f"Queries: {QUERIES_PATH}")


if __name__ == "__main__":
    main()
