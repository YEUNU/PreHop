"""HotpotQA 데이터셋 준비 (Yang et al., 2018), distractor 설정.

MultiHop-RAG(`prepare_multihoprag.py`)와 동일한 산출물 규약을 따른다:
  1. data/hotpotqa_corpus/*.txt   — Wikipedia 문단(context paragraph) 1개 =
     문서 1개 (인덱싱 입력). 여러 질문에서 같은 제목의 문단이 반복 등장하면
     제목 기준으로 중복 제거한다.
  2. data/hotpotqa_queries.json   — 벤치마크가 읽는 쿼리 포맷
     (dataset 마커 "hotpotqa", type별 category, supporting_facts 증거)

distractor 설정을 쓰는 이유: 각 질문마다 ~10개의 후보 문단(gold 2개 + 방해
문단)이 함께 제공되므로, MultiHop-RAG 때처럼 "여러 질문에 등장한 문단들을
모아 하나의 코퍼스로 합치는" 구성이 그대로 재사용된다. fullwiki 설정은 전체
위키피디아 검색을 전제해서 이 프로젝트의 코퍼스 규모에 맞지 않는다.

HuggingFace parquet export에서 직접 받는다(파이썬 `datasets` 라이브러리 의존
없이 pyarrow만 사용). 회사/페이지 개념이 없으므로 page_match는 사용하지
않는다. 인덱싱·벤치마크 시 `--corpus-tag hotpotqa`로 다른 데이터셋과 Neo4j
라벨을 분리한다.
"""
import argparse
import html
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import requests


PARQUET_URL = (
    "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/"
    "distractor/validation/0.parquet"
)

DATA_DIR = Path("data")
CORPUS_DIR = DATA_DIR / "hotpotqa_corpus"
QUERIES_PATH = DATA_DIR / "hotpotqa_queries.json"
RAW_PARQUET_PATH = DATA_DIR / "hotpotqa_distractor_validation.parquet"


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]', "_", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:150] or "untitled"


_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def clean_wiki_markup(text: str) -> str:
    """Strip leftover MediaWiki markup and unescape HTML entities that survive
    in a fraction of HotpotQA's source text (the HF parquet export isn't
    fully plain-text — found via a code-intent audit of indexed chunks:
    ~0.24% of hotpotqa chunks carried <nowiki>/<br>/[[wikilink]]/{{template}}
    tokens into the Q-/Q+ LLM prompt and embeddings; separately, article
    titles containing quotes/ampersands come HTML-escaped, e.g. `&quot;J&quot;
    Is for Judgment`). Applied to corpus body text, titles, and
    evidence_facts so gold facts/doc names stay consistent with what's
    actually indexed.

    A follow-up full-corpus scan (not just a chunk sample — see
    docs/CHANGELOG.md) found more residual markup the original narrow
    <nowiki>/<br> patterns missed: ruby-annotation tags (<ruby>/<rb>/<rt>/
    <rp>), <ref>/<a href>/<onlyinclude>/<section>/<small> etc. Rather than
    keep enumerating specific tags, any remaining HTML/wiki tag is now
    stripped generically (content between the angle brackets is dropped,
    text inside/around the tag is kept) after <br> is handled specially
    (needs a space, not nothing, so words don't get joined). Also strips
    bare Wikipedia citation markers ([12]) and any malformed [[/]] left
    over from a wikilink whose bracket count didn't match (both distinct
    from the well-formed-wikilink case _WIKILINK_RE already handles).
    """
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


def _download_parquet(url: str, dest: Path) -> "pq.Table":
    if dest.exists():
        print(f"Already exists: {dest}")
    else:
        print(f"Downloading {url} ...")
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"Downloaded: {dest}")
    return pq.read_table(dest)


def _rows_from_table(table: "pq.Table", limit: int) -> list[dict]:
    n = table.num_rows if limit <= 0 else min(limit, table.num_rows)
    cols = {name: table.column(name) for name in table.column_names}
    rows = []
    for i in range(n):
        row = {name: cols[name][i].as_py() for name in table.column_names}
        rows.append(row)
    return rows


def build_corpus(rows: list[dict]) -> dict[str, str]:
    """질문마다 딸려오는 context 문단들을 data/hotpotqa_corpus/*.txt로 저장.

    Returns: {paragraph_title: filename} 매핑 (참고용). 같은 제목의 문단은
    첫 등장만 채택(문단 내용은 제목당 사실상 동일).
    """
    if CORPUS_DIR.exists():
        import shutil
        shutil.rmtree(CORPUS_DIR)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    title_to_file: dict[str, str] = {}
    for row in rows:
        context = row.get("context") or {}
        titles = context.get("title") or []
        sentences_lists = context.get("sentences") or []
        for title, sentences in zip(titles, sentences_lists):
            title = clean_wiki_markup((title or "").strip())
            if not title or title in title_to_file:
                continue
            body = " ".join(clean_wiki_markup(s.strip()) for s in (sentences or []) if s and s.strip())
            if not body:
                continue

            base = sanitize_filename(title)
            name = base
            counter = 1
            while name in used_names:
                name = f"{base}_{counter}"
                counter += 1
            used_names.add(name)

            header = f"Title: {title}\n"
            (CORPUS_DIR / f"{name}.txt").write_text(f"{header}\n{body}", encoding="utf-8")
            title_to_file[title] = name

    print(f"Created {len(used_names)} corpus files in {CORPUS_DIR}")
    return title_to_file


def build_queries(rows: list[dict]) -> list[dict]:
    """HotpotQA distractor rows를 벤치마크 쿼리 포맷으로 변환.

    evidence_docs/evidence_facts는 MultiHop-RAG과 동일한 필드명을 쓴다 —
    cli/benchmark.py의 evaluate_multihoprag_response가 데이터셋에 상관없이
    이 스키마를 그대로 소비한다(코드 새로 안 만듦).
    """
    out = []
    for row in rows:
        supporting = row.get("supporting_facts") or {}
        sup_titles = supporting.get("title") or []
        sup_sent_ids = supporting.get("sent_id") or []

        context = row.get("context") or {}
        ctx_titles = context.get("title") or []
        ctx_sentences = context.get("sentences") or []
        sentences_by_title = {
            clean_wiki_markup((t or "").strip()): s for t, s in zip(ctx_titles, ctx_sentences)
        }

        evidence_docs: list[str] = []
        evidence_facts: list[str] = []
        for title, sent_id in zip(sup_titles, sup_sent_ids):
            title = clean_wiki_markup((title or "").strip())
            if title and title not in evidence_docs:
                evidence_docs.append(title)
            sentences = sentences_by_title.get(title) or []
            if 0 <= sent_id < len(sentences):
                fact = clean_wiki_markup((sentences[sent_id] or "").strip())
                if fact:
                    evidence_facts.append(fact)

        qtype = row.get("type", "unknown")  # "bridge" | "comparison"
        out.append({
            "_id": f"hotpotqa_{row.get('id', '')}",
            "query": (row.get("question") or "").strip(),
            "ground_truth": (row.get("answer") or "").strip(),
            "evidence_docs": evidence_docs,
            "evidence_facts": evidence_facts,
            "evidence_doc": evidence_docs[0] if evidence_docs else "",
            "evidence_page": None,
            "evidence_text": evidence_facts[0] if evidence_facts else "",
            "category": qtype,
            "question_type": qtype,
            "level": row.get("level", ""),
            "dataset": "hotpotqa",
        })

    with open(QUERIES_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"Created {len(out)} queries in {QUERIES_PATH}")
    return out


def print_stats(queries: list[dict]):
    print("\n=== HotpotQA Statistics ===")
    print(f"Total queries: {len(queries)}")
    type_counts: dict[str, int] = {}
    for q in queries:
        type_counts[q["question_type"]] = type_counts.get(q["question_type"], 0) + 1
    print("\nQuestion types:")
    for qt, count in sorted(type_counts.items()):
        print(f"  - {qt}: {count}")


def main():
    parser = argparse.ArgumentParser(description="HotpotQA(distractor) 데이터셋 준비")
    parser.add_argument("--skip-corpus", action="store_true",
                        help="코퍼스 디렉토리 생성 건너뛰기 (쿼리만 갱신)")
    parser.add_argument("--limit", type=int, default=2000,
                        help="가져올 validation row 수 (기본 2000; 0=전체 7405개)")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    table = _download_parquet(PARQUET_URL, RAW_PARQUET_PATH)
    rows = _rows_from_table(table, args.limit)
    print(f"Loaded {len(rows)} rows (limit={args.limit or 'all'})")

    if not args.skip_corpus:
        build_corpus(rows)
    else:
        print("Corpus generation skipped (--skip-corpus).")

    queries = build_queries(rows)
    print_stats(queries)
    print("\nHotpotQA data preparation complete!")


if __name__ == "__main__":
    main()
