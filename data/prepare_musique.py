"""MuSiQue 데이터셋 준비 (Trivedi et al., 2022), answerable(ans_v1.0) 설정.

MultiHop-RAG/HotpotQA(`prepare_multihoprag.py`/`prepare_hotpotqa.py`)와 동일한
산출물 규약을 따른다:
  1. data/musique_corpus/*.txt   — Wikipedia 문단(paragraph) 1개 = 문서 1개
     (인덱싱 입력). 여러 질문에서 같은 제목의 문단이 반복 등장하면 제목
     기준으로 중복 제거한다.
  2. data/musique_queries.json   — 벤치마크가 읽는 쿼리 포맷 (dataset 마커
     "musique", id 접두사(2hop/3hop/4hop)를 category로 사용, is_supporting
     문단 증거)

HotpotQA와 달리 gold 증거가 문장이 아니라 문단(paragraph) 단위로만 표시된다
(question_decomposition에 서브질문/서브답은 있지만 문장 인덱스는 없음) —
evidence_facts는 문단 전체 텍스트를 그대로 쓴다. dev split(JSONL, Git LFS)을
HuggingFace 미러(dgslibisey/MuSiQue)에서 직접 받는다. 회사/페이지 개념이
없으므로 --sample/--n, OCR, page_match는 사용하지 않는다. 인덱싱·벤치마크 시
`--corpus-tag musique`로 다른 데이터셋과 Neo4j 라벨을 분리한다.
"""
import argparse
import json
import re
from pathlib import Path

import requests


QUERIES_URL = (
    "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/"
    "musique_ans_v1.0_dev.jsonl"
)

DATA_DIR = Path("data")
CORPUS_DIR = DATA_DIR / "musique_corpus"
QUERIES_PATH = DATA_DIR / "musique_queries.json"
RAW_QUERIES_PATH = DATA_DIR / "musique_ans_v1.0_dev.jsonl"


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]', "_", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:150] or "untitled"


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


def build_corpus(rows: list[dict]) -> dict[str, str]:
    """질문마다 딸려오는 후보 문단들을 data/musique_corpus/*.txt로 저장.

    Returns: {paragraph_title: filename} 매핑 (참고용).
    """
    if CORPUS_DIR.exists():
        import shutil
        shutil.rmtree(CORPUS_DIR)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    title_to_file: dict[str, str] = {}
    for row in rows:
        for para in row.get("paragraphs") or []:
            title = (para.get("title") or "").strip()
            body = (para.get("paragraph_text") or "").strip()
            if not title or not body or title in title_to_file:
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


def _hop_category(row_id: str) -> str:
    # ids look like "2hop__460946_294723", "3hop1__...", "4hop2__..." etc.
    prefix = str(row_id or "").split("__", 1)[0]
    match = re.match(r"(\d+hop)", prefix)
    return match.group(1) if match else (prefix or "unknown")


def build_queries(rows: list[dict]) -> list[dict]:
    """MuSiQue rows를 벤치마크 쿼리 포맷으로 변환.

    evidence_docs/evidence_facts는 MultiHop-RAG/HotpotQA와 동일한 필드명을
    쓴다 — cli/benchmark.py의 evaluate_multihoprag_response가 데이터셋에
    상관없이 이 스키마를 그대로 소비한다(코드 새로 안 만듦).
    """
    out = []
    for row in rows:
        if row.get("answerable") is False:
            continue  # unanswerable 서브셋은 이번 벤치마크 범위 밖

        evidence_docs: list[str] = []
        evidence_facts: list[str] = []
        for para in row.get("paragraphs") or []:
            if not para.get("is_supporting"):
                continue
            title = (para.get("title") or "").strip()
            body = (para.get("paragraph_text") or "").strip()
            if title and title not in evidence_docs:
                evidence_docs.append(title)
            if body:
                evidence_facts.append(body)

        answer = (row.get("answer") or "").strip()
        qtype = _hop_category(row.get("id", ""))
        out.append({
            "_id": f"musique_{row.get('id', '')}",
            "query": (row.get("question") or "").strip(),
            "ground_truth": answer,
            "answer_aliases": row.get("answer_aliases") or [],
            "evidence_docs": evidence_docs,
            "evidence_facts": evidence_facts,
            "evidence_doc": evidence_docs[0] if evidence_docs else "",
            "evidence_page": None,
            "evidence_text": evidence_facts[0] if evidence_facts else "",
            "category": qtype,
            "question_type": qtype,
            "dataset": "musique",
        })

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
    parser = argparse.ArgumentParser(description="MuSiQue(answerable, dev) 데이터셋 준비")
    parser.add_argument("--skip-corpus", action="store_true",
                        help="코퍼스 디렉토리 생성 건너뛰기 (쿼리만 갱신)")
    parser.add_argument("--limit", type=int, default=2000,
                        help="가져올 dev row 수 (기본 2000; 0=전체 ~2400개)")
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
