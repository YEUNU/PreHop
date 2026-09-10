"""HotpotQA official scoring rules and gold-independent sentence projection.

Native RAG methods do not emit sentence selections. The declared common adapter
predicts every complete corpus sentence present in returned passage text. It
never consults supporting-fact annotations to select a prediction.
"""
import json
import re
import sqlite3
import string
from collections import Counter
from pathlib import Path

METRICS = tuple(prefix + name for prefix in ("", "sp_", "joint_") for name in ("em", "f1", "prec", "recall"))
PROJECTION = "retrieved-complete-sentences-v1"


def normalize(text):
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score(answer, predicted_facts, gold_answer, gold_facts):
    a, b = normalize(answer), normalize(gold_answer)
    at, bt = a.split(), b.split()
    common = sum((Counter(at) & Counter(bt)).values())
    if a != b and ({a, b} & {"yes", "no", "noanswer"}):
        common = 0
    p = common / len(at) if at else 0.0
    r = common / len(bt) if bt else 0.0
    pred, gold = set(map(tuple, predicted_facts)), set(map(tuple, gold_facts))
    hit = len(pred & gold)
    sp = hit / len(pred) if pred else 0.0
    sr = hit / len(gold) if gold else 0.0
    harmonic = lambda x, y: 2*x*y/(x+y) if x+y else 0.0
    return {"em": float(a == b), "f1": harmonic(p,r), "prec": p, "recall": r,
            "sp_em": float(pred == gold), "sp_f1": harmonic(sp,sr), "sp_prec": sp, "sp_recall": sr,
            "joint_em": float(a == b and pred == gold), "joint_f1": harmonic(p*sp,r*sr),
            "joint_prec": p*sp, "joint_recall": r*sr}


def project_sentences(sources, store):
    """Resolve exact source/title identities, then match complete sentences.

    Unknown identities predict no support; do not infer IDs from gold or count
    an entire paragraph merely because its title was retrieved.
    """
    path = Path(store).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HotpotQA sentence mapping is missing: {path}")
    result = set()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        for source in sources:
            if not isinstance(source, dict):
                continue
            text = str(source.get("text") or source.get("content") or "")
            sid = Path(str(source.get("source") or source.get("source_id") or "")).stem
            row = db.execute("SELECT title,sentences FROM paragraphs WHERE source_id=?", (sid,)).fetchone()
            if row is None:
                title = str(source.get("title") or source.get("doc") or "")
                row = db.execute("SELECT title,sentences FROM paragraphs WHERE title=?", (title,)).fetchone()
            if row is None:
                continue
            body = " ".join(text.split())
            for idx, sentence in enumerate(json.loads(row[1])):
                needle = " ".join(sentence.split())
                if needle and needle in body:
                    result.add((row[0], idx))
    return [list(fact) for fact in sorted(result)]
