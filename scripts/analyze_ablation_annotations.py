"""Report independent edge annotations, agreement and explicit adjudication."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

FIELDS = ("supplies_source_question", "meaningful_transition")
LABELS = {"yes", "no", "unclear", "not_applicable"}


def load(path, expected):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    mapped = {r["sample_id"]: r for r in rows}
    if len(rows) != len(mapped) or set(mapped) != expected:
        raise ValueError("Annotation IDs must match the entire sample exactly")
    for row in rows:
        for field in FIELDS:
            if row[field] not in LABELS or (field == "meaningful_transition" and row[field] == "not_applicable"):
                raise ValueError("Every annotation requires a permitted explicit label")
    return mapped


def compare(first, second, adjudicated):
    if not first or set(first) != set(second) or set(first) != set(adjudicated):
        raise ValueError("Require identical nonempty sample IDs in all annotations")
    result = {}
    n = len(first)
    for field in FIELDS:
        a = Counter(r[field] for r in first.values())
        b = Counter(r[field] for r in second.values())
        observed = sum(first[k][field] == second[k][field] for k in first) / n
        expected = sum(a[label]*b[label] for label in LABELS)/(n*n)
        result[field] = {"annotator_1": dict(a), "annotator_2": dict(b),
                         "agreement": observed, "cohen_kappa": (observed-expected)/(1-expected) if expected < 1 else None,
                         "adjudicated": dict(Counter(r[field] for r in adjudicated.values()))}
    return {"sample_size": n, "fields": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--annotator-1", type=Path, required=True)
    p.add_argument("--annotator-2", type=Path, required=True)
    p.add_argument("--adjudicated", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    items = json.loads(args.sample.read_text())["items"]
    expected = {r["sample_id"] for r in items}
    if len(expected) != len(items):
        raise ValueError("Duplicate sampled edge IDs")
    result = compare(*(load(path, expected) for path in (args.annotator_1, args.annotator_2, args.adjudicated)))
    result["input_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (args.sample, args.annotator_1, args.annotator_2, args.adjudicated)}
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
