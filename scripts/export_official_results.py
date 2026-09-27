"""Save dataset-specific official metrics separately from auxiliary diagnostics."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.io import _write_json
from utils.official_results import build_reports


def export(result_path: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    raw = result_path.read_bytes()
    official, diagnostics = build_reports(json.loads(raw))
    directory = output_dir or result_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, report in [("official", official), ("diagnostics", diagnostics)]:
        report.update(source_result=str(result_path), source_sha256=hashlib.sha256(raw).hexdigest())
        path = directory / f"{result_path.stem}.{suffix}.json"
        _write_json(path, report)
        paths.append(path)
    return tuple(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps({"written": [str(p) for p in export(args.result, args.output_dir)]}))


if __name__ == "__main__":
    main()
