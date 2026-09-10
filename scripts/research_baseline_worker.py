#!/usr/bin/env python3
"""JSON-line process boundary for pinned external research drivers."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Avoid shadowing third-party ``datasets`` with scripts/datasets when pinned
# runtimes import Hugging Face datasets.
SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != SCRIPT_DIR]
sys.path.insert(0, str(ROOT))

from core.benchmark_failures import BenchmarkIntegrityError
from core.strategy_registry import RESEARCH_EXTERNAL_STRATEGIES, get_strategy
from models.external_research.drivers.base import ResearchDriver, document_rows, load_rows

RESULT_PREFIX = "__PREHOP_OFFICIAL_RESULT__="


def emit(payload: dict[str, Any]) -> None:
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def load_driver(strategy: str, official_root: Path, output_dir: Path) -> ResearchDriver:
    target = get_strategy(strategy).driver
    if not target:
        raise RuntimeError(f"{strategy} has no audited research driver")
    module_name, class_name = target.split(":", 1)
    driver = getattr(importlib.import_module(module_name), class_name)(official_root, output_dir)
    if not isinstance(driver, ResearchDriver):
        raise TypeError(f"{strategy} driver does not implement the research driver protocol")
    return driver


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=RESEARCH_EXTERNAL_STRATEGIES, required=True)
    parser.add_argument("--mode", choices=("index", "serve"), required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-tag", required=True)
    args = parser.parse_args()
    driver = None
    try:
        driver = load_driver(args.strategy, args.official_root, args.output_dir)
        staged_ids = {row["source_id"] for row in load_rows(args.output_dir)}
        if args.mode == "index":
            request = json.loads(sys.stdin.readline())
            if request.get("operation") != "index":
                raise ValueError("index worker expected operation=index")
            emit({"ok": True, "stats": driver.index() or {}})
            return 0
        for line in sys.stdin:
            try:
                request = json.loads(line)
                operation = request.get("operation")
                if operation == "ready":
                    emit({"ok": True, "ready": True})
                elif operation == "query":
                    result = driver.query(str(request["query"]))
                    answer = None
                    documents = result
                    if isinstance(result, dict):
                        documents, answer = result.get("documents"), result.get("answer")
                    emit(
                        {
                            "ok": True,
                            "documents": document_rows(args.strategy, documents, staged_ids),
                            "answer": answer,
                        }
                    )
                elif operation == "query_batch" and args.strategy in {"linear_rag", "lightrag", "gfm_rag"}:
                    results = driver.query_batch(request["queries"])
                    serialized = []
                    for result in results:
                        if isinstance(result, BenchmarkIntegrityError):
                            raise result
                        if isinstance(result, Exception):
                            serialized.append({"error": f"{type(result).__name__}: {result}"})
                        else:
                            serialized.append({"documents": document_rows(args.strategy, result["documents"], staged_ids),
                                               "answer": result["answer"]})
                    workers = (driver.engine.config.max_workers if args.strategy == "linear_rag"
                               else getattr(driver, "native_qa_max_workers", None))
                    emit({"ok": True, "native_qa_max_workers": workers, "results": serialized})
                elif operation == "shutdown":
                    driver.close()
                    emit({"ok": True})
                    return 0
                else:
                    raise ValueError(f"unknown operation: {operation}")
            except Exception as exc:  # noqa: BLE001 - worker must serialize upstream failures
                traceback.print_exc(file=sys.stderr)
                emit({"ok": False, "error": f"{type(exc).__name__}: {exc}", "failure_scope": "target" if isinstance(exc, BenchmarkIntegrityError) else "query"})
        return 0
    except Exception as exc:  # noqa: BLE001 - process boundary reports any adapter failure
        traceback.print_exc(file=sys.stderr)
        emit({"ok": False, "error": f"{type(exc).__name__}: {exc}", "failure_scope": "target" if isinstance(exc, BenchmarkIntegrityError) else "query"})
        return 1
    finally:
        if driver is not None and args.mode == "index":
            try:
                driver.close()
            except Exception:  # noqa: BLE001 - preserve the primary index failure
                traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
