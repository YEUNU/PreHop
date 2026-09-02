"""Verify the final submission claims against complete-split artifacts.

This check is intentionally narrow: it validates the immutable evidence used
by the manuscript and scans the presentation for wording that would overstate
the executed controls. It does not rerun model inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS = {
    "multihoprag_gate": Path("data/results/final-multihoprag-performance-gate-20260831.json"),
    "musique_gate": Path("data/results/final-musique-performance-gate-20260831.json"),
    "coverage": Path("data/results/presentation-p0-analysis/gold_hop_coverage.json"),
    "graph_effect": Path("data/results/presentation-p0-analysis/graph_shortcut_effect_2417.json"),
    "components": Path("data/results/presentation-full-analysis/full_component_controls_2417.json"),
    "order": Path("data/results/presentation-full-analysis/frozen_candidate_order_replay_2417.json"),
    "rank": Path("data/results/presentation-full-analysis/frozen_rank_variants_2417.json"),
    "stages": Path("data/results/presentation-full-analysis/full_stage_profile_2417.json"),
}

DOCUMENTS = (
    Path("README.md"),
    Path("docs/prehop_paper.md"),
    Path("docs/RESULTS.md"),
    Path("docs/ABLATION_STUDY.md"),
    Path("docs/CONSISTENCY_AUDIT.md"),
    Path("presentation/prehop-academic-v2.html"),
    Path("presentation/prehop-professor-briefing.html"),
)

PRESENTATIONS = (
    Path("presentation/prehop-academic-v2.html"),
    Path("presentation/prehop-professor-briefing.html"),
)

PRESENTATION_FORBIDDEN = (
    "미실행",
    "아직 산출",
    "아직 계산",
    "재실행 필요",
    "점수로 재정렬",
    "rerank model",
    "reranker model",
    "동일 근거 재생성",
)


def _load(path: Path) -> dict[str, Any]:
    with (ROOT / path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected an object in {path}")
    return value


def _close(actual: float, expected: float, tolerance: float = 5e-6) -> bool:
    return abs(float(actual) - expected) <= tolerance


def verify() -> dict[str, Any]:
    errors: list[str] = []
    checks: list[str] = []

    missing = [str(path) for path in (*ARTIFACTS.values(), *DOCUMENTS) if not (ROOT / path).is_file()]
    if missing:
        return {"status": "failed", "checks": checks, "errors": [f"Missing {path}" for path in missing]}

    mhr_gate = _load(ARTIFACTS["multihoprag_gate"])
    mus_gate = _load(ARTIFACTS["musique_gate"])
    for name, gate, dataset, metric_count in (
        ("MultiHop-RAG", mhr_gate, "multihoprag", 4),
        ("MuSiQue", mus_gate, "musique", 5),
    ):
        if gate.get("dataset") != dataset or gate.get("evaluation_scope") != "full_benchmark":
            errors.append(f"{name} gate has the wrong dataset or scope")
        if gate.get("paper_eligible") is not True or gate.get("pass") is not True:
            errors.append(f"{name} gate is not paper eligible")
        if len(gate.get("metrics", [])) != metric_count or not all(row.get("pass") for row in gate.get("metrics", [])):
            errors.append(f"{name} gate metric rows are incomplete")
    checks.append("complete-system eligibility gates")

    coverage = _load(ARTIFACTS["coverage"])
    expected_depths = {"2hop": 1252, "3hop": 760, "4hop": 405}
    observed_depths = {
        depth: int(coverage.get("by_depth", {}).get(depth, {}).get("queries", -1)) for depth in expected_depths
    }
    if observed_depths != expected_depths or sum(observed_depths.values()) != 2417:
        errors.append(f"MuSiQue hop counts mismatch: {observed_depths}")
    if len(coverage.get("details", [])) != 2417:
        errors.append("Structural coverage does not contain 2,417 query details")
    if coverage.get("by_depth", {}).get("4hop", {}).get("gold_subgraph_connected_rate") != 0.0:
        errors.append("The 4-hop full-connectivity result is not zero")
    checks.append("MuSiQue structural hop coverage")

    graph = _load(ARTIFACTS["graph_effect"])
    groups = graph.get("groups", {})
    expected_group_counts = {"all": 2417, **expected_depths}
    for group, count in expected_group_counts.items():
        if groups.get(group, {}).get("queries") != count:
            errors.append(f"Graph effect group {group} has the wrong count")
    all_metrics = groups.get("all", {}).get("metrics", {})
    if not _close(all_metrics.get("answer_em", {}).get("effect_on_minus_off", {}).get("mean", 9), 0.0053785685):
        errors.append("Graph-on/off Answer EM effect changed")
    if not _close(all_metrics.get("support_f1", {}).get("effect_on_minus_off", {}).get("mean", 9), 0.0043464948):
        errors.append("Graph-on/off Support F1 effect changed")
    checks.append("same-index graph-on/off effects")

    components = _load(ARTIFACTS["components"])
    if components.get("queries") != 2417 or components.get("latency_included") is not False:
        errors.append("Component-control scope or latency eligibility changed")
    conditions = components.get("conditions", {})
    for condition in ("no_refinement", "no_candidate_order"):
        if condition not in conditions:
            errors.append(f"Missing component condition {condition}")
    checks.append("query-stage component controls")

    order = _load(ARTIFACTS["order"])
    if order.get("status") != "completed" or order.get("valid_records") != 2417 or order.get("failed_records") != 0:
        errors.append("Frozen candidate-order replay is incomplete")
    calibrated = order.get("order_effect_beyond_same_order_variability", {}).get(
        "hash_shuffle_vs_search_replay", {}
    ).get("jaccard")
    if calibrated is None or not _close(calibrated, -0.3285086362):
        errors.append("Frozen order calibrated Jaccard changed")
    checks.append("frozen candidate input-order replay")

    rank = _load(ARTIFACTS["rank"])
    if rank.get("queries") != 2417:
        errors.append("Frozen rank analysis does not cover 2,417 queries")
    decay_zero = rank.get("variants", {}).get("graph_decay_zero_fused", {}).get(
        "paired_vs_fused", {}
    ).get("paragraph_support_f1", {}).get("mean")
    if decay_zero is None or not _close(decay_zero, 0.0045669):
        errors.append("Frozen rank decay-0 effect changed")
    checks.append("frozen deterministic rank variants")

    stages = _load(ARTIFACTS["stages"])
    if stages.get("queries") != 2417 or stages.get("declared_concurrency") != 32:
        errors.append("Stage profile scope or concurrency changed")
    if stages.get("timing_eligible") is not True or stages.get("cross_run_absolute_timing_eligible") is not False:
        errors.append("Stage profile timing eligibility changed")
    generation_share = stages.get("mean_generation_share_of_accounted_stages")
    if generation_share is None or not _close(generation_share, 0.8485297):
        errors.append("Stage profile generation share changed")
    checks.append("fixed-concurrency stage profile")

    presentation = "\n".join((ROOT / path).read_text(encoding="utf-8") for path in PRESENTATIONS)
    for phrase in PRESENTATION_FORBIDDEN:
        if phrase.casefold() in presentation.casefold():
            errors.append(f"Presentation contains forbidden wording: {phrase}")
    for required in (
        "A1–A3는 같은 4B 통제 색인에서 질의 단계를 비교하고",
        "저장 연결 구조",
        "Ablation 1",
        "Ablation 2",
        "Ablation 3",
        "Ablation 4",
        "한 단계 확장 켬/끔",
        "후보 선택 정책",
        "MuSiQue 2,417개",
        "기록 합계",
        "qwen3-embedding-8b",
        "qwen3-embedding-4b",
    ):
        if required not in presentation:
            errors.append(f"Presentation is missing required wording: {required}")
    checks.append("presentation stage boundaries and claim wording")

    document_text = "\n".join((ROOT / path).read_text(encoding="utf-8") for path in DOCUMENTS)
    for required in ("0.9268", "0.9494", "0.4150", "0.5115", "0.00435", "84.9%"):
        if required not in document_text:
            errors.append(f"Submission documents are missing canonical value {required}")
    checks.append("canonical values in submission documents")

    return {
        "status": "passed" if not errors else "failed",
        "checks": checks,
        "errors": errors,
        "artifact_paths": {name: str(path) for name, path in ARTIFACTS.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path, relative to the repository root")
    args = parser.parse_args()
    result = verify()
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
