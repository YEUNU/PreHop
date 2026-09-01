# Submission Ablation Study

This protocol fixes the causal scope of the submission. External systems are
official-setting reference results only. Component attribution uses Prehop
internal controls on all 2,417 MuSiQue questions. The two datasets are never
pooled, and no subset result is reported.

## 1. Reporting rules

1. A query-stage control reuses the completed
   `musique_final_20260830` index and changes only the named environment key.
2. A fixed-candidate control reuses the exact stored candidate texts and
   identities and reports support metrics.
3. Paired effects join immutable query IDs and use 10,000 bootstrap resamples
   with seed 42.
4. Latency is eligible only within one fixed-concurrency run or a synchronized
   controlled pair. Resumed or mixed-load run times are excluded.
5. Structural edge coverage is read-only and retrospective. It does not
   measure edge semantic precision or prove query-time activation.
6. Hop-depth controls apply only to MuSiQue, whose prepared records identify
   2-hop, 3-hop, and 4-hop questions. MultiHop-RAG has no compatible hop-depth
   labels.

## 2. Stage map

| ID | Pipeline stage | Tested object | Changed setting | Reindexing |
|---|---|---|---|---|
| 연결 구조 | 색인 | 정답 문단 사이의 저장 연결률 | 저장 연결과 정답 문단 전수 대조 | 같은 색인 읽기 |
| Ablation 1 | 질의: 그래프 확장 | 한 단계 이웃 확장의 기여 | 확장 켬 ↔ 끔 | 같은 색인 |
| Ablation 2 | 질의: 질문 보충 | 검색 근거로 만든 추가 질문의 기여 | 초기 질문만 사용 | 같은 색인 |
| Ablation 3 | 질의: 후보 선택 | 질문 역할별 선택의 기여 | 역할별 선택 ↔ 통합 순위 상위 12개 | 같은 색인 |
| Ablation 4 | 고정 후보: 점수식 | 순위 신호와 거리 가중치의 영향 | 같은 후보에서 점수 재계산 | 같은 후보 |
| 강건성 | 고정 후보: 입력 순서 | 후보 제시 순서의 영향 | 기준 순서 ↔ 질의별 고정 섞기 | 같은 후보 |
| 단계별 시간 | 질의 전체 | 단계별 처리시간 비중 | 동시성 32 전수 실행 | 같은 실행 |

Ablation 1–3은 같은 최종 색인에서 질의 단계를 비교한다. Ablation 4와
강건성 실험은 검색된 후보를 고정한 뒤 점수 계산과 제시 순서를 각각
바꾼다. 따라서 어떤 단계에서 무엇을 바꿨는지가 결과 해석의 단위가 된다.

## 3. Executed MuSiQue controls

### 연결 구조. 저장 연결 포괄률

- **의도.** 색인 단계에서 만든 문서 간 연결이 정답 문단 사이를 얼마나
  포괄하는지 확인한다.
- **대상.** Q+와 가장 가까운 외부 문서 Q− 문단의 저장 연결.
- **실험 방식.** 최종 색인의 저장 연결을 MuSiQue 정답 문단과 전수 대조한다.
- **Data.** All 2,417 MuSiQue questions: 1,252 2-hop, 760 3-hop, and 405 4-hop.
- **Primary metric.** Gold-only graph fully connected.
- **Secondary metric.** At least one edge between gold paragraphs.
- **Baseline.** The official gold paragraph set for each query.
- **결과.** 정답 문단 전체 연결률은 2-hop 23.56%, 3-hop 1.32%, 4-hop
  0.00%였다. 하나 이상의 정답 문단 연결은 각각 23.56%, 21.97%,
  25.68%였다.
- **Artifact.**
  `data/results/presentation-p0-analysis/gold_hop_coverage.json`.

Reproduction command:

```bash
.venv/bin/python -m scripts.analyze_gold_hop_coverage \
  --queries data/musique_queries.json \
  --corpus-tag musique_final_20260830 \
  --out data/results/presentation-p0-analysis/gold_hop_coverage.json
```

### Ablation 1. Query-time graph expansion

- **의도.** 질의 단계의 한 단계 이웃 확장이 답과 근거 선택에 기여하는지
  확인한다.
- **대상.** 앞뒤 문단과 문서 간 연결을 함께 사용하는 한 단계 확장.
- **Fixed controls.** Same completed index, query IDs, query text, model IDs,
  seed, top-k, rewrite policy, ordering policy, and synthesis prompt.
- **Data.** All 2,417 MuSiQue questions, with fixed 2/3/4-hop strata.
- **Primary metrics.** Answer EM and retrieval passes.
- **Secondary metrics.** Answer F1, paragraph Support F1/Recall, and
  retrospective gold-edge subgroups.
- **Baseline.** Final graph-on run with depth 1; control sets depth 0.
- **결과.** 전체 Support F1은 +0.00435였다. Answer EM은 +0.00538,
  검색 회차는 +0.00372였으며 두 신뢰구간은 0을 포함했다. 깊이별
  Support F1 차이는 2-hop +0.00596, 3-hop +0.00543, 4-hop −0.00268이었다.
- **Artifacts.** Graph-off result under
  `data/results/prehop-no-graph-full-2417-20260831/`; paired analysis in
  `data/results/presentation-p0-analysis/graph_shortcut_effect_2417.json`.

Query-only control and analysis commands:

```bash
RAG_RUN_ID=prehop-no-graph-full-2417 \
RAG_BENCHMARK_TIMESTAMP=prehop-no-graph-full-2417 \
RAG_BENCHMARK_CONCURRENCY=4 RAG_BENCHMARK_SEEDS=42 RAG_JUDGE_ENABLED=false \
RAG_GRAPH_HOP_DEPTH=0 \
./run_benchmark.sh --model prehop \
  --queries data/musique_queries.json \
  --corpus-tag musique_final_20260830

.venv/bin/python -m scripts.analyze_graph_shortcut_effect \
  --graph-on data/results/final-musique-prehop-full-20260830/prehop/musique_final_20260830/seed_42/prehop_musique_final_20260830.json \
  --graph-off data/results/prehop-no-graph-full-2417-20260831/prehop/musique_final_20260830/seed_42/prehop_musique_final_20260830.json \
  --coverage data/results/presentation-p0-analysis/gold_hop_coverage.json \
  --out data/results/presentation-p0-analysis/graph_shortcut_effect_2417.json \
  --expected-queries 2417 \
  --scope full_split_paired_exploratory_structural_subgroups
```

### Ablation 2. 근거 기반 질문 보충

- **의도.** 검색된 문단을 보고 만든 추가 질문이 답과 근거 선택에 기여하는지
  확인한다.
- **대상.** 근거 기반 Q−/Q+ 추가 질문과 반복 검색.
- **Fixed controls.** Same index and all final settings; the initial
  role-aligned rewrite remains enabled.
- **Data.** All 2,417 MuSiQue questions.
- **Primary metrics.** Answer EM and Support F1.
- **Secondary metrics.** Answer F1, Support Recall, and candidate-set overlap.
- **Baseline.** Final `role_aligned_evidence_iterative`; control uses
  `role_aligned`.
- **Decision.** Removing refinement changes Answer EM by −0.12784 and Support
  F1 by −0.04811, with both 95% intervals below zero. Refinement is a major
  contributor to the complete pipeline.
- **Artifacts.** Control under
  `data/results/prehop-full-no-refinement-2417-20260901/`; paired report in
  `data/results/presentation-full-analysis/full_component_controls_2417.json`.

```bash
RAG_RUN_ID=prehop-full-no-refinement-2417 \
RAG_BENCHMARK_TIMESTAMP=prehop-full-no-refinement-2417 \
RAG_BENCHMARK_CONCURRENCY=4 RAG_BENCHMARK_SEEDS=42 RAG_JUDGE_ENABLED=false \
RAG_QUERY_REWRITE_VARIANT=role_aligned \
./run_benchmark.sh --model prehop \
  --queries data/musique_queries.json \
  --corpus-tag musique_final_20260830
```

### Ablation 3. 질문 역할별 후보 선택

- **의도.** 직접 근거·연결 근거·본문 후보를 구분해 고르는 정책이 통합 순위
  상위 12개를 고르는 정책보다 답과 근거 선택에 기여하는지 확인한다.
- **대상.** 질의 단계의 후보 선택 정책.
- **Fixed controls.** Same index, query text, model IDs, seed, top-k, scoring
  signals, rewrite variant, and synthesis prompt.
- **Data.** All 2,417 MuSiQue questions.
- **Primary metrics.** Answer EM and Support F1.
- **Secondary metrics.** Answer F1, Support Recall, and candidate-set overlap.
- **비교 조건.** 질문 역할별 선택 정책과 통합 순위 상위 12개 선택 정책.
- **결과.** 통합 순위 정책으로 바꾸면 Answer EM이 0.08854, Support F1이
  0.04952 낮아졌다. 두 차이의 95% 신뢰구간은 0보다 작았다.
- **Artifacts.** Control under
  `data/results/prehop-full-no-candidate-order-global-2417-20260901/`; paired report in
  `data/results/presentation-full-analysis/full_component_controls_2417.json`.

```bash
RAG_RUN_ID=prehop-full-no-ordering-2417 \
RAG_BENCHMARK_TIMESTAMP=prehop-full-no-ordering-2417 \
RAG_BENCHMARK_CONCURRENCY=4 RAG_BENCHMARK_SEEDS=42 RAG_JUDGE_ENABLED=false \
RAG_SOURCE_SELECTION_VARIANT=global \
./run_benchmark.sh --model prehop \
  --queries data/musique_queries.json \
  --corpus-tag musique_final_20260830
```

### 강건성. 후보 입력 순서

- **Hypothesis.** If the ordering call is content-based and presentation-order
  invariant, a fixed shuffle of identical candidates should preserve selected
  sets after accounting for same-order generation variability.
- **Component.** Presentation order entering the generation-model ordering
  prompt.
- **Fixed controls.** Exact question, candidate IDs, titles, texts, metadata,
  model, prompt, top-k, and shuffle seed.
- **Data.** 고정된 후보 목록 2,417개, 실패 0건.
- **Primary metric.** Selected-set Jaccard relative to same-order replay.
- **Secondary metric.** Support F1 and first-input selection rate.
- **Baseline.** Same deterministic fused input order replay.
- **Decision.** Fixed shuffle lowers calibrated Jaccard by −0.32851 and Support
  F1 by −0.00368. The call is input-order sensitive, though the 22.1% first-item
  selection rate does not indicate simple first-item copying.
- **Artifact.**
  `data/results/presentation-full-analysis/frozen_candidate_order_replay_2417.json`.

```bash
.venv/bin/python -m scripts.replay_frozen_candidate_order \
  --trace data/results/presentation-full-analysis/frozen_candidate_pools_2417.jsonl \
  --queries data/musique_queries.json \
  --benchmark data/results/prehop-full-clean-profile-c32-2417-20260901/prehop/musique_final_20260830/seed_42/prehop_musique_final_20260830.json \
  --orders search hash_shuffle \
  --seed 42 --concurrency 32 \
  --out data/results/presentation-full-analysis/frozen_candidate_order_replay_2417.json \
  --expected-traces 2417 --resume
```

### Ablation 4. 점수식 변형

- **Hypothesis.** Equal reciprocal-rank fusion and graph attenuation affect
  evidence selection, but are heuristics rather than a probability model.
- **Component.** Semantic order, representation order, HOP semantic input, and
  graph-path decay.
- **Fixed controls.** 같은 후보 문단과 문단 식별자.
- **Data.** 고정된 후보 목록 2,417개.
- **Primary metric.** Paragraph Support F1.
- **Secondary metrics.** Support precision and recall.
- **Baseline.** Equal reciprocal-rank fusion with decay 0.5.
- **Decision.** Semantic-only and representation-only reduce Support F1;
  decay 0 improves it by 0.00457 and decay 1 reduces it by 0.00870. The rule is
  consequential, and no probabilistic or generally optimal interpretation is
  supported.
- **Artifact.**
  `data/results/presentation-full-analysis/frozen_rank_variants_2417.json`.

```bash
.venv/bin/python -m scripts.analyze_full_frozen_rank_variants \
  --trace data/results/presentation-full-analysis/frozen_candidate_pools_2417.jsonl \
  --queries data/musique_queries.json \
  --benchmark data/results/prehop-full-clean-profile-c32-2417-20260901/prehop/musique_final_20260830/seed_42/prehop_musique_final_20260830.json \
  --out data/results/presentation-full-analysis/frozen_rank_variants_2417.json \
  --expected-queries 2417
```

### Timing profile. Query-stage latency

- **Hypothesis.** The dominant online cost comes from generation-model stages,
  not graph traversal.
- **Component.** Non-overlapping rewrite/refinement, retrieval, graph,
  deterministic scoring, candidate-ordering, and synthesis timers.
- **Fixed controls.** One complete run at declared concurrency 32.
- **Data.** All 2,417 MuSiQue questions; zero failed rows.
- **Primary metric.** Within-run stage share of accounted time.
- **Secondary metric.** Mean stage time.
- **Baseline.** Total of the six non-overlapping timers in the same run.
- **Decision.** Generation-model stages account for 84.9%; graph expansion is
  1.7%. The architecture shifts edge construction offline but still has a
  generation-heavy online pipeline.
- **Artifact.**
  `data/results/presentation-full-analysis/full_stage_profile_2417.json`.

```bash
.venv/bin/python -m scripts.analyze_full_stage_profile \
  --artifact data/results/prehop-full-clean-profile-c32-2417-20260901/prehop/musique_final_20260830/seed_42/prehop_musique_final_20260830.json \
  --out data/results/presentation-full-analysis/full_stage_profile_2417.json \
  --expected-queries 2417 --declared-concurrency 32
```

## 4. MultiHop-RAG 전체 성능 범위

MultiHop-RAG 전체 성능은 2,556개 질문으로 평가했다. 검색 지표는 답이 있는
2,255개를 분모로 사용하고, 답이 없는 301개는 별도 지표로 집계했다.
MuSiQue 2,417개에서 수행한 구성요소 실험은 MuSiQue 결과표에만 제시한다.

The complete-system artifact is under
`data/results/final-multihoprag-prehop-full-20260830/`; eligibility and official
metric comparisons are fixed by
`data/results/final-multihoprag-performance-gate-20260831.json`.
