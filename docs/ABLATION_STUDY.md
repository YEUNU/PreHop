# Submission Ablation Study

This protocol fixes the causal scope of the submission. External systems are
official-setting reference results only. Component attribution uses Prehop
internal controls on all 2,417 MuSiQue questions. The two datasets are never
pooled, and no subset result is reported.

These completed controls use the recorded `qwen3-embedding-4b`,
2,560-dimensional Prehop index. They remain evidence for that configuration.
Results from a separately rebuilt 4,096-dimensional `qwen3-embedding-8b`
index are not combined with these paired deltas as though they came from the
same controlled comparison.

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

Ablation 1–3은 같은 최종 색인에서 질의 단계만 비교한다. Ablation 4와
강건성 실험은 검색 후보를 고정하고 점수 계산과 제시 순서를 각각 바꾼다.
결과는 표에 적힌 단계와 변경 조건의 범위에서만 해석한다.

## 3. Executed MuSiQue controls

### 연결 구조. 저장 연결 포괄률

- **의도.** 색인 단계에서 만든 문서 간 연결이 정답 문단 사이를 얼마나
  포괄하는지 확인한다.
- **대상.** Q+와 가장 가까운 외부 문서 Q− 문단의 저장 연결.
- **실험 방식.** 최종 색인의 저장 연결을 MuSiQue 정답 문단과 전수 대조한다.
- **데이터.** MuSiQue 전체 2,417개: 2-hop 1,252개, 3-hop 760개,
  4-hop 405개.
- **주요 지표.** 정답 문단만으로 구성한 그래프의 전체 연결 여부.
- **보조 지표.** 정답 문단 사이에 저장 연결이 하나 이상 있는지 여부.
- **비교 기준.** 질의별 공식 정답 문단 집합.
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

### Ablation 1. 질의 단계 그래프 확장

- **의도.** 질의 단계의 한 단계 이웃 확장이 답과 근거 선택에 기여하는지
  확인한다.
- **대상.** 앞뒤 문단과 문서 간 연결을 함께 사용하는 한 단계 확장.
- **고정 조건.** 완성된 색인, 질의 ID와 본문, 모델 ID, 시드, top-k,
  질문 보충 정책, 후보 선택 정책, 답변 생성 프롬프트.
- **데이터.** MuSiQue 전체 2,417개와 고정된 2/3/4-hop 구간.
- **주요 지표.** Answer EM과 검색 회차.
- **보조 지표.** Answer F1, 문단 Support F1/Recall, 저장된 정답 연결
  유무에 따른 사후 하위집단.
- **비교 조건.** 깊이 1의 그래프 확장을 사용한 최종 실행과 깊이 0 통제.
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
- **고정 조건.** 같은 색인과 최종 설정을 사용하며, 최초 질문의 역할별
  재작성은 유지한다.
- **데이터.** MuSiQue 전체 2,417개.
- **주요 지표.** Answer EM과 Support F1.
- **보조 지표.** Answer F1, Support Recall, 후보 집합 중복률.
- **비교 조건.** 최종 `role_aligned_evidence_iterative`와 질문 보충을
  제거한 `role_aligned`.
- **결과.** 질문 보충을 제거했을 때 변화량은 Answer EM −0.12784,
  Support F1 −0.04811이었다. 두 차이의 95% 신뢰구간은 모두 음수였다.
- **산출물.** 통제 실행은
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
- **고정 조건.** 같은 색인, 질의 본문, 모델 ID, 시드, top-k, 점수 신호,
  질문 보충 방식, 답변 생성 프롬프트.
- **데이터.** MuSiQue 전체 2,417개.
- **주요 지표.** Answer EM과 Support F1.
- **보조 지표.** Answer F1, Support Recall, 후보 집합 중복률.
- **비교 조건.** 질문 역할별 선택 정책과 통합 순위 상위 12개 선택 정책.
- **결과.** 통합 순위 정책으로 바꾸면 Answer EM이 0.08854, Support F1이
  0.04952 낮아졌다. 두 차이의 95% 신뢰구간은 모두 음수였다.
- **산출물.** 통제 실행은
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

### 강건성 실험. 후보 입력 순서

- **의도.** 후보 내용이 같을 때 제시 순서가 선택 결과에 미치는 영향을
  측정한다.
- **대상.** 생성 모델의 후보 선택 프롬프트에 들어가는 문단 순서.
- **고정 조건.** 질의, 후보 ID, 제목, 본문, 메타데이터, 모델, 프롬프트,
  top-k, 섞기 시드.
- **데이터.** 고정된 후보 목록 2,417개, 실패 0건.
- **주요 지표.** 같은 순서 재실행을 기준으로 한 선택 집합 Jaccard.
- **보조 지표.** Support F1과 첫 입력 문단 선택률.
- **비교 조건.** 결정적 통합 순위와 질의별 고정 섞기 순서.
- **결과.** 고정 섞기 순서에서 변화량은 보정 Jaccard −0.32851,
  Support F1 −0.00368이었다. 입력 순서의 영향은 확인됐지만, 첫 문단 선택률
  22.1%만으로 첫 문단을 그대로 복사한다고 볼 수는 없다.
- **산출물.**
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

- **의도.** 순위 신호와 그래프 거리 가중치가 근거 선택에 미치는 영향을
  같은 후보에서 측정한다.
- **대상.** 의미 순위, 표현 순위, HOP 의미 점수 입력, 그래프 경로 감쇠.
- **고정 조건.** 같은 후보 문단과 문단 식별자.
- **데이터.** 고정된 후보 목록 2,417개.
- **주요 지표.** 문단 Support F1.
- **보조 지표.** Support precision과 recall.
- **비교 조건.** 동일 가중 reciprocal-rank fusion과 감쇠 0.5.
- **결과.** 의미 순위만 사용하거나 표현 순위만 사용하면 Support F1이
  낮아졌다. 감쇠 0에서는 0.00457 높아졌고, 감쇠 1에서는 0.00870
  낮아졌다. 이 결과는 점수식의 영향을 보여주지만 확률적 해석이나
  일반적 최적성을 뒷받침하지는 않는다.
- **산출물.**
  `data/results/presentation-full-analysis/frozen_rank_variants_2417.json`.

```bash
.venv/bin/python -m scripts.analyze_full_frozen_rank_variants \
  --trace data/results/presentation-full-analysis/frozen_candidate_pools_2417.jsonl \
  --queries data/musique_queries.json \
  --benchmark data/results/prehop-full-clean-profile-c32-2417-20260901/prehop/musique_final_20260830/seed_42/prehop_musique_final_20260830.json \
  --out data/results/presentation-full-analysis/frozen_rank_variants_2417.json \
  --expected-queries 2417
```

### 단계별 시간. 질의 처리 지연시간

- **의도.** 질의 처리시간이 어느 단계에서 발생하는지 측정한다.
- **대상.** 서로 겹치지 않는 질문 재작성·보충, 검색, 그래프 확장,
  결정적 점수 계산, 후보 선택, 답변 생성 시간.
- **고정 조건.** 동시성 32로 수행한 하나의 전체 실행.
- **데이터.** MuSiQue 전체 2,417개, 실패 0건.
- **주요 지표.** 같은 실행에서 측정한 단계별 시간 비중.
- **보조 지표.** 단계별 평균 시간.
- **비교 기준.** 같은 실행의 여섯 비중복 타이머 합계.
- **결과.** 생성 모델을 사용하는 단계가 84.9%, 그래프 확장이 1.7%를
  차지했다. 간선 구성은 오프라인에서 수행하지만 온라인 질의 경로의
  대부분은 생성 모델 처리시간이다.
- **산출물.**
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
