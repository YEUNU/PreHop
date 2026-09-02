# Submission Consistency Audit

This audit cross-checks the method description, implementation,
complete-split artifacts, manuscript, and presentation.

## 1. Claim-to-evidence matrix

| Topic | Implementation/configuration evidence | Complete-split evidence | Presentation wording | Status |
|---|---|---|---|---|
| Q+와 근거 문단 연결 | 색인 구축 시 각 Q+를 외부 문서의 Q−와 매칭하고 해당 문단을 연결 대상으로 저장 | 저장 연결과 정답 문단의 전수 대조 | “Q+와 가장 가까운 외부 Q− 문단 연결” | verified |
| 최종 문단 선택 | 설정된 생성 모델이 후보 목록을 한 번 읽고 12개 문단을 선택 | Ablation 3과 입력 순서 강건성 실험 | “같은 생성 모델의 문단 선택” | verified |
| 검색 회차별 그래프 깊이 | `RAG_GRAPH_HOP_DEPTH` accepts 0 or 1; traversal expands one frontier | Final run records depth 1; graph-off records depth 0 | “각 검색 회차에서 깊이 1 확장” | verified |
| 저장 연결 포괄률 | 저장된 문서 간 연결을 읽어 정답 문단과 대조 | 전체 연결률: 2-hop 23.56%, 3-hop 1.32%, 4-hop 0.00% | “깊이가 커질수록 전체 연결률 하락” | verified |
| 그래프 확장과 답 정확도 | 같은 색인에서 한 단계 확장 켬/끔 비교 | ΔEM +0.00538, 95% CI [−0.00455, +0.01489] | 수치와 신뢰구간 함께 제시 | verified |
| 그래프 확장과 근거 선택 | 같은 통제 | ΔSupport F1 +0.00435, 95% CI [+0.00269, +0.00603] | “Support F1 +.00435” | verified |
| 질문 보충의 기여 | `role_aligned` retains initial rewrite but removes evidence-conditioned loop | All 2,417: ΔEM −0.12784; ΔSupport F1 −0.04811 | “질문 보충 제거 시 성능 하락” | verified |
| 후보 선택 정책 | 질문 역할별 선택과 통합 순위 상위 12개 선택을 비교 | ΔEM −0.08854; ΔSupport F1 −0.04952 | “후보 선택 정책 비교” | verified |
| 후보 입력 순서 | 같은 후보를 기준 순서와 고정 섞기 순서로 제시 | calibrated Jaccard −0.32851; ΔSupport F1 −0.00368 | “입력 순서에 민감” | verified |
| 첫 항목 선택률 | 고정 섞기 결과에서 첫 위치 선택률 계산 | 22.1%, 무작위 위치 기대값 19.3% | 두 비율을 함께 제시 | verified |
| 점수식 민감도 | 동일 후보에서 순위 신호와 거리 가중치 재계산 | decay 0은 기본값보다 Support F1 +0.00457 | “점수 구성에 따라 Support F1 변화” | verified |
| 질의 단계 시간 | 한 실행에서 여섯 단계의 겹치지 않는 타이머 기록 | 생성 단계 84.9%, 그래프 확장 1.7% | “생성 단계가 처리시간 대부분 차지” | verified |
| 데이터셋별 집계 분리 | Dataset adapters use different official metrics and denominators | Separate eligible gates and separate full-system tables | Never pool metrics or hop analyses | verified |
| 전체 데이터 사용 | Eligibility checks bind count, query digest, corpus fingerprint, and completion status | MultiHop-RAG 2,556; MuSiQue 2,417; all internal controls 2,417 | State denominators beside every table | verified |
| External comparisons | HopRAG/MS GraphRAG의 공식 설정 유지 | 공식 설정의 전체 결과 | “성능 참고값” | verified |
| 단계별 시간 | 한 실행의 단계별 타이머 합계 사용 | 동시성 32, 2,417개, 실패 0건 | 같은 실행 안의 비중 | verified |
| 결과 추적성 | 산출물에 코드, 입력, 모델, 시드, 동시성, 색인 ID 기록 | JSON 결과와 요약 파일 | 전체 수치와 산출물 연결 | verified |
| 설정 변경 경계 | 모델 버전이나 벡터 차원을 바꾸면 새 실행 ID로 처음부터 색인 | 표의 모든 수치는 해당 산출물의 설정을 유지 | 과거 수치를 새 설정으로 바꾸어 표기하지 않음 | verified |

## 2. Dataset and denominator audit

| Dataset | Full count | Retrieval/answer denominator | Hop-depth analysis | Eligible artifact |
|---|---:|---|---|---|
| MultiHop-RAG | 2,556 | Retrieval: 2,255 answerable; null: 301 separate | — | `data/results/final-multihoprag-performance-gate-20260831.json` |
| MuSiQue answerable dev | 2,417 | Answer/support: 2,417 | 2-hop 1,252; 3-hop 760; 4-hop 405 | `data/results/final-musique-performance-gate-20260831.json` |

Numbers from one row must not appear under the other dataset's metric names.
MultiHop-RAG uses Hits/MRR/MAP; MuSiQue uses Answer EM/F1 and paragraph
Support precision/recall/F1.

## 3. Stage and artifact audit

| Pipeline location | Executed test | Artifact | Output eligibility |
|---|---|---|---|
| Index, read only | Stored-edge gold coverage | `data/results/presentation-p0-analysis/gold_hop_coverage.json` | Structural claims only |
| Query, graph use | One-step neighbor expansion on vs off | `data/results/presentation-p0-analysis/graph_shortcut_effect_2417.json` | Answer, support, retrieval passes |
| Query, refinement | Iterative vs initial rewrite only | `data/results/presentation-full-analysis/full_component_controls_2417.json` | Paired complete-split diagnostic; no latency |
| Query, candidate selection | Question-role selection vs integrated top 12 | same component-control artifact | Answer and support |
| Fixed candidate pool | Reference order vs fixed shuffle | `data/results/presentation-full-analysis/frozen_candidate_order_replay_2417.json` | Selection and support |
| Fixed candidate pool | Rank signal and distance-weight variants | `data/results/presentation-full-analysis/frozen_rank_variants_2417.json` | Support |
| Query, timing | Six non-overlapping stage timers | `data/results/presentation-full-analysis/full_stage_profile_2417.json` | Within-run shares only |

## 4. Document responsibilities

| Document | Final role |
|---|---|
| `README.md` | User-facing method, commands, and compact final results |
| `docs/ARCHITECTURE.md` | Normative implementation and measurement contract |
| `docs/prehop_paper.md` | Submission manuscript: claims, method, protocol, full results, limitations |
| `docs/RESULTS.md` | Canonical number-to-artifact register |
| `docs/ABLATION_STUDY.md` | Stage-specific causal design, commands, and decision rules |
| `presentation/prehop-academic-v2.html` | Academic oral presentation |
| `docs/CHANGELOG.md` | Chronological engineering history |
| `CLAUDE.md` | Repository maintenance policy |

## 5. Mechanical acceptance checks

The submission package is internally consistent only when all of the following
hold:

1. Both performance gates record `paper_eligible=true` and exact full counts.
2. Every reported MuSiQue internal control contains 2,417 valid query IDs and
   no failed rows.
3. Structural hop counts sum to 2,417.
4. Presentation slides describe the candidate selection as one call to the
   configured generation model.
5. External official-setting comparisons are labelled point estimates rather
   than paired causal controls.
6. The PDF renders every slide without clipped titles, overflowing tables, or
   hidden notes entering the visible frame.
7. Repository tests, formatting checks, and document-link checks pass from the
   current workspace state.
8. A complete-system table does not mix model revisions or vector dimensions.
   The completed MuSiQue component controls remain scoped to their recorded
   2,560-dimensional `qwen3-embedding-4b` index; they are not presented as a
   same-configuration intervention on a separately rebuilt 4,096-dimensional
   index.
