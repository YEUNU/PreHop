# AGENTS.md

이 문서는 저장소 전체에서 작업하는 에이전트의 지침이다. 사용자의 명시적인 요청을 우선하며,
현재 코드와 실행 산출물을 근거로 작업한다. 엔지니어링 변경이 연구 방법이나 측정 결과에
영향을 주면 아래 두 분야의 지침을 함께 적용한다.

Prehop은 질의가 도착하기 전에 문서의 질문 표현을 연결하여 multi-hop 검색을 지원하는
연구 프로젝트다. Prehop, Naive RAG, HopRAG, MS GraphRAG, LightRAG, GFM-RAG,
LinearRAG를 MultiHop-RAG와 HippoRAG의 HotpotQA 배포본에서 비교한다.

## 1. 엔지니어링 지침

### 코드 구조와 책임

| 위치 | 역할 |
| --- | --- |
| `main.py` | CLI, 환경 초기화, index/benchmark/synthesize 등의 실행 분기, 서비스 종료 |
| `cli/index.py`, `cli/benchmark.py` | 인덱싱 조정, 벤치마크 질의 실행·어댑터 수명 관리 |
| `core/benchmark_evaluation.py`, `core/benchmark_checkpoint.py` | 평가 집계·상태, 체크포인트 저장·재개 |
| `cli/synthesis.py`, `core/synthesis_replay.py` | 저장된 검색 결과를 이용한 답변 생성 |
| `models/prehop/graphrag.py` | 인덱싱·검색 mixin 조합과 `GraphRAG.run_workflow()` |
| `models/prehop/indexing/` | 문서 분할, Q−/Q+ 생성, 임베딩, Neo4j 쓰기, 오프라인 연결 생성 |
| `models/prehop/retrieval/` | 표현별 검색, 점수 결합, 그래프 확장, 최종 근거 선택 |
| `models/naive/`, `models/hoprag/`, `models/ms_graphrag/` | 각 비교 방법의 구현 또는 공식 런타임 어댑터 |
| `models/external_research/` | LightRAG·GFM-RAG·LinearRAG의 격리 실행과 native driver |
| `core/` | 설정, 추론 transport, 런타임·인덱스 식별, 실행 정책·계측 |
| `utils/prompts/`, `utils/metrics.py`, `utils/hotpotqa.py`, `utils/official_results.py` | 프롬프트와 평가·결과 내보내기 |
| `scripts/`, `scripts/datasets/` | 실험 실행·재개·분석, 데이터 준비 |
| `tests/`, `docs/`, `configs/` | 회귀 테스트, 설계·재현 문서, 실행·런타임 설정 |

- 먼저 [README](README.md)와 [아키텍처](docs/ARCHITECTURE.md)를 읽고, 변경할 기능의
  진입점·구현·관련 테스트를 확인한다. 과거 문서나 주석이 현재 동작과 다르면 실제 호출 경로를 추적한다.
- 책임에 맞는 기존 모듈을 확장한다. CLI, shell launcher, 개별 어댑터에 같은 정책을 중복 구현하지 않는다.
- HopRAG는 별도로 준비된 pinned runtime을 사용한다. 공식 비교 방법의 고정된 upstream
  소스는 수정하지 않고, 필요한 통합 변경은 저장소의 어댑터에서 수행한다.
- 공통 프로세스 관리·로그 처리는 `scripts/campaign_runtime.py`, 실행 환경 로딩·선택은
  `scripts/runner_environment.py`에서 관리한다. 개별 실행기에 같은 기능을 중복 구현하지 않는다.

### 설정과 실행 환경

- 주 환경은 Python 3.12와 `uv`를 사용한다. 이 저장소는 일반적인 설치형 패키지가 아닌
  flat layout이며, 명령은 저장소 루트에서 실행한다. 의존성 기준은 `pyproject.toml`과 `uv.lock`이다.
- 검색·평가 설정 해석은 `core/config.py`, 방법 목록·공통 기본값·upstream revision은
  `core/strategy_registry.py`, 유효 추론 설정은 `core/inference_transport.py`에서 관리한다.
  registry 기본값은 import 시 프로파일에 따라 바뀌지 않는다. 운영 설정은
  `core/execution_profile.py::resolved_execution_environment`에서 해석하고,
  Python 실행기의 `.env` 로딩은 `scripts/runner_environment.py`를 재사용한다.
  우선순위는 기본값 < `.env` < export < 프로파일의 해당 처리량 항목이다.
  실험별 namespace·출력 경로·seed는 `core/paper_policy.py::configure_target_environment`를
  통해 설정한다. shell이나 개별 실행기에 이 정책을 다시 작성하지 않는다.
  설정 조회는 `resolved_target_environment`의 복사본을 사용하고, 조회 중 `os.environ`을
  임시로 덮어쓰거나 복구하지 않는다.
  로컬 모델과 별도 Python 환경 요구사항은 `configs/paper_runtime_requirements.json`을 따른다.
- `RAGConfig` 등은 import 시 환경변수를 읽는다. 환경과 `RAG_EXECUTION_PROFILE`은
  Python 시작 전에 설정한다. 테스트에서는 `monkeypatch`로 환경 또는 실제 소비되는 설정을 격리한다.
- 실행 프로파일은 `core/execution_profile.py`와 `configs/execution_profiles/direct-8.json`을 따른다.
  명시적으로 선택한 프로파일은 해당 동시성 설정을 덮어쓴다. 방법의 의미와 처리량 설정을 구분한다.
- 생성·임베딩은 하나의 OpenAI-compatible LiteLLM gateway를 사용한다. 공개 입력은
  `RAG_INFERENCE_BASE_URL`, `RAG_INFERENCE_API_KEY`, `RAG_GENERATION_MODEL`,
  `RAG_EMBEDDING_MODEL`이다. native child용 호환 변수를 공개 설정으로 확장하지 않는다.
- shell launcher의 환경 로딩과 Python 선택은 `scripts/lib.sh`를 재사용한다.
  export한 값은 `.env`보다 우선한다. `PYTHON_BIN`과 `UV_PROJECT_ENVIRONMENT`를 함께
  지정하면 같은 환경을 가리켜야 한다. 실행 중인 프로세스가 쓰는 환경을 재설치하거나 교체하지 않는다.
- `run_servers.sh`는 Neo4j 관리와 외부 모델 서비스 확인을 담당하며 모델 프로세스를 시작하지 않는다.
  baseline 설치는 `scripts/setup_official_baselines.sh`의 별도 준비 단계다.
  실행 경로에 자동 의존성 설치나 동기화를 추가하지 않는다.
- 새 환경변수를 추가하면 실제 소비 지점과 기본값을 명확히 하고, 필요한 `.env.example` 및 문서를 함께 갱신한다.
  `.env`의 키·비밀번호를 출력하거나 코드·테스트·산출물에 기록하지 않는다.

새 개발 환경을 준비할 때만 다음 명령을 사용한다. 이미 준비된 환경은 그대로 사용한다.

```bash
uv sync --locked --python 3.12 --extra dev
# .env가 없는 경우에만 실행한 뒤 연결 정보를 설정한다.
cp .env.example .env
```

### 구현과 검증

- 주변 코드의 구조·타입 표기·비동기 처리 방식을 따른다. Ruff의 줄 길이 기준은 120이며
  `third_party/`는 검사 대상에서 제외된다. 요청 범위 밖의 일괄 포맷 변경은 하지 않는다.
- 외부 호출은 공통 transport와 클라이언트의 재시도·타임아웃 정책을 사용한다.
  중첩 재시도로 요청 횟수를 늘리거나, 응답 품질에 따라 유효한 응답을 다시 뽑지 않는다.
- 구조화 출력은 `core/structured_outputs.py`와 관련 프롬프트에서 관리한다.
  현재 JSON 디코딩·재시도 계약을 확인하고, 파싱·소비 오류를 빈 성공 결과로 바꾸지 않는다.
  스키마·프롬프트·오류 처리 변경이 출력 분포를 바꾸면 연구 방법 변경으로도 기록한다.
- 동시성 변경에서는 문서·청크 생산자, 모델 클라이언트, native worker의 중첩을 확인한다.
  클라이언트별 제한은 gateway 전체 제한이 아니다. 현재 요청 경로에는 로컬 HTTP queue server가 없다.
- 테스트는 변경된 동작과 실패 경계를 검증한다. Neo4j와 추론 API는 단위 테스트에서 mock하고
  파일은 `tmp_path`를 사용한다. 문서만 바꾸면 참조 경로·명령·diff 검토로 충분하다.
- 원자적 텍스트 저장은 `utils/io.py::atomic_text_writer`를 재사용한다. 벤치마크는 trace를
  먼저 저장하고 main JSON을 확정하며, 재개 시 query ID로 연결한다. 필수 체크포인트 저장
  오류는 실행 오류로 전파하고, 중단·예외에도 소유한 어댑터와 작업자를 정리한다.
- 현재 dispatch에는 별도의 corpus/config/index-reuse 검증 gate가 없다.
  리팩터링 중 이를 몰래 복원하지 않는다. 실행 오류의 전파와 연구 결과 검토는 별도로 다룬다.

검증 명령은 준비된 `.venv` 기준이다. 다른 환경을 선택했다면 그 Python을 사용한다.

```bash
# 변경 영역의 테스트부터 실행한다. 아래는 Prehop 검색 흐름 변경의 예다.
.venv/bin/python -m pytest -q tests/test_run_workflow.py tests/test_retrieval_constraints.py

# 공통 동작 변경 등 더 넓은 회귀 검증이 필요한 경우
.venv/bin/python -m pytest -q -m "not integration"

# 실제 수정한 Python 파일만 지정한다.
.venv/bin/python -m ruff check path/to/changed.py
git diff --check
```

| 변경 영역 | 우선 확인할 테스트 |
| --- | --- |
| 문서 분할·인덱싱 | `test_chunking.py`, `test_index_scheduler.py`, `test_graph_write_memory.py` |
| 검색·답변 생성 | `test_run_workflow.py`, `test_retrieval_constraints.py`, `test_abstain.py` |
| 설정·추론·스키마 | `test_configuration_resolution.py`, `test_structured_outputs.py`, `test_structured_format_retry.py` |
| 실행·재개·격리 | `test_shell_env.py`, `test_index_namespace.py`, `test_execution_without_validation.py`, `test_paper_campaign.py` |
| 평가·저장 결과 | `test_metrics.py`, `test_hotpotqa.py`, `test_official_results.py`, `test_saved_retrieval_evaluation.py` |
| 추적·공통 reader | `test_prehop_tracing.py`, `test_synthesis_replay.py` |

위 테스트 파일은 모두 `tests/` 아래에 있다. 실제 서비스를 사용하는 테스트는 `integration`
marker로 분리되어 있다. live 검증이 작업에 필요한 경우에만 준비된 환경에서
`.venv/bin/python -m pytest -q -m integration`을 실행한다.
서비스 부재나 live 단계 오류로 skip될 수 있으므로 skip을 통과로 보고하지 않는다.

### 작업 상태와 산출물 보존

- 수정 전에 `git status --short`를 확인하고 사용자의 기존 변경·삭제·미추적 파일을 보존한다.
  관련 없는 변경을 되돌리거나 함께 커밋하지 않는다.
- 독립 실행에는 새로운 `RAG_RUN_ID`와 필요한 `RAG_INDEX_NAMESPACE`를 사용한다.
  corpus tag는 데이터셋 식별자이고 namespace는 저장 공간 식별자이므로 구분한다.
- `--clear-graph`와 `--mode clear_graph`는 선택된 Neo4j DB의 데이터·스키마를 전역 삭제한다.
  일반적인 새 실험이나 테스트 준비에는 새 namespace를 사용한다. 기존 실험 그래프를 청소 목적으로 지우지 않는다.
- `data/results/`, `data/index_stats/`, `data/traces/`, `data/debug/`, 각 방법의 인덱스 출력,
  `neo4j_data/`는 로컬 실행 산출물이다. ignore되었다는 이유로 삭제하거나 소스에 포함하지 않는다.
  실행 중인 corpus, 인덱스, checkpoint와 trace는 보존한다.
- 장시간 실행은 기존 supervisor와 소유 프로세스 정보를 사용한다. 코드 편집만으로 실행 중인
  Python 프로세스가 새 코드를 읽었다고 가정하지 않으며, 다른 작업의 프로세스를 종료하지 않는다.
- 완료 보고에는 변경 이유·범위, 실제 수행한 검증과 결과, 미검증 부분을 적는다.
  기존 실패와 이번 변경으로 생긴 실패를 구분한다.

## 2. 리서치 지침

### 연구 질문과 현재 방법

- 중심 질문은 **질의 이전에 만든 연결이 multi-hop 근거 검색을 얼마나 지원하는가**이다.
  가설, 관찰된 결과, 그 결과의 해석을 구분한다. 구조적 차이만으로 신규성·성능·비용 우위를 주장하지 않는다.
- 현재 기본 Prehop은 페이지 안에서 6문장 단위로 나누고 마지막 부분 청크도 보존한다.
  청크마다 내부에서 답할 수 있는 Q−와 외부 정보가 필요한 Q+를 각각 최대 3개 생성한다.
- 인덱싱에서 Q+는 다른 source file의 Q−를 검색하고 그 소유 청크로 `HOP_ANSWER`를 만든다.
  동일 문서의 인접 청크에는 `NEXT`가 있다. HOP는 반환된 ANN 후보 중 선택한 연결이며,
  정확한 전역 최근접 이웃·검증된 답변·완성된 추론 경로를 뜻하지 않는다.
- 기본 질의 경로는 원래 질문으로 body/Q−/Q+를 검색하고, 모든 시작 청크에서 저장된 HOP와
  양방향 NEXT를 한 번 확장한 뒤, 직접 검색·확장 후보를 함께 점수화하고 LLM으로 근거를 선택한다.
  선택 요청의 목표는 최대 12개이며 실제 반환 개수·중복은 기록된 결과를 확인한다.
  질의 재작성이나 확장된 청크의 재확장은 기본 방법에 포함되지 않는다.
- 표현, 연결 방식, 확장 깊이, selector, reader prompt를 바꾸면 실험 조건을 명시한다.
  새로운 변형을 기존 기본 방법의 결과로 표시하지 않는다.

### 실험 설계와 비교 통제

- 실행 전에 가설, 기준 조건, 바꾸는 요인, 고정할 요인, 대상 query ID, 평가 지표를 정한다.
  탐색적 분석과 사전에 정한 비교를 구분하고, 좋은 결과만 골라 최종 결과로 바꾸지 않는다.
- gold answer와 supporting evidence는 평가에만 사용한다. 인덱싱, 연결 후보 구성,
  검색·근거 선택·support prediction에 정답 정보를 넣지 않는다.
  HopRAG도 query별 gold 문서 묶음이 아닌 준비된 전체 corpus를 사용한다.
- HOP/NEXT 비교에서는 같은 인덱스와 저장된 시작 후보를 사용하고 확장 조건을 바꾼다.
  selector 비교에서는 후보 집합을 고정한다. 연결 표현과 초기 검색 표현은 별개의 요인으로 다룬다.
  기존 도구는 [실험 재현](docs/REPRODUCING.md)에 정리되어 있다.
- 공식 baseline은 registry에 고정된 native retrieval·prompt·local model·budget을 유지한다.
  요청 복구나 어댑터 개입도 기록한다. 동일한 top-k가 동일한 passage 크기·근거량을 뜻하지 않는다.
- 검색 품질, native reader의 답변 품질, 공통 reader의 답변 품질을 구분한다.
  모델·프롬프트·입력량이 다른 답변 점수 차이를 검색 기법만의 효과로 해석하지 않는다.
- [공통 reader 재생](docs/SYNTHESIS_REPLAY.md)은 `details[].retrieved_sources`의 모든
  passage와 순서를 보존하고 gold 및 이전 답변을 입력에서 제외한다. 새 top-k·토큰 절단이나
  live 검색 대체를 추가하지 않는다. 문맥 한도 초과는 실행 오류로 남긴다.
  이 결과는 노출된 근거의 비교이며 각 native 답변 파이프라인의 재현은 아니다.

### 데이터와 평가 규약

| 데이터셋 | 평가 모집단과 주의사항 | 공식 지표 |
| --- | --- | --- |
| MultiHop-RAG | 609문서, 총 2,556질문. 근거가 있는 2,255질문의 retrieval과 null을 포함한 전체 QA를 구분 | Hits@4/10, MRR@10, MAP@10, QA Accuracy |
| HotpotQA (HippoRAG 배포본) | 9,221 passage, 1,000 released row, 원래 질문 944개. 공식 fullwiki 설정이 아님 | Answer·Supporting Fact·Joint의 EM/F1/precision/recall |

- corpus manifest의 source/query ID와 내용 hash를 보존한다. 실행 중 데이터 준비를 다시 하거나
  corpus를 바꾸지 않는다. HotpotQA의 중복 출현 행을 임의로 제거하지 않는다.
- HotpotQA는 원래 문서 제목과 문장 인덱스를 유지한다. 반환 passage에 온전히 포함된 corpus
  문장을 gold와 무관하게 support로 투영하며, 매핑 coverage와 누락을 보고한다.
  자세한 정의는 [HotpotQA 규약](docs/HOTPOTQA.md)을 따른다.
- 공식 지표와 공통 passage retrieval 진단 지표를 분리한다. HotpotQA에 적용한
  Hits/MRR/MAP/Recall을 공식 HotpotQA 지표라고 부르지 않는다. 단위와 분모를 명시하고
  데이터셋별 점수를 합쳐 하나의 평균으로 제시하지 않는다.
- 기존 평가 구현과 answer extraction을 재사용한다. 실패한 질문을 조용히 제외하지 않는다.
  terminal failure의 품질 점수는 0이며, null 실패가 retrieval 분모에 미치는 영향도 명시한다.
  관측되지 않은 usage·비용·judge 값은 0점이나 성공으로 바꾸지 않는다.
- LLM judge는 기본적으로 꺼진 보조 분석이다. 공식 지표를 대체하지 않으며, 사용할 때는
  별도 조건과 evaluator를 기록한다. 현재 비동기 Batch judge는 지원하지 않는다.
- 최종 표는 명시적인 결과 경로를 `scripts/export_official_results.py`에 전달하여 만든다.
  임의로 최신 파일을 고르지 않는다. 이 도구는 저장된 점수의 집계이며 모델 재실행이나
  공식 평가 코드와의 동등성 인증이 아니다.

### 재현성, 비용, 결론의 범위

- 결과에는 실제 code revision과 작업 트리 상태, 데이터·query 식별자, 원본 인덱스 경로,
  방법 설정, 모델·알려진 revision, 프롬프트·스키마, 실행 프로파일, seed·cache·동시성 조건을 남긴다.
  기존 provenance 기능을 사용하고 과거 결과를 현재 기본값으로 다시 표시하지 않는다.
- 현재 paper-mode 생성 요청은 LLM seed를 보내지 않는다. 평가·표본 seed 42와 혼동하지 않으며,
  temperature 0이나 같은 seed만으로 생성과 인덱싱의 완전한 재현성을 주장하지 않는다.
- 같은 run ID는 저장 파일의 존재에 따라 인덱스 재사용과 query 재개를 유발한다.
  파일 존재가 의미적 호환성을 보증하지 않으므로 입력·방법이 바뀐 독립 실험에는 새 ID를 사용한다.
  reader만 바꾼 경우에는 저장 근거의 재생을 활용하고 retrieval 결과의 출처를 보존한다.
- 실행 receipt의 `admitted` 같은 상태명은 실험 타당성이나 논문 게재 적합성의 인증이 아니다.
  결과를 해석할 때 모집단, terminal failure, 원본 인덱스, 측정 범위를 확인한다.
  연구 결과 검토를 이유로 실행 경로에 새로운 승인 절차를 추가하지 않는다.
- **인덱스 wall time**, **query batch wall time / 전체 질문 수**, **질문별 latency**,
  **재개 segment 시간**은 별개의 측정치다. batch당 평균 시간은 처리량의 역수이지 개별 응답 지연이 아니다.
- 인덱스 복사·재사용·cache recovery 비용은 최초 cold indexing 비용과 분리한다.
  부분·재개 실행을 중단 없는 전체 처리량으로 보고하지 않는다. 실패한 시도의 비용도 보존한다.
  공통 reader 재생 시간은 새로운 end-to-end query latency가 아니다.
- 같은 serving/load 조건에서 시간을 비교한다. Neo4j logical payload 추정과 파일 기반 실제
  byte 수를 같은 물리 저장량으로 순위화하지 않는다. trace 기록 비용은 실행 시간에 포함되지만
  trace 파일 크기는 검색 인덱스 크기와 분리한다.
- 품질 차이는 동일 query ID의 paired 분석으로 평가한다. HotpotQA는 원래 질문 ID 단위로
  cluster bootstrap하여 중복 행의 종속성을 반영한다. 질문 bootstrap은 모델 생성·인덱스 구축
  변동성이나 조건 선택 편향까지 설명하지 않는다.
- 근거가 없는 우위나 완료를 주장하지 않는다. 결과와 실패 사례를 함께 제시하고,
  표본·전체 실행, 관측·추정, 상관·인과를 구분한다. 진행 중 카운터와 임시 분석은 생성 산출물에,
  재사용 가능한 방법·평가 정의는 `docs/`에 기록한다.

실행과 측정의 자세한 범위는 [실행 프로토콜](docs/THROUGHPUT_EXECUTION.md),
환경 준비는 [런타임 요구사항](docs/RUNTIME_REQUIREMENTS.md)을 따른다.
