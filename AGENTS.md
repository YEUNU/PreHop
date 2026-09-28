# Agent instructions

저장소 전체에 적용한다. 사용자의 명시적인 요청을 우선하고, 현재 코드와 실제 실행
산출물을 근거로 작업한다. 엔지니어링 변경이 연구 방법·측정에 영향을 주면 두 분야의
지침을 함께 적용한다. 먼저 [README](README.md), 변경 영역의
[방법·구조](docs/METHOD.md), 진입점·구현·관련 테스트를 확인한다.

## 1. 엔지니어링

### 구조와 설정

- 책임에 맞는 기존 모듈을 확장한다. CLI·shell·어댑터에 정책을 중복 구현하지 않는다.
  모듈 책임은 [코드 구조](docs/METHOD.md#code-ownership), 설정 원본·우선순위는
  [설정 규약](docs/SETUP.md#configuration-ownership-and-precedence)을 따른다.
- 공식 비교 방법의 pinned upstream 소스, 알고리즘, native retrieval·프롬프트·모델·budget을
  보존한다. 통합 변경은 우리 어댑터에서 한다. HopRAG는 별도 pinned runtime을 사용한다.
- 주 환경은 Python 3.12·uv이며 flat layout의 저장소 루트에서 실행한다. 의존성 원본은
  `pyproject.toml`·`uv.lock`, 별도 런타임 요구사항은 `configs/paper_runtime_requirements.json`이다.
  실행 중인 환경을 재설치·교체하지 않고, 실행 경로에 자동 설치·동기화를 추가하지 않는다.
- 공통 프로세스·로그 처리는 `scripts/campaign_runtime.py`, 환경 로딩·Python 선택은
  `scripts/runner_environment.py`와 `scripts/lib.sh`를 재사용한다. export는 `.env`보다 우선한다.
  `PYTHON_BIN`과 `UV_PROJECT_ENVIRONMENT`를 함께 설정하면 같은 환경을 가리켜야 한다.
- registry 기본값은 import 시 프로파일에 따라 바꾸지 않는다. 명시한 실행 프로파일은
  해당 처리량 설정을 덮어쓴다. 방법 의미와 처리량을 구분한다. `RAGConfig`의 import-time
  값은 Python 시작 전에 설정하고, 테스트는 실제 소비 설정·환경을 `monkeypatch`로 격리한다.
- 설정 조회는 복사된 환경을 전달하며 `os.environ`을 임시 변경·복구하지 않는다.
  실험 namespace·경로·seed는 `core/paper_policy.py`의 공통 해석을 사용한다.
- 추론은 하나의 OpenAI-compatible LiteLLM gateway를 사용한다. 공개 base URL·key·생성/임베딩
  모델 입력은 `.env.example`을 따른다. native child용 호환 변수를 공개 설정으로 늘리지 않는다.
  새 환경변수는 소비 지점·기본값을 명시하고 예제와 관련 문서를 갱신한다.
- `run_servers.sh`는 Neo4j 관리·서비스 확인만 한다. baseline 설치는 별도 준비 단계다.
  `.env`의 키·비밀번호를 출력하거나 코드·테스트·산출물에 넣지 않는다.

### 구현과 검증

- 주변 타입·비동기 구조를 따른다. Ruff 줄 길이는 120이며 `third_party/`는 제외한다.
  요청 범위 밖의 일괄 포맷 변경은 하지 않는다.
- 외부 호출은 공통 transport의 타임아웃·재시도를 따른다. 중첩 retry나 응답 품질에 따른
  유효 응답 재생성을 추가하지 않는다. 구조화 출력은 `core/structured_outputs.py`와 관련
  프롬프트의 JSON 디코딩·retry 계약을 확인한다. 파싱·소비 오류를 빈 성공으로 바꾸지 않는다.
- 스키마·프롬프트·오류 처리 변경이 출력 분포를 바꾸면 연구 방법 변경으로도 기록한다.
  동시성은 문서·청크 생산자, 클라이언트, native worker의 중첩을 확인한다. 클라이언트별
  제한은 gateway 전체 제한이 아니며, 요청 경로에 로컬 HTTP queue server는 없다.
- 원자적 텍스트 저장은 `utils/io.py::atomic_text_writer`를 재사용한다. 체크포인트는 trace를
  먼저 저장한 뒤 main JSON을 확정하고 재개 시 query ID로 연결한다. 필수 저장 오류는
  전파한다. 예외·취소에서도 소유한 task·클라이언트·worker를 정리하고 원래 오류를 보존한다.
- dispatch에 corpus/config/index-reuse 승인·검증 gate를 추가하거나 복원하지 않는다.
  실행 오류 전파와 연구 결과 검토를 구분한다.
- 변경 동작·실패 경계를 검증한다. 단위 테스트에서 Neo4j·추론 API는 mock, 파일은 `tmp_path`를
  사용한다. 문서만 변경하면 본문·링크·경로·명령·diff 검토로 충분하다.
- 준비된 환경으로 변경 영역 테스트부터 실행한다. 공통 변경에는 전체 비통합 테스트를 수행한다.

```bash
.venv/bin/python -m pytest -q tests/test_run_workflow.py tests/test_retrieval_constraints.py
.venv/bin/python -m pytest -q -m "not integration"
.venv/bin/python -m ruff check path/to/changed.py
git diff --check
```

선택한 환경이 다르면 그 Python을 사용한다. 관련 테스트는 `tests/`에서 실제 호출 경로로
찾는다. 실제 서비스 검증은 필요할 때만 `-m integration`으로 실행하고 skip을 통과로 보고하지 않는다.

### 문서와 작업 상태

- README는 사용자 진입점, AGENTS는 에이전트 지침이다. 필수 방법·준비·재현 설명은 기존
  `docs/`에 통합한다. 기능마다 문서를 추가하지 않는다. 공개 문서에 작업 일지, 임시 결과,
  정리 보고서·계획서·완료 요약을 만들지 않는다. 설명·설정값의 복사본 대신 원본을 링크한다.
- 수정 전 `git status --short`를 확인하고 사용자 변경·삭제·미추적 파일을 보존한다.
  관련 없는 변경을 되돌리거나 함께 커밋하지 않는다.
- 독립 실행은 새 `RAG_RUN_ID`와 필요한 `RAG_INDEX_NAMESPACE`를 사용한다. corpus tag는
  데이터셋, namespace는 저장 공간 식별자다. `--clear-graph`/`clear_graph`는 선택 DB의
  데이터·스키마를 전역 삭제하므로 일반 테스트나 청소에 사용하지 않는다.
- 로컬 논문 원고·그림, corpus, 결과·인덱스·checkpoint·trace, `neo4j_data/`, 준비된 환경을
  ignore라는 이유로 삭제하거나 소스에 포함하지 않는다. 실행 중 데이터 준비를 다시 하지 않는다.
- 장시간 실행은 기존 supervisor와 소유 프로세스 identity를 확인한다. 타 작업 프로세스를
  종료하지 않는다. 코드 편집이 실행 중 Python에 반영됐다고 가정하지 않는다.
- 완료 보고는 대화에서 변경 이유·범위, 실제 검증·skip·미검증 부분을 설명한다.
  기존 실패와 이번 변경으로 생긴 실패를 구분하고 근거 없이 완료를 주장하지 않는다.

## 2. 리서치

### 방법과 비교 통제

- 중심 질문은 질의 전에 만든 연결이 multi-hop 근거 검색을 얼마나 지원하는가이다.
  [현재 방법](docs/METHOD.md)과 실행 산출물을 확인하고 가설·관찰·해석을 구분한다.
  구조적 차이만으로 신규성·성능·비용 우위를 주장하지 않는다.
- HOP는 반환 ANN 후보 중 선택한 연결이며 정확한 전역 최근접 이웃, 검증된 답변,
  완성된 추론 경로가 아니다. 선택 목표 수와 실제 반환 수·중복을 구분한다.
- 표현·연결·확장·selector·reader prompt를 바꾸면 조건을 명시하고 기본 방법 결과로 표시하지 않는다.
  실행 전 가설, 기준 조건, 변경·고정 요인, query ID와 지표를 정한다. 탐색적 분석과
  사전 비교를 구분하고 좋은 실행만 골라 최종 결과로 바꾸지 않는다.
- gold answer·support는 평가에만 사용한다. 인덱싱·후보·검색·선택·support prediction에
  정답을 넣지 않는다. HopRAG도 query별 gold 문서 묶음이 아닌 전체 준비 corpus를 쓴다.
- HOP/NEXT 비교는 같은 인덱스·저장된 starts, selector 비교는 같은 후보 집합을 사용한다.
  연결 표현과 초기 검색 표현은 별개 요인이다. [재현 절차](docs/REPRODUCING.md)를 재사용한다.
- 공식 baseline의 native 조건을 유지하고 요청 복구·어댑터 개입을 기록한다.
  동일 top-k가 동일 passage 크기·근거량을 뜻하지 않는다. 검색, native reader,
  common reader 품질을 구분하고 다른 모델·프롬프트·입력량의 효과를 검색 효과로 단정하지 않는다.
- 공통 reader는 저장된 모든 passage와 순서를 보존하며 gold·이전 답변을 제외한다.
  새 top-k·토큰 절단·live 검색 대체를 넣지 않고 문맥 초과를 실행 오류로 남긴다.
  이는 노출 근거 비교이며 각 native 답변 파이프라인의 재현은 아니다.

### 데이터·평가·해석

- [데이터·평가 규약](docs/REPRODUCING.md#data-and-evaluation)의 모집단·단위를 보존한다.
  source/query ID와 내용 hash, HotpotQA 원래 제목·문장 인덱스·중복 출현 행을 유지한다.
  온전한 반환 corpus 문장을 gold와 무관하게 support로 투영하고 coverage·누락을 보고한다.
- 공식 지표와 공통 passage retrieval 진단을 분리한다. HotpotQA의 Hits/MRR/MAP/Recall을
  공식 HotpotQA 지표라 부르지 않는다. 데이터셋 점수를 하나의 평균으로 합치지 않는다.
- 기존 evaluator·answer extraction을 재사용한다. 실패 질문을 조용히 제외하지 않는다.
  terminal failure 품질은 0이며 null 실패가 retrieval 분모에 주는 영향을 명시한다.
  관측되지 않은 usage·비용·judge 값은 0점·성공으로 바꾸지 않는다.
- LLM judge는 기본적으로 꺼진 별도 보조 분석이며 공식 지표를 대체하지 않는다.
  최종 표는 명시적인 결과 경로를 `scripts/export_official_results.py`에 전달한다.
  최신 파일을 임의 선택하지 않고 집계를 모델 재실행·공식 코드 동등성 인증이라 하지 않는다.
- 실제 revision·dirty 상태, 데이터·query·원본 index 식별자, 방법·모델·알려진 revision,
  prompt·schema, profile·seed·cache·동시성을 기존 provenance로 남긴다. 과거 결과를
  현재 기본값으로 재표시하지 않는다. paper 생성의 seed 생략과 평가 seed를 구분한다.
  temperature 0이나 같은 seed만으로 생성·인덱싱의 완전 재현성을 주장하지 않는다.
- 파일 존재나 `admitted` receipt를 의미적 호환성·연구 타당성·게재 적합성 인증으로 해석하지 않는다.
  입력·방법이 바뀌면 새 ID를 쓰고, reader만 바꾸면 원본을 보존하며 저장 근거를 재생한다.
- [측정 규약](docs/REPRODUCING.md#measurement-definitions)의 index wall time, batch당 평균,
  개별 latency, resume segment, clone/cache recovery, 공통 reader 시간을 구분한다.
  실패 시도 비용을 보존하고 재개·부분 실행을 연속 전체 처리량으로 보고하지 않는다.
  Neo4j 논리 용량과 파일 물리 bytes를 같은 저장량으로 순위화하지 않는다.
- 같은 serving/load 조건에서 시간을 비교한다. 품질은 같은 query ID의 paired 분석을 사용하고
  HotpotQA는 원래 질문 단위 cluster bootstrap으로 중복 의존성을 반영한다. query bootstrap이
  생성·인덱스 변동성이나 조건 선택 편향까지 설명한다고 하지 않는다.
- 결과·실패 사례를 함께 제시하고 표본·전체, 관측·추정, 상관·인과를 구분한다.
  진행 카운터·임시 분석은 생성 산출물에, 재사용할 방법·평가 정의만 공개 문서에 둔다.
