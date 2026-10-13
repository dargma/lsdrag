# 멀티모달 A-RAG 환경과 사전 확인

확인일: 2026-10-13. 본 문서는 실제 설치·최소 실행 결과이며 최종 성능 평가가 아니다. 실행 지시서는 [EXPERIMENT_PLAN.md](../../docs/EXPERIMENT_PLAN.md), 원시 점검 결과는 같은 디렉터리의 JSON 파일에 저장한다.

## 확인 결과

| 항목 | 실제 확인 |
|---|---|
| A-RAG | 지정 커밋 설치·import 성공, 기본 소스·설정에 rerank 없음 |
| 데이터 도구 | M3DocVQA 생성 패키지 설치·import, 공식 MMQA 메타데이터 다운로드 성공 |
| PDF 생성 | Chromium으로 개발용 183개 문서 시도, 181개 PDF 생성, 2개 HTTP 404 |
| Qwen3-Embedding-4B | GPU BF16 실행, 쿼리 지시문 적용, 2,560차원·유한 값·관련 문장 순위 확인 |
| EmbeddingGemma 2 | GPU BF16 텍스트·이미지 실행, 768차원, 합성 도형 교차 모달 검색 확인 |
| MinerU2.5-Pro 2604 | 실제 Wikipedia PDF 1페이지 파싱, 공식 save API로 구조화 결과와 그림 4개 저장 |
| Gemma 4 E4B | vLLM native tool call 성공, 이미지 열람 요청 후 전달 이미지의 색·형태 응답 성공 |

MinerU 첫 점검은 advanced tier에서 총 131.8초(모델 준비 포함), 분석 단계 114.1초였다. 이는 한 페이지와 초기 실행 결과여서 전체 처리량 추정에 직접 사용하지 않는다. 표준 hybrid tier와 여러 페이지 처리량을 후속 측정한다.

## Reader와 라이브러리 결정

기본 26B-A4B 모델의 safetensors 용량은 48.07GiB로 초기 로컬 가용 공간 약 39GiB보다 크다. 사용자가 허용한 Reader 축소에 따라 14.89GiB의 `google/gemma-4-E4B-it`를 사용한다. 크기 변경은 모든 실험에 동일 적용한다. 모델 revision은 [model_preflight.json](model_preflight.json)에 고정했다.

실행 환경은 Python 3.13.15와 사전 설치된 torch 2.11.0+cu130을 공유하는 별도 venv다. 임베딩 환경은 Transformers 5.19.0·Sentence Transformers 6.1.0, 서버 환경은 vLLM 0.20.0·Transformers 5.6.2를 사용한다. 이전 Transformers 5.18.0은 EmbeddingGemma 2 아키텍처를 인식하지 못했으며, 같은 버전으로 Gemma Reader를 vLLM 0.20에서 띄우면 이종 attention head 설정 접근 오류가 발생했다. 따라서 두 환경을 분리했다. MinerU 최초 점검은 서버·클라이언트 Transformers 5.18.0에서 수행했으므로 서버 환경 버전 고정 후 표준 tier를 재검증한다.

EmbeddingGemma의 BF16 출력은 정규화 뒤에도 반올림 때문에 벡터 norm에 약 0.2% 오차가 있었다. 검색용 저장 직전 float32로 변환하고 다시 L2 정규화한다. 차원은 축소하지 않는다.

GPU는 RTX PRO 6000 Blackwell 96GB급이지만 다른 작업의 사용량이 변한다. 외부 프로세스를 종료하지 않고 파서와 Reader를 순차 실행한다. Reader 최소 실행 설정은 32,768 컨텍스트, 동시 요청 상한 4, 이미지 상한 8, temperature 0이다. 본 평가 예산은 개발 단계에서 이 범위 안에 고정한다.

파서에서 Transformers 5.6.2를 재시험했을 때 tied `lm_head` 가중치 초기화 오류가 발생했다. 따라서 파서 전용 overlay 환경은 Transformers 5.18.0·tokenizers 0.23.1로 고정하고 Reader 환경과 분리했다. 원본 A-RAG import 테스트와 M3DocVQA PDF 유틸리티 테스트는 총 13개 통과했다.

## 이미지 전달 방식

합성 그림을 사용하는 독립 점검에서 Reader가 `figure_read(figure_id="f1")`를 실제 호출했다. 같은 이미지로 두 메시지 배치를 시험했다.

| 배치 | 결과 |
|---|---|
| tool-role content에 image_url 포함 | HTTP 500, 멀티모달 placeholder 치환 오류 |
| tool 응답 직후 user-role 이미지 메시지 | HTTP 200, 실제 그림에 대해 `Red circle` 응답 |

따라서 이미지 식별자와 도구 호출 관계를 텍스트로 연결하고, 이미지 블록은 도구 결과 뒤의 user 메시지로 전달한다. 실제 시각 인식 성공까지 검사했으며 HTTP 수락만으로 통과시키지 않았다.

실패한 tool-role 요청 이후 vLLM 0.20의 멀티모달 캐시가 불일치하는 문제도 재현됐다. 서버의 `--mm-processor-cache-gb 0` 설정으로 두 배치를 독립적으로 재시험했다. [reader_smoke.json](reader_smoke.json)에 최종 결과, [초기 실패 기록](reader_smoke_initial_cache_failure.json)에 원시 오류를 보존한다. 이는 확인한 버전 조합의 결과이며 모든 vLLM 버전에 일반화하지 않는다.

## 평가 정답과 표본

원 논문은 M3DocVQA에 **EM과 F1**을 사용한다. 공개 코드는 정규화된 답 목록을 비교하며, `answers`의 여러 항목은 정답의 대안 목록이 아니라 **하나의 정답을 이루는 복수 항목**으로 다룬다. F1은 항목 간 최적 대응으로 계산한다. 문자열 하나로 무조건 연결하거나 일부 항목 포함만으로 공식 정답 판정을 하지 않는다. [논문 4절](https://arxiv.org/html/2411.04952v1#S4), [공식 평가 코드](https://github.com/bloomberg/m3docrag/blob/29e6ac2294d6b87075a1d45b8a8df175b214248a/src/m3docrag/datasets/m3_docvqa/evaluate.py)

실제 dev 메타데이터는 2,441개이며, `supporting_context[].doc_part == "image"`인 질문은 940개다. 고정 시드와 근거 조합·질문 유형의 비례 층화로 개발 50개, 최종 후보 200개를 선정했다. 두 집합은 질문 ID뿐 아니라 정답 근거 문서 URL도 겹치지 않는다. 선정 manifest·파일 해시·층별 수는 [selection.json](../registry/selection.json)에 저장했다. 모델 성능을 이용한 선정은 하지 않았다. 문서 수집과 근거 존재 QC가 끝나기 전까지 최종 데이터 유효성은 확정하지 않는다.

원본 생성 코드 방식으로 dev의 supporting document ID를 집계하면 현재 내려받은 메타데이터에서는 **3,366개 ID, 3,255개 URL**이다. 논문의 3,368개 수치와 다르므로 논문 수치로 덮어쓰지 않고 입력 파일 해시와 함께 보고한다.

근거 ID·모달리티는 직접 얻을 수 있으나 렌더링 PDF의 정답 페이지·파싱 그림 ID는 제공되지 않는다. 선택된 개발·평가 질문에 필요한 원본 MMQA 이미지 192개를 HTTP Range로 내려받았다. 원본 ZIP은 약 2.2GiB이며 실제 전송은 약 18MiB였다. 이 이미지는 정답 매핑과 QC에만 사용하고 검색 인덱스에는 넣지 않는다.

## PDF와 파싱 품질

개발 코퍼스는 근거 문서와 무관 문서 100개를 합친 183개 ID다. 174개 문서는 본문 이미지 로드까지 통과했고, 7개는 PDF는 생성됐지만 본문 이미지 누락 검토가 필요하다. 2개 문서(Rudy Robbins, Shashi Balooja)는 현재 URL에서 HTTP 404였다. 실패 문서를 정상 코퍼스로 집계하지 않는다.

최초 페이지를 직접 렌더링해 확인했다. 인포박스의 앨범 표지와 앞뒤 캡션이 구조화 출력에 연결됐고 그림 파일도 저장됐다. UI 아이콘과 안내 배너도 이미지로 파싱되어 검색 잡음이 될 수 있다. 페이지 하단의 본문·인포박스 절단과 일반 표의 처리 품질은 추가 샘플로 점검한다. 배너 이미지 실패와 본문 이미지 실패는 DOM 위치로 구분해 기록하되, 최종 정답 그림 존재 여부는 원본 이미지 매핑으로 따로 확인한다.

MinerU 4.x의 `ParseResult.save(FileBasedDataWriter(...))`가 이미지 materialization과 참조 검증을 수행한다. `structured_content.json`의 `image_source`를 사용하며 `page_0_image_27.jpg` 같은 파일명을 코드에서 추측하지 않는다. 기본 MinerU 버전이 다른 모델 revision을 자동 선택하지 않도록 지정한 2604 모델의 로컬 서버에 명시적으로 연결한다.

## 외부 비교 기준

M3DocRAG 논문 Table 1에서 ColPali + Qwen2-VL 7B의 4페이지 설정은 전체 EM 31.4, F1 36.5이며 이미지 근거 열의 F1은 24.7이다. 전체 dev 및 해당 논문의 모달리티 집계에 대한 수치이므로 이번 이미지 관련 200개와 직접 비교하지 않는다. [원 논문 Table 1](https://arxiv.org/html/2411.04952v1#S3.T1)

## 다음 작업

표준 tier 파싱 속도·품질 검증, 누락 이미지와 404 문서 처리, 정답 그림 매핑, 공통 IR·텍스트/그림 인덱스 구축, B0 개발 실행을 진행한다. 본 평가와 최종 비교 보고서는 아직 수행하지 않았다.
