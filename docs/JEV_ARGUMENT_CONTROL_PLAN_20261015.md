# 도구 인자 생성 최소화: 재검토안

2026-10-15. 상태: 사용자 확정 전 제안. 기존 실험은 중단 유지. 과거 Q3 결과와 중단된 v2 라벨을 새 학습에 사용하지 않는다.

## 결정할 설계

기존 다섯 도구+FINAL의 함수, 스키마, 검색 동작과 G1 실제 이미지 전달을 유지한다. 변경 대상은 선택된 도구의 인자를 만드는 경로다. 첫 버전은 **관측한 값으로 구성한 완전한 인자 후보를 선택하거나 기존 생성 경로를 선택하는 하이브리드**로 한다. 후보를 고르면 Python이 원래 스키마의 JSON을 구성한다. 후보 밖 인자가 필요하면 GENERATE_ARGS로 기존 생성 경로를 사용한다. FINAL 답변 생성은 유지한다.

- read_chunk: 현재까지 실제 tool 결과에 노출된 chunk ID와 previous/next ID의 단일 후보 및 최근 검색 결과의 순서 보존 최대8개 묶음. figure_read는 노출된 figure ID를 같은 방식으로 취급한다. 임의 corpus ID, gold 또는 미래 관측은 후보에 넣지 않는다.
- semantic_search / image_search: 원 질문과 이전에 실제 사용한 query를 정확히 재사용하는 후보. 의미를 자동 요약/변형하지 않는다. 새 검색어는 GENERATE_ARGS에서 생성한다.
- keyword_search: 이전 실제 keywords 배열을 재사용하는 후보. 원 질문 전체를 keyword로 바꾸거나 모든 검색을 semantic으로 통합하지 않는다. 첫 keyword 호출에는 새 인자 생성 경로를 사용한다.
- 재사용 검색 후보의 기본 top_k는5. 이전 실제 인자 전체도 보존하여 top_k와 image_search document_ids를 재사용할 수 있다. 임의 top_k(1–20), 새 문서 범위, 새 다중 ID 조합은 GENERATE_ARGS로 표현 가능하다. 선택 경로는 기존 도구의 표현력을 대체하는 제한이 아니다.
- 후보는 최신 관측부터 결정적인 순서로 중복 제거하고 최대15개를 제시한다. 나머지는 생성 경로에서 접근 가능하다. 후보0개이면 선택 prefill 없이 기존 인자 생성으로 직행한다. 후보 수/내용/기원/해시를 매 step 저장한다.
- 초기 후보 선택기는 동일 VLM의 단일-token label logits를 사용한다. A–P 중 실제 후보 라벨과 GENERATE_ARGS만 비교하며 실제 tokenizer에서 단일 토큰임을 검증한다. 모델의 전체 vocab top1이 라벨이라는 가정은 하지 않는다. 이는 학습된 pointer head가 아니다. 후보 선택용 multimodal prefill과 이미지 처리를 모두 비용에 포함한다.
- 선택 후보는 dispatch 직전 원래 JSON Schema로 검증한다. 선택 오류/형식 오류는 기록하며 다른 도구로 조용히 대체하지 않는다. GENERATE_ARGS는 오류 복구가 아니라 사전에 명시한 정상 선택지다.

## 왜 이 순서인가

도구 선택 head만으로는 별도 인자 prefill과 문자열 생성이 없어지지 않는다. 기존 evidence의 ID/검색어를 정확히 복사하는 호출에서 인자 생성을 생략할 여지가 있다. 반면 검색어를 무조건 원 질문으로 고정하면 multi-hop 탐색 능력이 바뀐다. ID를 한 개로 고정하면 원래1–8개 읽기 의미가 바뀐다. 그래서 원래 인자 생성 경로와 여러 ID 묶음 능력을 남긴다.

처음부터 cached-state controller, early exit, learned pointer, 새 query writer를 함께 넣지 않는다. 어떤 생성이 생략되는지 먼저 분리 측정한다. 후보 선택 prefill이 JSON 생성 prefill보다 짧거나 싸다는 보장은 없다. 토큰 감소/형식 오류 감소를 검증하고, shared GPU에서 비교 latency 우월성은 주장하지 않는다.

## 이번 목표의 1·2단계 제안

### 1단계: 계약 확정·구현·기능 검증

사용자 설계 확정 후 별도 argument-control 모듈과 새 runner/config를 구현한다. 기존 v2 runner/core와 결과 identity는 보존한다. 기존 다섯 도구+FINAL 순서, 관측 기반 후보, GENERATE_ARGS, 1–8개 읽기, G1 실제 이미지 전달, 오류와 비용 기록을 검사한다. 실제 고정 Gemma/EmbeddingGemma를 이용해 다섯 도구와 FINAL의 기능 smoke를 수행한다. scripted action은 기능 확인 전용이며 정책 품질 증거가 아니다.

완료 증거: 계약/설정 해시, 표적 CPU 검사, 실모델 도구 호출 및 전달 이미지 해시, fallback 경로의 실제 인자 생성, 출력 manifest. 데이터·모델 복원 검증은 전제 조건이다.

### 2단계: 고정20문항 paired 인자제어 진단

새 train 순서의 first20을 사용한다. dev20은 손대지 않는다. 동일 frozen backbone과 동일 first-token 여섯 행동 선택기를 사용하는 두 조건을 각각 처음부터 독립 실행한다: (A) 원래 매번 argument JSON 생성, (B) 후보 선택+GENERATION_ARGS. 최종 답, retrieval, iteration/이미지/context budget과 decoding 설정은 동일하게 고정한다. A/B의 전체 rollout은 분기할 수 있으므로 그 차이를 숨기지 않는다. 매 질문 양 조건을 모두 저장하고 실패 문항도 포함한다.

기록: 공식 list EM/F1, 문항별 승/패와 paired 차이, 실제 인자 생성 tokens와 calls, 후보 선택 calls/tokens, GENERATE_ARGS 비율, 도구별 분포와 오류, 실제 이미지 전달 수, total prompt/completion tokens, 구성요소별 관측 시간(비교 속도 결론 금지). 인프라 오류는 중단 후 동일 identity에서 재시도하며 정책 실패와 구분한다.

고정 진단 기준: B가 A보다 총 argument generated tokens를 줄이고, 추가 schema/dispatch 오류가 없으며, 20문항 평균 F1 하락이0.05 이하이면 후속 학습 검토 대상으로 한다. 이 수치는 작은 train 진단의 설계 기준이며 비열등성·일반화 증명이 아니다. 어떤 도구/분기가 자연 발생하지 않으면 coverage 미확인으로 명시한다. 실패해도 진단 단계는 완료하되 성공했다고 보고하지 않는다. 결과를 본 뒤 같은20문항에 맞추어 기준/후보를 조정하지 않는다.

이번 2단계에는 learned action/pointer head 훈련을 포함하지 않는 제안이다. 인자 절감 경로의 실효성을 먼저 확인하고 이후 별도 확정된 단계에서 fresh60/80/100 teacher 수집, coverage gate,3-seed head 훈련과 untouched dev20 평가로 이어간다. 따라서 이번 결과를 '학습된 head 품질'이라고 부르지 않는다. 사용자가 2단계에 학습까지 포함하길 원하면 착수 전에 단계 범위를 확장한다.

## 아직 증명되지 않은 것

실제 인자 복사 가능 비율, query 재사용의 품질, 후보 선택기의 생성 경로 선택 능력, 전체 계산량 이득, 학습된 head 성능 및 연구 신규성. 이 문서는 회복되지 않은 이전 Astra 검토를 대신한 현재 코드 기반 재검토안이다.
