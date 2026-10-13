# 최신 실험 계획: 임베더 3조건 공정 비교

사용자 2026-10-13 수정 지시. 이 계획이 이전 혼합 임베더·V2·oracle 계획을 대체한다. 이전 설계는 EXPERIMENT_PLAN_LEGACY.md에 보존한다.

## 목적과 비교 조건

같은 VLM Reader와 같은 A-RAG 도구·프롬프트에서 Qwen 텍스트 검색, Google 텍스트 검색, Google 멀티모달 검색을 비교한다.

| 실행 코드 | 조건 | 텍스트·캡션 검색 | 이미지 벡터 검색 |
|---|---|---|---|
| B0 | Qwen text | Qwen3-Embedding-4B | 없음 |
| C1 | Google text-only | EmbeddingGemma 2 | 없음 |
| G1 | Google multimodal | EmbeddingGemma 2 | EmbeddingGemma 2 |

모든 조건은 검색된 이미지·캡션 청크를 read_chunk로 읽으면 실제 연결 이미지를 Reader에 전달한다. G1도 동일한 semantic_search와 read_chunk를 사용한다. 전용 image_search/figure_read 및 추가 검증 프롬프트는 사용하지 않는다. G1은 Google 텍스트·이미지 순위를 RRF k=60으로 결합한다. Qwen과 Google 혼합 조건 및 정답 근거 직접 제공 조건은 새 비교에서 제외한다.

## 공통 설정

- Reader: 고정 revision의 Gemma 4 E4B-it. temperature 0, seed 20261013, 출력 최대 1024 tokens, context 32768, 이미지 최대 8개.
- max iteration = 8. 최종 답변 호출을 포함한 전체 Reader 호출이 8회를 넘지 않는다. 마지막 호출은 도구 없이 최종 답변하며, 조기 종료를 허용한다.
- 동일 baseline prompt와 keyword_search / semantic_search / read_chunk. 검색 결과에는 텍스트만, 실제 이미지는 읽기 후 전달한다.
- 동일 질문·코퍼스·파싱·후보 예산. 텍스트 문장 상위 180개에서 60개 청크, 이미지 후보 60개, 기본 반환 5개. reranker 없음.
- Mem-Gallery는 모든 조건에서 질문 텍스트만 입력한다. 원래 첨부된 질문 이미지는 전부 제외하고 질문 원문·정답은 수정하지 않는다. 검색된 기억 이미지는 정상 전달한다. 원래 질문 이미지 유무별 결과를 별도 집계한다. 이는 공식 원형이 아니라 사용자 요청의 text-query 비교이다.

## 이번 재측정

직전 개발 결과를 대체하기 위해 같은 M3DocVQA 개발 50문항 및 Mem-Gallery 개발 10문항을 세 조건 모두 새로 실행·채점한다. 별도 최종 50+50문항은 개발 집합과 분리하여 유지하며 이번 재측정에 섞지 않는다.

실행 경로: /content/local_fast/lsdrag-work/results/three_arm_iter8_textquery/.
각 결과는 불변 run manifest, 입력·인덱스·프롬프트 해시, Reader 설정, 텍스트 전용 질문 정책과 코드 revision을 기록한다. Mem-Gallery gold의 question_image 표시는 원 데이터 속성이며 실제 공급 여부와 구분한다.

## 평가와 보고

EM, F1, LLM 판정 및 근거 후보/전달 recall을 같은 평가 코드로 계산한다. 주요 비교는 G1 대 B0의 문항별 EM 차이이며 paired bootstrap과 exact McNemar를 보고한다. C1은 텍스트 임베더 교체 통제 조건이다. 개발 표본에서 관찰한 차이를 최종 성능으로 단정하지 않는다.

보고서는 배경·방법·벤치마크·평가·정성 사례·시사점·참조의 7개 항목으로 유지한다. 최신 조건만 본문 표에 포함하고 과거 혼합·oracle 결과는 별도 역사 자료로 보존한다. 정성 사례는 전체 성능 개선이 확인될 때 같은 문항에서 Qwen이 틀리고 G1이 맞힌 사례만 선택한다. 왼쪽 Qwen 실패, 오른쪽 Google multimodal 성공으로 실제 전달 이미지·검색 과정·답변을 보여준다. 조건을 만족하는 사례가 없거나 향상이 불확실하면 그 상태를 그대로 쓴다.
