# 임베더 3조건 재측정 — 진행 중

사용자 수정 지시에 따라 모든 조건을 새로 측정 중이다. 이전 15회·혼합 임베더 결과는 현재 비교에 사용하지 않는다.

## 1. 배경과 목적

같은 A-RAG와 VLM에서 Qwen을 Google 텍스트 또는 멀티모달 임베더로 바꾼 효과를 확인한다.

## 2. 방법

| 조건 | 텍스트·캡션 검색 | 이미지 자체 검색 | 근거 읽기 |
|---|---|---|---|
| Qwen | Qwen | 없음 | 연결 이미지 → 동일 VLM |
| Google text-only | Google | 없음 | 연결 이미지 → 동일 VLM |
| Google multimodal | Google | Google | 연결 이미지 → 동일 VLM |

Reader는 Gemma 4 E4B, 파서는 MinerU2.5-Pro-2604-1.2B. 세 조건의 도구·프롬프트·이미지 예산이 같으며 A-RAG 에이전트 루프를 최대 8회 반복한다. 한도 도달 후 최종 답변 호출은 별도이며 Reader VLM 호출 수 자체를 제한하지 않는다. Qwen+Google 조합과 정답 근거 직접 제공 조건은 제외한다.

## 3. 벤치마크

M3DocVQA 개발 50문항: PDF의 텍스트·표·이미지를 연결해 답한다. Mem-Gallery 개발 10문항: 과거 이미지 포함 대화에서 근거를 찾는다. Mem-Gallery 질문은 세 조건 모두 텍스트만 입력하고 질문 첨부 이미지는 제외한다. 검색된 대화 이미지는 VLM에 전달한다.

## 4. 평가 결과

세 조건의 재측정·채점 완료 후 EM·F1·LLM 판정·이미지 전달 결과를 갱신한다. 아직 새 결과의 우위를 판단하지 않는다.

## 5. 정성 평가

전체 결과에서 Google 멀티모달의 성능 향상이 확인되면, 동일 문항의 Qwen 실패 / Google 성공을 좌우로 비교한다. 실제 실행 이미지·근거·답변만 사용한다.

## 6. 방향과 시사점

재측정 전 결론 보류. 텍스트 임베더 교체 효과와 이미지 검색 추가 효과를 분리해 판단한다.

## 7. 참조 링크

[최신 실험 계획](../../docs/EXPERIMENT_PLAN.md) · [이전 조건 보고서](LEGACY_MIXED_REPORT.md) · [A-RAG](https://github.com/Ayanami0730/arag) · [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)
