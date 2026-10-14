# Gemma decision head — 실험 진행 보고

브랜치: `jev_v1.0` · 2026-10-14 UTC

**첫 마일스톤: 로컬 Gemma의 text/image prefill → 행동 선택 → 인자/최종 답변 생성 경로를 실제 GPU에서 확인했다. 학습된 head의 정확도·속도 비교 결과는 아직 없다.**

| 단계 | 현재 상태 | 근거 |
|---|---|---|
| 평가 분리 | 새 M3 20문항 잠금, 기존 test/dev 학습 제외 | [고정 계획](../../registry/jev_v1.0/label_collection_plan.json) |
| M3 teacher | 새 train 82문항 수집 중; 각 문항 한 번 | 원래 Gemma A-RAG, EmbeddingGemma 2 텍스트+이미지 검색 |
| 보강 후보 QC | MMQA train 37/37 통과 | [QC](mmqa_train_qc.json): URL·이미지 파일/hash 존재 확인. 정답 crop 의미 정렬 검증과는 별개 |
| HF 동작 검증 | text/image 상태 각각 baseline + 임시 head 확인 | [실행 기록](hf_smoke.json) |
| 코드 검증 | 관련 테스트 26개 통과 | 라벨·미래 출력 누출 방지·typed tool parser·원래 Agent |
| head 학습 | 대기 | 성공 문항 ≥30, 각 행동 관측 성공 문항 ≥10 필요 |
| 최종 비교 | 대기 | 새 M3 20문항 × B-HF/D-head, 각 조건 한 번 |

## 확인한 동작

학습 데이터의 text 상태 1개와 image 상태 1개만 사용했다. baseline은 원래 native tool call을 생성한다. 동작 확인용 임시 head는 text에서 FINAL, image에서 semantic_search를 강제로 고르게 하여 답변 생성과 인자 생성 경로를 확인했다. **임시 head는 학습 모델이 아니며 이 출력의 정답률이나 속도 이득을 주장하지 않는다.**

Gemma 전체 멀티모달 forward와 PLE를 유지한다. 마지막 정규화 표현을 한 번 읽고, 같은 prefill KV 뒤에 선택한 툴의 native prefix를 넣어 인자를 생성한다. 실제 이미지 입력도 통과했다. HF 프로세스의 peak allocated memory는 약 **22.5GB**였다. 수집 작업과 동시에 실행한 smoke 시간은 비교 성능에 사용하지 않는다.

중간 설명 생성은 생략하지만 검색어·청크 ID 생성은 필요하다. FINAL 확률은 학습 teacher의 행동 선택에 대한 상대 확률이며 정답 확신도가 아니다. 툴 인자 형식 오류는 원래 생성 경로로 fallback하고, 버린 작업·추가 prefill 비용도 기록하도록 구현했다.

## 남은 실행

82문항 완료 → 성공 라벨 감사 → 부족 시 QC 통과 MMQA 37문항 → 그래도 부족 시 고정 순서 MultiHop-RAG 100문항 단위 수집이다. MultiHop-RAG에서는 전체 609개 기사를 검색하며 정답 evidence_list를 검색 대상으로 직접 제공하지 않는다. 원본 정답이 `Insufficient information.`인 null 질문은 같은 정규화 EM으로 채점하고, 빈 답변은 성공으로 바꾸지 않는다. null 성공은 30개 answerable 성공 조건에 포함하지 않는다.

최소 라벨 조건을 통과한 뒤 한 번의 고정 head 학습을 수행한다. head 학습은 full-batch AdamW, CE, 100 epochs, lr 0.001, weight decay 0.01, seed 20261014로 고정하며 test를 보며 설정을 바꾸지 않는다. GPU 비교는 다른 작업과 경합 없이 동일 HF backend에서 수행한다.

구현: [HF Reader](../../../src/lsdrag/hf_reader.py) · [상태/라벨 복원](../../../src/lsdrag/decision.py) · [학습](../../../scripts/train_decision_head.py) · [계획](../../../docs/JEV_EXPERIMENT_PLAN.md).
