# Gemma decision head — 실험 진행 보고

브랜치: `jev_v1.0` · 2026-10-14 UTC

**두 번째 마일스톤: 새 M3 학습용 82문항 수집 완료. 성공 라벨이 부족해 예정한 MMQA 37문항을 보강한다. 학습된 head의 정확도·속도 비교 결과는 아직 없다.**

| 단계 | 현재 상태 | 근거 |
|---|---|---|
| 평가 분리 | 새 M3 20문항 잠금, 기존 test/dev 학습 제외 | [고정 계획](../../registry/jev_v1.0/label_collection_plan.json) |
| M3 teacher | 새 train 82/82문항 완료; 각 문항 한 번 | 원래 Gemma A-RAG, EmbeddingGemma 2 텍스트+이미지 검색 |
| 보강 후보 QC | MMQA train 37/37 통과·수집 완료 | [QC](mmqa_train_qc.json): URL·이미지 파일/hash 존재 확인. 정답 crop 의미 정렬 검증과는 별개 |
| HF 동작 검증 | text/image 상태 각각 baseline + 임시 head 확인 | [실행 기록](hf_smoke.json) |
| 코드 검증 | 관련 테스트 26개 통과 | 라벨·미래 출력 누출 방지·typed tool parser·원래 Agent |
| head 학습 | 대기 | 성공 문항 ≥30, 각 행동 관측 성공 문항 ≥10 필요 |
| 최종 비교 | 대기 | 새 M3 20문항 × B-HF/D-head, 각 조건 한 번 |

## 확인한 동작

학습 데이터의 text 상태 1개와 image 상태 1개만 사용했다. baseline은 원래 native tool call을 생성한다. 동작 확인용 임시 head는 text에서 FINAL, image에서 semantic_search를 강제로 고르게 하여 답변 생성과 인자 생성 경로를 확인했다. **임시 head는 학습 모델이 아니며 이 출력의 정답률이나 속도 이득을 주장하지 않는다.**

Gemma 전체 멀티모달 forward와 PLE를 유지한다. 마지막 정규화 표현을 한 번 읽고, 같은 prefill KV 뒤에 선택한 툴의 native prefix를 넣어 인자를 생성한다. 실제 이미지 입력도 통과했다. HF 프로세스의 peak allocated memory는 약 **22.5GB**였다. 수집 작업과 동시에 실행한 smoke 시간은 비교 성능에 사용하지 않는다.

중간 설명 생성은 생략하지만 검색어·청크 ID 생성은 필요하다. FINAL 확률은 학습 teacher의 행동 선택에 대한 상대 확률이며 정답 확신도가 아니다. FINAL에서는 기존 요구 형식인 `{"answers":` prefix 뒤 답변을 생성하고, tools에서는 native tool prefix 뒤 인자를 생성한다. 두 prefix 처리 비용을 모두 센다. 단순히 tool 토큰을 막는 초기 방식은 일반 텍스트로 툴 호출을 흉내 내는 출력이 있어, JSON prefix와 형식 검증으로 보완한 뒤 text/image smoke를 다시 통과했다. 툴 인자 형식 오류는 원래 생성 경로로 fallback하고, 버린 작업·추가 prefill 비용도 기록하도록 구현했다.

## 남은 실행

82문항 완료 → 성공 라벨 감사 → 부족 시 QC 통과 MMQA 37문항 → 그래도 부족 시 고정 순서 MultiHop-RAG 100문항 단위 수집이다. MultiHop-RAG에서는 전체 609개 기사를 검색하며 정답 evidence_list를 검색 대상으로 직접 제공하지 않는다. 원본 정답이 `Insufficient information.`인 null 질문은 같은 정규화 EM으로 채점하고, 빈 답변은 성공으로 바꾸지 않는다. null 성공은 30개 answerable 성공 조건에 포함하지 않는다.

최소 라벨 조건을 통과한 뒤 한 번의 고정 head 학습을 수행한다. head 학습은 full-batch AdamW, CE, 100 epochs, lr 0.001, weight decay 0.01, seed 20261014로 고정하며 test를 보며 설정을 바꾸지 않는다. GPU 비교는 다른 작업과 경합 없이 동일 HF backend에서 수행한다.

구현: [HF Reader](../../../src/lsdrag/hf_reader.py) · [상태/라벨 복원](../../../src/lsdrag/decision.py) · [학습](../../../scripts/train_decision_head.py) · [계획](../../../docs/JEV_EXPERIMENT_PLAN.md).

## 마일스톤 2 — M3 학습용 82문항 완료

[문항별 채점·라벨 감사](train82_label_audit.json). **EM 6/82 = 7.32%, F1 16.84%.** 이는 teacher 수집 결과이며 최종 비교용 20문항의 성능이 아니다. 성공 6궤적에서 행동이 관측된 서로 다른 문항 수는 keyword_search 6, semantic_search 5, read_chunk 1, FINAL 6이다. 최소 성공 30문항·행동별 10문항 기준에 미달하므로 이 상태에서 head를 학습하거나 속도 이득을 주장하지 않는다.

기존 train의 성공 궤적은 새 문항과 합쳐 사용할 수 있지만 같은 문항의 여러 조건은 서로 다른 문항으로 중복 계산하지 않는다. QC 통과 MMQA 37문항을 이어서 각각 한 번 수집한다.

## 마일스톤 3 — MMQA 보강 37문항 완료

[MMQA 37문항 채점](mmqa37_label_audit.json): EM **1/37 = 2.70%**, F1 **7.86%**. 기존 train 성공 궤적 + 새 M3 82 + MMQA 37을 합친 [라벨 감사](pooled_label_audit.json)에서 서로 다른 answerable 성공 문항은 **11개**, 행동별 성공 문항은 keyword_search 11·semantic_search 10·read_chunk 2·FINAL 11이다. 아직 학습 기준에 부족하다.

유일한 보완책인 MultiHop-RAG 전체 609개 기사·4,235개 텍스트 청크를 준비하고 77,850개 문장 벡터 인덱스를 만든다. 기존 M3 인덱스는 변경하지 않았다. 고정된 1,000개 질문 순서에서 100개씩 한 번 풀고 라벨 기준을 충족하면 즉시 수집을 중단한다.

원본 풀이 궤적은 Drive의 `outputs/jev_v1.0/`에도 백업한다. 수집 종료 후 기준을 통과하면 head 고정 학습 → 같은 HF backend 20문항×2 비교 → 결과 문서/git push를 이어서 실행하도록 후속 runner를 준비했다. 라벨 기준 미달이면 학습·최종 비교를 실행하지 않고 그 이유를 보고한다.

## Teacher label collection completed

성공한 answerable 학습 문항 **45개**. 행동별 서로 다른 성공 문항: `{"FINAL": 45, "keyword_search": 34, "read_chunk": 36, "semantic_search": 43}`. 라벨 진입 기준 통과: **True**. [수집 감사](pooled_label_audit.json).

## Frozen Gemma decision head trained

backbone 동결·linear head만 CE로 한 번 학습했다. 성공 궤적의 **197턴**을 사용했다. [학습 설정·출처](head_training.json). 학습 loss는 정답률이나 폐루프 성능을 의미하지 않는다. 최종 비교용 새 M3 20문항은 아직 학습/설정 선택에 사용하지 않았다.

## Paired Gemma decision comparison completed

| 조건 | 문항 | EM | F1 | 평균 전체 시간 | 중앙값 | 평균 fallback |
|---|---:|---:|---:|---:|---:|---:|
| B-HF | 20 | 5.0% | 9.9% | 7.273s | 5.938s | 0.00 |
| D-head | 20 | 10.0% | 16.3% | 5.048s | 4.555s | 0.00 |

평균 전체 시간 비율 B-HF/D-head: **1.441배**. 인자·prefix·검색·fallback·최종 답변 시간을 포함한다. [문항별 결과·시간 분해](paired_summary.json). 20문항의 작은 PoC이므로 정확도 비열등성·일반적 속도 개선·보정된 정답 확률을 주장하지 않는다. 보완 자료 사용 여부와 행동별 학습 수는 수집 감사에 명시했다.
