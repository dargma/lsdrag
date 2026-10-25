# POC 결과 요약 — selection head가 바닐라 A-RAG가 못 하는 결정을 할 수 있는가 (2026-10-25)

대략적인 POC다. 표본이 작고(train48, dev17), 일반화를 주장하지 않는다.

**고정:** Qwen3.5-9B(frozen), 공유 코퍼스 `musique_shared_20261023`, base 정책 value_aux. 원본 수치는 `lab/reports/poc_decision_axis_20261025/`.

## 핵심: ④ 근거 선별 전후 (train48)

| | 전: 에이전트 원래 답 | 후: 9B probe로 상위 5 chunk 선별 후 재답변 |
|---|---|---|
| EM | 21/48 (43.8%) | **28/48 (58.3%)** |
| F1 | .516 | **.700** |
| 문항별 EM 변화 | — | 9 획득 / 2 상실 |
| 답변 시 컨텍스트 | 누적 이력(마지막 턴 수천 토큰) | **약 900 토큰 (chunk 4.9개)** |
| 문항당 전체 입력 | 17,542 | 17,542(검색 궤적, 그대로) + 900(답변) + 선별 pass |

선별 pass는 이번 POC에서 별도 forward 1회(약 3.6k)를 썼다. 에이전트가 이미 prefill한 hidden에서 바로 뽑으면 추가 비용이 0이 되지만, 아직 검증하지 않았다.

**참고 비교 (같은 train48)**

| 조건 | EM |
|---|---|
| 본 chunk 전부로 재구성 | 23 |
| 질문 유사도 상위 5 | 19 |
| gold만 (oracle) | 30 |

**chunk 판정 AUC (에이전트가 본 chunk 860개)**

| 판정 방법 | AUC |
|---|---|
| 9B 16층 평균 probe | .972 |
| 질문 유사도 | .888 |

probe는 실험 문항과 atomic id가 겹치지 않는 train 300문항의 20문단 pool로 학습했다.

## 그 밖의 결과

| 실험 | 결과 | 판정 |
|---|---|---|
| P3 천장 (dev64) | gold 2문단 직접 제공 F1 73.8 / EM 41, 20문단 pool F1 60.4, 에이전트 기록 49.5 | 근거만 있으면 9B는 잘 맞힌다 |
| P1 부분 (dev 17문항) | head와 native 모두 EM 8/17. 입력은 head 15.7k, native 17.4k | 도구 선택 head는 native와 비슷하다 |
| 분기 분석 (train48, 317상태) | head가 고른 행동 F1 .506 > 대안 평균 .406. 대안이 나은 상태 12.6% | 도구 선택 레버는 작다 |
| E0 재구성 | 본 chunk 전부로 재답변: train F1 +8.4pp, dev17 +5.9pp | 이력을 빼면 정확도가 오른다 |
| ② 멈춤 / ⑦ 초안 | oracle 멈춤도 EM +2. probe(AUC .72)는 지배하는 점이 없고, 초안은 비용만 증가 | 중단 |

## 결론과 다음

selection head의 가치는 도구 선택이 아니라 **"무엇을 남길지"(④)**에 있다. 9B 자신의 중간층 hidden이 필요한 근거를 잘 구분하고(AUC .97), 그것만 남기면 정확도와 답변 컨텍스트가 함께 좋아진다.

다음 단계:
1. 에이전트 루프에 연동한다(FINAL 시점에 자기 prefill hidden으로 선별).
2. 매 턴 선별로 입력 자체를 줄인다.
3. 새 문항에서 바닐라 A-RAG와 비교한다.

## 재현

| 단계 | 스크립트 |
|---|---|
| P3 천장 | `scripts/poc_oracle_context_qwen35.py` |
| E0 재구성 | `scripts/poc_e0_clean_context.py` |
| 분기 분석 | `scripts/poc_branch_headroom.py` |
| 멈춤 probe | `scripts/poc_stop_probe.py` |
| 1단계 멈춤 분석 | `scripts/poc_stage1_state_answers.py`, `scripts/poc_stage1_analyze.py` |
| ④ 선별 | `scripts/poc_idea4_selector.py` |

- **환경:** 공개 소스에서 재구축했다(모델, MuSiQue, 코퍼스 벡터 모두 해시 일치). 별도 복원 대상은 아래 두 가지다.
  - `value_aux.pt`, 분기 아카이브 `ad4b8537`: Drive에서 사용자 승인으로 복원
- **설치:** sentence-transformers 6.1.0(embedding venv), lm-format-enforcer 0.11.3
