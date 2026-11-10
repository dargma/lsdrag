# 사전 결과 비교 템플릿

상태: 실험 전. 아래 빈칸은 미측정이며 0을 의미하지 않음. 판정 시점과 설정을 실행 전에 동결함.

## 비교 조건

| ID | 조건 | 시작점 | 목적 |
|---|---|---|---|
| N0 | native constrained, 학습 전 | 공통 base LLM | native 출발점 |
| H0-P | KD tool + pointer | base LLM + KD heads | 제안 구조의 출발점 |
| H-RL | LLM 전체 + heads 전체 GRPO | H0-P | 2차 주실험 |
| N-RL | LLM 전체 native GRPO | N0 | 3차 대조군 |

H-RL과 N-RL은 같은 reward, 질문, rollout8, 학습 반복 수와 tool/context cap을 사용함.
추가 action-only GRPO는 현재 필수 실행에 포함하지 않음. 따라서 RL 후 차이를 pointer 단독 효과로 분해하지 않음.

## 조건부 원인 진단 — 필수 비교 아님

트리거: 같은 공통 계약의 dev192에서 H0-P의 EM 또는 F1 점추정이 N0보다 낮을 때만 실행. 두 지표 모두 N0 이상이면 H0-G는 미실행으로 기록함. 미실행 칸을 0으로 채우지 않음.

H0-G는 동일한 base LLM·KD tool checkpoint·미독 후보 mask·답 cap·문항·greedy 설정을 유지하고, read ID만 pointer 대신 LLM이 생성함. N0가 옛 legacy 설정이면 직접 트리거 비교하지 않고 공통 계약 N0를 사용함.

| 진단 조건 | 실행 사유 | EM | F1 | tool calls | 생성 토큰 | 오류 |
|---|---|---:|---:|---:|---:|---:|
| H0-G | 조건 충족 시만 | — | — | — | — | — |

- H0-G vs N0: tool head/KD 경로의 변화 진단.
- H0-P vs H0-G: 같은 tool checkpoint에서 read 선택 경로 전환의 영향 진단. 이후 trajectory 상호작용 포함.
- H0-G도 나쁘면 tool/KD 문제가 있을 가능성, H0-G에서 회복되면 pointer 경로의 기여 가능성을 사례·오류와 함께 확인. 단일 원인을 단정하지 않음.
- 선택적으로 실행한 사후 진단으로 명시하며 주 네 조건 표/성공 gate/GRPO 비교 조건을 바꾸지 않음. dev 기반 재학습·온도/보상 튜닝 금지.

## 주 결과 표 — 같은 문항, 사전 고정 최종 checkpoint

| 조건 | 문항 수 | EM | F1 | agent iteration | tool calls | 생성 토큰 | 입력 토큰 | 빈 답 | 오류 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N0 | — | — | — | — | — | — | — | — | — |
| H0-P | — | — | — | — | — | — | — | — | — |
| H-RL | — | — | — | — | — | — | — | — | — |
| N-RL | — | — | — | — | — | — | — | — | — |

모든 수치는 전체 문항을 포함함. 실패 사례를 빼고 평균을 개선하지 않음. dev192와 untouched test320을 섞지 않고 별도 표로 작성함.

## 핵심 비교와 해석

1. H0-P 대 N0: tool+pointer 구조와 KD의 결합 변화. pointer 단독 인과 효과를 주장하지 않음.
2. H-RL 대 H0-P / N-RL 대 N0: 각각의 학습 전후 변화.
3. H-RL 대 N-RL: 동일 GRPO rollout 예산에서 최종 시스템 비교. head arm의 추가 KD 비용 별도 기재.
4. 차이의 차이: (H-RL−H0-P)−(N-RL−N0). 초기 성능 차이를 보완하는 기술통계이며 무작위 인과 증명 아님.

qid-paired bootstrap CI와 wins/losses/ties는 별도 부록. 작은 점추정 차이를 유의한 우월성이나 비열등성으로 단정하지 않음.

## 정확도와 비용을 함께 판정

| 비교 | 둘 다 정답 | 정답 획득 | 정답 손실 | 둘 다 오답 | 둘 다 정답에서 Δcalls | 둘 다 정답에서 Δtokens |
|---|---:|---:|---:|---:|---:|---:|
| H-RL vs H0-P | — | — | — | — | — | — |
| N-RL vs N0 | — | — | — | — | — | — |
| H-RL vs N-RL | — | — | — | — | — | — |

- 전체 평균 비용 감소가 빈 답·오답·조기 FINAL 증가 때문이면 효율 개선 성공으로 판정하지 않음.
- 모두 정답인 교집합 분석은 설명용 조건부 분석이며 전체 표를 대체하지 않음.
- 목표: 정확도 유지/개선 + tool calls 10% 이상 감소 + 생성 토큰 감소. 10%는 과거 PoC 목표와 연결한 제안 기준이며 실행 전 동결 필요.
- EM/F1 손실이 있으면 quality-cost tradeoff로 보고함. 통계적 품질 유지 주장에는 사전 비열등성 margin과 충분한 표본이 필요함.
- calls는 그대로이고 ID tokens만 줄면 생성 비용 감소만 확인된 것임. iteration 축소 성공으로 쓰지 않음.

## 청크 선택 진단

| 항목 | N0/teacher 관측 | H0-P | H-RL |
|---|---:|---:|---:|
| read 상태/고유 질문 수 | — | — | — |
| 후보 K: median/p95/max | — | — | — |
| 후보에 gold support가 있는 비율 | — | — | — |
| teacher ordered-list exact / set F1 | — | — | — |
| 선택 수 평균 / STOP 정확도 | — | — | — |
| 없는 ID·중복·빈 선택·span 오류 | — | — | — |
| read support recall / 반복 읽기 | — | — | — |
| read ID 생성 토큰 | — | — | — |

고정 teacher 상태 imitation과 자율 rollout QA를 구분함. support 진단은 scorer-only.
현재 native도 observed-ID constrained decoding을 쓰므로 없는 ID 오류 제거를 고유 성과로 과장하지 않음.

## 학습·비용 진단

각 iteration: EM/F1, 호출, 토큰, reward 성분, reward std, zero-adv 그룹, 8개 중 서로 다른 trajectory 수, tool/pointer entropy, clip fraction, KL, replay gap, gradient norm.
비용: teacher/KD/GRPO trajectory 수와 토큰·시간을 분리함. decode tokens, forced-prefix tokens, 전체 prefill tokens, read 반환 tokens, pointer 연산·시간을 별도 기록함.
공유 GPU wall-clock은 관측치로만 보고하고 인과적인 속도 향상 수치로 해석하지 않음.
2/3/4-hop 별 표, K 구간별 표, 정답 손실/획득 사례, reward hacking 사례를 함께 첨부함.

## 결론 작성 칸

- 기능 구현 통과: 미측정
- KD imitation: 미측정
- 자율 QA 품질: 미측정
- 정답 유지하며 호출 감소: 미측정
- 생성량 감소: 미측정
- native GRPO 대비 개선: 미측정
- 한계 및 실패 사례: 미측정
- 다음 단계: 결과 확인 전 자동 확장 없음

## 보상 사전선정·smoke 기록

| round | 고정 rollout/출발 checkpoint | 식·계수 | 변경 이유 | 보정15 순위/신호 | 확인9(동결 후1회) | 실제 update 수 | 판정 |
|---|---|---|---|---|---|---:|---|
| offline12 | 미수집 | 계획 후보12 | 초기 계수 미검증 | 미측정 | 미측정 | 0 | 미실행 |
| smoke1~3(필요시) | 미정 | round 전 동결 | 관측 근거 필요 | 미측정 | 재튜닝에 사용 금지 | 최대2/arm | 미실행 |

최종 결과에 탐색/teacher/KD/smoke 비용을 따로 합산함. offline 순위 최적을 학습 후 성능 최적으로 부르지 않음.
