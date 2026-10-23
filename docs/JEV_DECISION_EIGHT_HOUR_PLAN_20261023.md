# 8시간 decision policy 개선 계획 — 2026-10-23

사용자 승인: 8시간 내 학습 신호·입력/본체 구조·확률 활용을 효율적으로 조합하고, 결과에 따라 동적으로 선택. 단일 GPU, Qwen3-4B-Instruct-2507 non-thinking. 시작 14:48 UTC, 상한 22:48 UTC. 모든 조합 순회 금지. 현재는 계획 확정 단계이며 신규 worker 미기동.

## 목적과 목표

공통 LLM의 표현을 작은 policy head가 읽어 행동을 선택하고, 같은 LLM의 생성 경로가 인자·답변을 생성하는 통합 decision-generation 모듈. head 자체가 문장을 생성하거나 미래 상태를 예측한다고 주장하지 않는다. 별도 decision LLM 없는 정책 선택과 유연한 생성이 기여 후보이며 신규성·효과는 검증 필요.

주 목표: 새 공통 평가셋에서 고정 GRPO96 checkpoint 대비 F1 +0.05(5 percentage points) 이상, 평균 generated_tokens 30% 이상 감소 동시 달성. 기존80의 GRPO F1 .3343263/generated56.25 기준 설명용 목표는 .3843263/39.375 이하이나, 새 평가의 동시 baseline 수치로 판정한다. point-target 달성과 paired CI상 유의한 개선을 분리. 입력 token/호출/error/utility 및 실제 측정시간도 보조 보고; 통제 latency claim 없음.

## 비용을 공유하는 실험 흐름

0. 기존 GRPO1392episodes 검증·결과 저장·Drive 백업. frozen baseline 보존.
1. 기존 train96만 이용해 공개 prefix의 hidden/합법행동과 동일 상태의 행동별 후속 F1·cost를 수집. 모든 후보가 같은 cache와 분기결과를 사용. 전체 legal action이 3~4개로 작으므로 대표 상태에서는 전부 평가해 확률가중 기대효용을 정확히 계산; 비대표 상태 추가 수집은 예산에 따라 제한. 분기행동을 강제했다면 on-policy GRPO 자료로 넣지 않는다. rollout source policy/seed/initial state hash/forced intervention/전체 비용 기록.
2. 학습 축: 기존 GRPO 유지 대조 + 동일상태 행동결과를 이용한 보조 정책 개선. Q(s,a)=suffix F1−lambda*suffix cost, J=sum_a pi(a|s)Q(s,a)−KL regularization. 모든 행동을 평가했을 때 쓰는 기대효용 목적이며 GRPO의 무보정 강제행동 ratio로 대체하지 않는다. 기존 round0 outcome-PG 실험과의 차이(현재 GRPO initialization, 수집 state, train96, 고정 trunk 여부)를 명시해 단순 재포장 금지.
3. 구조 축: 캐시된 표현으로 last+linear / last+MLP / multi-position+linear / multi-position+MLP의 작은 2x2 비교. multi-position은 last·question-end·latest-observation mean의 공개 prefix 표현. attention pooling은 multi-position 이득이 보이고 시간이 남을 때 한 후보만. 동일 학습자료·정규화 train-only·업데이트 예산을 적용하고 parameter 수/초기화 차이를 기록. 초기화 영향을 줄이는 train-only GRPO distillation warmstart와 원래 구조 matched control을 포함; 기존 head는 trunk frozen인데 새 head 전체학습인 차이를 숨기지 않는다.
4. 확률 활용 축: argmax label만 학습하지 않고 행동 전체의 확률로 기대효용·KL·entropy를 계산. 확률은 행동 선택확률이며 성공확률이 아님. 불확실 상태 추가 수집은 후보가 필요할 때만 train에서 진행; entropy 단독이 중요도라는 가정 금지. 전체 budget의 최소 절반 uniform으로 커버리지 확보, 추가 selective 방식의 효과는 matched uniform과 비용을 맞추지 않으면 주장하지 않는다. 확률 threshold로 inference fallback을 추가하는 실험은 후순위(추가 생성 비용까지 집계 필요).
5. 전문가 추가 축: 품질–비용 선호. 동일 branch F1/cost로 lambda .01/.03/.06의 저비용 head 재학습 가능. validation에서 정확도 유지와 비용 tradeoff를 확인. budget-conditioned single head는 선호별 정책의 차이가 유용할 때만 추가하고, 현재 목표에 불필요하면 생략. 큰 matrix cross 금지.
6. 검증셋에서 최대2개 후보만 선택. 최종 새 family-disjoint test160 목표(가용성 검증 후 freeze), 기존 GRPO와 선택후보 동시 평가. test 결과로 후보/threshold/lambda를 재선택하지 않음. train/validation/test ID와 decomposition family 분리. 기존 노출 test80 및 과거 모든 test는 학습·튜닝 금지. 최소 독립 seed2 반복은 선정방식/예산을 평가 전에 동결한 경우만 실행.

## 적응적 선택과 시간 배분

0–0.5h 검증/저장/manifest. 0.5–2h shared rollout 및 구현 smoke. 2–3.5h cache 기반 후보 학습/validation. 3.5–5.5h 유망 설정 하나의 on-policy 보강 또는 새 분기 수집. 5.5–7h 동결 최종 평가. 마지막1h raw audit/실패 포함보고/Drive SHA/Git push. 시간은 상한 배분으로, 더 일찍 결론이 나면 무의미한 추가 실험 없이 완료.

- 같은 상태에서 행동별 성과차가 약하면 구조를 키우지 않고 부족한 상태 coverage/반복 추정부터 보완.
- 학습에서는 잘 맞지만 validation에서 실패하면 용량·업데이트를 줄이고 새 구조 확대 중지.
- multi-position이 validation에서 도움이 없으면 last 구조 유지.
- 정확도 개선 없이 비용만 감소하면 더 낮은 cost weight 후보 검토; 성공으로 포장 금지.
- validation에서 목표에 가까운 후보가 없으면 최대2개로 제한해 최종 비교하고 미달 원인 보고. test tuning 금지.
- 구현/버그 수정/무효 결과는 명시해 보존. 전체 wall deadline, GPU worker 한 개, token/episode cap은 실제 runner manifest에 수치로 고정 후 기동.

## 이후 로드맵(이번 8시간에 실행하지 않음)

A-RAG → 짧은 Memory-R1 관리과제 → ACA-RL 기반 단일턴 응답방식 선택. AskBench 다중턴/긴 reasoning/월드모델은 비용과 범위 때문에 보류. Memory-R1/Mem-alpha는 방법 baseline, MemoryAgentBench는 평가 benchmark. 메모리·대화 전이는 아직 검증되지 않은 가설.

참고: https://arxiv.org/abs/2508.19828 ; https://arxiv.org/abs/2509.25911 ; https://arxiv.org/abs/2507.05257 ; https://arxiv.org/abs/2608.16554 ; https://arxiv.org/abs/2602.11199 .

## 실행 전 추가 고정

기존노출family를제외한dev2hop이20개뿐이어서미사용MuSiQue train+dev에서family-disjoint validation32/test160을동결했다. 공식dev-only성적이아닌미사용원천holdout으로표기. train96은기존그대로. 같은공개prefix에서legal3~4행동×2suffix를수집,최대192state/1632episode/7200초,생성1.2M/input40M. baseline GRPO checkpoint 고정. initial+last actual controllable state,capFINAL제외. 동일prefix SHA/hidden parity 및강제첫행동검증.

실행 고정 상세: 새 구조는train-only GRPO분포distillation400epochs(lr.001)→동일상태expectedutility100epochs(lr1e-4),KL.02/entropy.01. 기존구조는원GRPO부터action/skip만같은100epochs. 새구조warmstart4개도validation에서측정해구조변경초기화효과분리. 후보선택순위 max min(F1delta/.05,generated_reduction/.30),동률utility. 최초5개후보중1개만lambda .01/.06추가(정확도또는생성비용중하나라도개선될때). 최종2후보까지만선정. 일반GRPO추가학습대조는동일train96×4×3=1152,현재GRPO초기화+validation32×2;분기수집과tokenbudget완전일치가아니므로정확한compute효율우위주장금지.

## 최신 목표 상향 및 validation 기반 재선택

사용자가validation1/2결과후F1+5%p/generated−30%로상향. test160미열람. 기존validation순위와선정기록은보존하고새선택순위min(deltaF1/.05,token_reduction/.30)적용. linearλ.06은val F1+.05417/token−.25143로새token목표미달. multi-MLP는val F1여유+.10104/token−.13056이므로해당구조의λ.06과generation직접cost후보를추가로비교할근거가있다. 전조합확대금지,추가최대3후보. validation후주후보고정→독립수집/학습seed2재현→finaltest,순서유지. 목표상향은확인된최종성공이아니며validation32의선택편향을명시.
