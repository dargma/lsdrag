# 구조 재검토 및 attention 확장 — 2026-10-24

## 사용자 방향과 재검토 결론
PPO/GRPO 없이 오프라인 학습. 기준 d987: F1 25.18 / generated 76.75. Native ARAG 프로토콜 고정. 목표는 downstream decision policy 기반 구조이며 imitation loss 최적화 자체가 목표가 아니다.
기존 depth/input_gate/depth_input_gate/observation 네 후보는 유지한다. depth+gate는 독립 축이 아니라 상호작용 후보. 파라미터 수는 동일하지 않아 순수 구조의 인과 효과라고 주장하지 않는다. 첫 평가 worker379465/coordinator379451은 후보 재검토 요청으로 종료되었으며 실제 프로세스 없음 확인. 마지막 저장 status 8 episodes/299 generated/16965 input; F1은 후보 선정에 사용하지 않았다. 중단 namespace와 원본을 보존하며 재시작하지 않는다.

## 추가 후보
| 후보 | 입력과 구조 | 가설 |
|---|---|---|
| question_attention | 질문 끝 hidden query → 32 prefix bin K/V → 4 logits residual | 현재 질문에 필요한 문맥 선택 |
| action_attention | 질문 query + 4 learned action vectors → 행동별 문맥 attention → shared scorer | 행동마다 필요한 증거가 다름 |
| action_set | 행동별 문맥 집약 + legal actions 간 4-head self-attention → shared scorer | 상대적인 행동 기회비용을 표현할 용량 |

width32, output zero-init으로 최초 d987 logits 정확 일치. Backbone 동결, 기존 base head와 새 module은 함께 학습. LayerNorm은 학습 파라미터이며 base normalization buffers는 고정. 32 bin means는 raw token attention이 아니다. Learned action vectors는 option text embeddings가 아니며 새 도구 zero-shot generalization을 주장하지 않는다. Attention map이나 weight 변화는 정책 이해 증거가 아니다.

같은 train192/377 states, seed20261061, 100 updates, AdamW lr1e-4, clip1, Q=F1−.03 mixedcost, KL .02, entropy .01. 마지막 checkpoint만 저장; epoch cherry-pick 없음. 2 suffix/action의 noisy old-policy Q라는 한계 유지. 기존 실험 source는 변경하지 않는다. 새 plan.json에서 source/input hashes 고정 후 실행.

## Loss 필터와 누수
학습 loss는 NaN/발산/미학습 경로를 거르는 용도다. 기존 d987는 train192 전체를 학습했으므로 일부 질문을 나눠 continuation 검증을 하더라도 진정한 holdout이 아니다. 깨끗한 내부 검증 loss가 필요하면 base BC와 outcome fit, normalization 모두 fold 밖 질문을 제외하고 다시 학습해야 한다. 그때도 old collector policy가 전체 train teacher를 학습했다는 수집 편향은 별도 한계다. 이번 continuation 후보를 contaminated loss로 탈락시키지 않는다. 공통 objective 외 보조 total loss를 직접 순위화하지 않는다.

## 공개 decision model에서 배울 점 (2026-10-24 확인)
- [Kev 공식 구현 설명](https://github.com/jaredpalmer/kev#how-it-works): option 끝 hidden과 decide hidden의 pointer scoring. 우리 캐시에는 option 끝 hidden이 없어 동일 구현이 아니다. 상태-행동 matching 원칙만 차용. 현재 Kev의 LoRA/backbone 학습과 우리 frozen backbone도 다르다.
- [Jeva v2](https://arxiv.org/abs/2610.18773v2), [공식 저장소](https://github.com/LiQiiiii/Jeva): request/option 별도 encoding, 고정 option memory를 읽는 decision stack, shared scoring 및 coverage. 이번 action_set은 이 구조의 재현이 아니라 집합 비교 가설의 저비용 변형. Null coverage를 FINAL과 동일시하지 않는다.
- [D3-Flash 공식 card](https://huggingface.co/vllm-sr/d3-flash): text 생성 없이 choice probabilities. 기존 정적 코드 감사는 docs/JEV_DECISION_POLICY_RESEARCH_PLAN_20261023.md 참고. 외부 decision backbone을 추가하지 않는다.
- Jev 자체의 비공개 내부 구조를 추측으로 재현했다고 주장하지 않는다. 공개 대체 모델의 인터페이스 유사성과 실제 architecture를 구분한다.

## 검증과 다음 실행
7 CPU tests 통과: 최초 logits 일치, 학습 후 context sensitivity/key gradient, illegal action probability zero, finite weights. 실제377상태에서도 최초 parity와 학습된 context path를 확인한다. 기존4 + 새3 + continuation control + selected reference를 같은 validation64에서 평가할 예정. 비용 절약을 위해 순차 실행하고 결과는 paired F1/tokens로 판단. 이 단계는 development이며 reserved80 접근이나 목표 달성 주장을 하지 않는다.
