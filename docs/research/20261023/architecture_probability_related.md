# 구조·확률 해석 보충 조사 — 2026-10-23

원문 확인이며 재현 실험은 아니다. 아래 차별화 판단은 우리 연구에 대한 해석이다.

## SanSi — 2026-10-06, v2 2026-10-10

[논문 이력](https://arxiv.org/abs/2610.07730), [v2 §3–4, Appendix B/H](https://arxiv.org/html/2610.07730v2).

Loop 사전학습된 Ouro의 마지막 prompt hidden에서 선택지 확률을 읽는다. Base weights는 고정하지만 attention/FFN LoRA와 loop별 저랭크 readout은 학습하므로 **head-only가 아니다**. 각 loop에 CE+Brier를 주고 12,800개 typed decision으로 학습한다. RL 변형과 외부 generator의 verifier 역할도 별도 평가한다. Loop가 늘면 정확도와 calibration이 함께 좋아진다고 보장하지 않으며, 반복 연산 비용이 든다.

우리 적용: “토큰을 생성하지 않는 결정”을 곧바로 저비용으로 취급하지 말고 forward·prefill을 센다. 복잡한 head 검토 시 linear/MLP와 같은 데이터·예산 비교가 먼저다. SanSi의 verifier와 우리 다단계 action policy는 역할이 다르다. Readout 구조만 바꿔 loop 학습 효과를 얻는다고 주장하지 않는다.

## Hev — card 명시 최초 공개 2026-09-24

[저자 model card, Training/Limits](https://huggingface.co/EryriLabs/DiffusionGemma-26B-A4B-Hev). 논문 심사 여부는 확인하지 않은 커뮤니티 모델 공개이며 아래는 저자의 기술·평가 기록이다.

Frozen DiffusionGemma의 여러 층/위치 표현을 projection+Hydra blocks+pointer readout에 넣는다. 세 head ensemble이며 총 약44.5M head parameters. typed-decisions train 및 jev-bench 19개 source를 사용하고 KL+0.1Brier로 학습한다. Card가 직접 synthetic teacher agreement, schema specialization, 다른 domain에서 backbone보다 낮은 성능을 명시한다.

우리 적용: frozen backbone 위 구조화 head, 여러 위치 readout, 확률 출력 자체는 이미 공개 사례가 있다. 작은 train24에서 대형 head가 일반화할 근거는 없다. Teacher 일치율·calibration을 환경에서의 정답·탐색 정책 성능으로 바꾸어 말하지 않는다. Head ensemble 비용과 우리가 쓰는 약0.34M MLP를 구분한다.

## 우리 출력 계약에 대한 제안

현재 `action_probabilities`는 π(a|s), 즉 선택 분포다. 정답확률·가치·탐색 필요성은 서로 다른 target이다. 향후 필요하다면 동일 표현에서 다음을 별도로 학습·검증한다. 이는 현재 pilot2 구현 완료 주장이나 자동 실행 계획이 아니다.

- `action_probabilities`: 실행/샘플링할 정책 분포.
- `expected_answer_quality[a]`: 명시된 continuation policy 아래의 최종 품질 추정.
- `expected_remaining_cost[a]`: 같은 정책 아래의 남은 비용 추정.
- 검증: 선택 확률의 entropy만 보지 말고 outcome 예측의 Brier/NLL·reliability와 실제 quality–cost를 분리한다. F1 회귀와 binary success calibration도 구분한다.

정책이 바뀌면 continuation-conditioned target도 달라진다. 같은 state의 비용·정답 target을 영구적인 환경 속성으로 취급하지 않는다.
