# 선정 head의 GRPO 추가 학습

사용자 명시 지시: 새 BC부터 시작하지 않고, 이미 outcome 학습된 선정 head 자체를 GRPO로 계속 학습한다. 기존 완료 실험과 별도 후속 실행이며, 결과를 과거 최종평가에 소급 적용하지 않는다.

- 시작점: `decision_shared_structure_20261023/multi_mlp_outcome.pt`, SHA `d987142ff7ea00efe06748fa205e84095a500abc675f290f449e035e70eacb99`.
- 동일 head의 multi-position MLP 전체 `net.*`를 업데이트한다. 정규화 mean/scale 및 Qwen3-4B-Instruct-2507 본체는 고정한다. head를 재초기화하거나 별도 행동 선택기를 추가하지 않는다.
- 현재 head로 stochastic rollout 수집 → 해당 head를 GRPO로 업데이트 → 업데이트된 head로 다음 라운드 수집. KL 기준만 시작 head의 고정 복사본이다.
- 기존 train192 × group4 × 2라운드 = 1,536 episode. seed20261055, AdamW lr1e-4, 4 epochs/round, minibatch32, clip.2, KL.02, entropy.01, grad clip1. 보상은 기존과 동일한 F1−.03×cost. Trajectory 길이 균등 가중치와 group-standardized advantage를 유지한다. critic은 없다.
- 단일 GPU lock, 학습 최대3,600초(모델 로드 이후), coordinator 상한3,900초. 생성1.8M/input100M 상한. 실제 사용량 별도 기록. 기존 원8시간 window에 속한다고 주장하지 않는다.
- 학습 후 원자료/behavior logprob replay/정규화 고정/finite checkpoint 검사 및 Drive archive. 기존 validation64에서 학습 전·후 2조건,128episode,최대1,200초. 최종 iteration1을 사전 지정하며 검증 결과로 iteration을 고르지 않는다.
- 원래 최종80은 이미 노출됐으므로 새 확증 평가로 쓰지 않는다. 예비80도 이번 실행에서는 사용하지 않는다. 검증64는 노출된 개발 자료이며 전후 비교는 개발 결과로 보고한다.
- 같은 모델·공통101,962문단·프롬프트·도구·생성 옵션을 유지한다. 이번 변경은 추가 on-policy GRPO 학습이다. 앞선 GRPO 단독과는 초기화/구조/학습 가능 범위/누적 계산량이 달라 순수 알고리즘 비교가 아니다.

실행: `PYTHONPATH=src OPENBLAS_NUM_THREADS=2 /content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/coordinate_decision_selected_head_grpo.py`

학습 run: `decision_selected_head_grpo192_20261024`; 전후 검증: `decision_selected_head_grpo_validation64_20261024`. 후속 sweep, 월드모델 및 메모리 확장은 실행하지 않는다.
