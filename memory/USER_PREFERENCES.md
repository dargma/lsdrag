# Durable user preferences

- Discuss the project in Korean; be candid about novelty and experimental evidence.
- Single NVIDIA RTX PRO 6000 Blackwell Server Edition, about96GB. Shared GPU use for functional/quality experiments is authorized within available VRAM. Speed measurement is not currently required.
- Execute code, models, datasets and Git in local fast. Download public data directly from the web. Drive is storage/backup; avoid thousands of per-file Drive writes. Bundle phase artifacts into archives.
- User explicitly authorizes code changes, staged experiments, Git commits/pushes, parallel agents and Astra review. Avoid repeated permission questions and planning without execution.
- Active priority: test the decision head first. World-model/observation-prediction/surprise ideas are deferred. Do not combine unrelated directions into a kitchen-sink method.
- Preserve original A-RAG intent, jev_v1.0 multimodal extension and current direct dispatch. Include original and new tools; synchronize actual instruction, schema, head and function binding.
- Previous reduced text/image/stop measurements were discarded by the user. Inputs may be reused with fresh full-tool labels; old predictions/features/policy labels and conclusions must not be reused.
- Research should test decision quality, why it works and later agent transfer. Direct tool calling itself is not a novelty claim. Give primary-paper links, strong baseline/benchmark recommendations and a learning checklist.
- Keep persistent Git memory especially carefully; loss of conversation recovery has repeatedly frustrated the user.

## Latest scope override — 2026-10-16

- Text-only RAG 파일럿으로 전환. 이미지 입력/이미지 tools를 제외하며, 이전 multimodal 유지 선호는 과거 프로필 보존에만 적용한다. 최신 중간 크기 Gemma VLM은 text-only 입력으로 유지.
- 행동 head와 인자 생성 최소화를 동시에 적용/비교. 관측 인자는 바로 호출하고 새로 필요한 인자만 생성. 추가 VLM candidate-selection 호출은 피하고 같은 prefill에서 학습형 head로 선택하는 계획.
- GPU 자원 제약 때문에 작은 고정 규모/상한을 우선. 계획: train32→최대48, dev12×3, seed1, worker wall budget60분. 이전60/80/100와3-seed GPU 규모를 자동 적용하지 않는다.
- 모든 변경의 우선순위·계획 vs 구현 vs 실제 실행 상태를 명시하고 Git/Drive에 영속 저장한다. 기준 `docs/JEV_TEXT_ONLY_INTEGRATED_PLAN_20261016.md`.

- Latest model choice: Gemma4E4B-IT text-only, replacing12B for new pilot. Explicitly requested external Decision3.0 D3-Flash9B benchmark; retain resource caps and preserve main integrated-head objective.
