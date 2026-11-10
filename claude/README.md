# ./claude — Claude 기록 폴더

Codex 기록은 `./codex/`. 이 폴더는 Claude의 검토 의견·메모만 둠.

## 파일
- `RESEARCH_BRIEF_select_heads_write_tokens_20261110.md` — 사용자가 준 연구 브리프 원문 (수정 없음)
- `PROTOCOL.md` — Claude ↔ Codex 소통 프로토콜 (Codex는 이것부터 읽기)
- `PLAN.md` — Claude가 쓴 PoC 계획 (사용자 요청, 미실행)
- `REVIEW_01_restore_verified.md` — 복원 기록 검토
- `REVIEW_02_plan_and_template.md` — codex/PLAN.md·COMPARISON_TEMPLATE.md 검토 (BLOCK 2, FIX 9, ASK 1)
- `REVIEW_03_response_02.md` — codex/RESPONSE_02.md 검토 (Claude 정정 6건, 남은 의견 5건)
- `REVIEW_04_plan_rev1_final.md` — codex/PLAN.md 개정 1 마지막 검토 (BLOCK 없음, FIX 3)
- `OBS_NN_*.md` — 실행 중 관찰 기록 (OBS_01: 실행 시작 시점)
- `INSTRUCT_codex.md` — Codex가 실행 중 지킬 것 7가지

## 사용자가 밝힌 목표 (2026-11-10, Claude 세션)

1. **목표**: selection head를 도입해 A-RAG에서 **iteration을 줄이고 → 생성량을 줄이고 → 더 정확하게**.
2. **지금 하는 일**: 그 목표를 위한 PoC.
3. **학습 설계가 핵심**: head 학습(KD) → 그것을 기반으로 한 강화학습(GRPO)에서 **의도한 효과가 실제로 나오도록** 설계하는 것이 목적. GRPO 등 학습 부분의 개선점이 있으면 반영.

5. **범위**: 지금은 A-RAG baseline을 개선하는 것만 함. 브리프의 AtomMem(메모리)·Q-head 등은 이번 PoC 범위 밖.
4. **PoC 원칙**: 중복·장황함 없이 최소 범위로. smoke 실행으로 초기 실수를 빨리 잡은 뒤에 본 실행.

검토 기준은 위 항목들임: 계획의 각 단계가 "iteration↓ · 생성↓ · 정확도↑"를 실제로 만들어 내고, 측정으로 그것을 확인할 수 있는가.

### 이전 실험에서 이미 확인된 장애물 (jev_v2.1, 계획이 피해야 할 것)
- GRPO의 그룹 표준화 때문에 길이 페널티 계수 λ가 사실상 무력했음.
- 긴 궤적일수록 보상이 높았음 (상관 +.34) → 호출 수가 줄지 않음.
- 생성 비용의 대부분이 형식·청크 ID 토큰과 반복 프롬프트였음.
- head 최대 확률이 .92로 rollout 다양성이 낮았음.
- head trunk를 얼리고 10.8k 파라미터만 학습했음. 병목은 용량이 아니라 학습 신호였음.
- 4개 arm 모두 목표(호출 −10~30%, 정확도 유지 이상)를 못 맞춤. B1은 iteration 2에서 호출 −6.0%였다가 iteration 4에서 +6.5%로 뒤집힘.
- 2hop 데이터는 낭비 호출이 약 6%뿐이라 줄일 여지가 없었음 → mixed-hop으로 전환함.
