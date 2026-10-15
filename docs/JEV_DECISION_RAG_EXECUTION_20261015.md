# Current override — native-target audit and GPU idle gate

See `JEV_DISTILLATION_PROTOCOL_20261015.md` for the newly authorized experiment. The original v13 protocol below remains historical; do not automatically continue its stage2 with changed source or native labels.

# 순차 실행 계약 —2026-10-15

사용자 “너의 제안대로 차례로 진행하자”, “단계별로 잘 저장하고”에 따른 현재 실행 계획. 이전 argument candidate+GENERATION_ARGS 결합안보다 이 문서가 우선한다. 모델·data·five tools+FINAL·G1 검색 및 실제 이미지 전달을 유지하며, 인자 생성 경로는 변경하지 않는다.

##1단계: 기존 환경의 품질·비용 진단

Fixed train first20, native tool generation / generated-letter+common JSON arguments / first-token+common JSON arguments의3조건을 paired 실행한다. Native는 같은 HF backbone에서 원래 tool serialization을 생성하며 A-RAG 논문 재현이라고 부르지 않는다. Native에는 letter-only instruction을 주지 않는다. Controlled 두 조건은 동일 selector prefix/args/final 경로다. 같은32K context,8 tool rounds,8 images,1120 image tokens,256 output tokens per call, greedy decoding. Native는 한 출력에 여러 tool call을 포함할 수 있어 실제 handler 호출 수를 따로 센다. 단일 call의 budget이 같아도 controlled는 action/argument가 별도 forward임을 공개한다. 정확도·비용의 backend/출력방식 차이를 숨기지 않는다.

먼저 fixed first2를 실행해 실제 오류를 확인한 뒤 unchanged manifest로 first20까지 확장한다. 전체20 전후로 원형 tool/이미지 전달 real-backbone synthetic smoke를 검증한다. Synthetic smoke는 정책 품질 증거가 아니다. Parser는 vLLM v0.20.0의 typed helper3개를 변경 없이 분리했으며 provenance/hash를 기록한다.

모든 문항의 official list EM/F1, format validity, 도구 분포/중복 호출/실제 이미지 전달, prompt/completion tokens, decision vs arguments/final generation의 관측 시간을 기록한다. Native의 decision/argument decode는 분리되지 않는다고 표시한다. 모델출력의 정책 오류와 infra 실패를 분리한다. Gold는 scoring 및 post-hoc failure audit 전용이며 retrieval/candidate/runtime 입력에 제공하지 않는다. 잘못된 검색/선택/이미지/종료/reader 실패는 자동 proxy와 확인된 사례를 구분하며 metadata support hit를 sufficient evidence로 부르지 않는다.

성공/실패에 관계없이20 paired 결과와 원인·비용 보고서가1단계 완료 증거다. Parser/model/infra 오류가 있으면 바로잡기 전 진행하지 않는다. 심각한 원형/controlled 형식차이는 별도 버전으로 수정하고 해당 실행을 invalid protocol attempt로 보존한다. QA 점수가 낮다는 이유만으로 실패 질문을 제외하거나 설정을 맞추지 않는다.

##2단계: 같은 interface에서 decision readout의 추가 가치

새 generated-letter/commonargs trajectories만 teacher로 사용한다.1단계 fresh generated20은 동일 frozen contract이므로 재사용 가능하다. 과거 v2/Q3 labels는 재사용 금지. Train60, coverage 부족하면80/100. 최소30 distinct natural teacher qids 및 여섯 행동마다5 distinct qids. ForcedFINAL 제외, 실패 episode 포함.100에서도 부족하면 NoGo이며 훈련·평가 완료로 주장하지 않는다.

Coverage 통과 시 linear6 plain CE, train-only normalization folded weights, seeds20261015/16/17,100epochs. 기존 first-token/commonargs와 generated/commonargs 및 모든3 head/commonargs를 dev20에서 paired 평가한다. Best seed 선택 금지. Native dev20도 별도 참고조건으로 평가하되 prompt/serialization 차이가 있는 원형 참고선임을 명시한다. Head imitation accuracy와 closed-loop QA/cost를 분리한다. 이 단계는 행동 readout/BC feasibility이지 outcome-optimal policy나 연구 신규성 입증이 아니다.

1단계의 실패 진단이 새로운 outcome labels가 필요함을 보이면 해당 제안을 보고서에 기록한다. 새로운 branch target/RL/candidate model을 이 실험에 몰래 추가하지 않는다.2단계 완료 후에 인자 최소화의 독립 ablation을 설계·실행한다. A-RAG 전면 재설계는 interface 병목 증거가 있을 때 검토한다.

## 경로와 복구

- Run `/content/local_fast/lsdrag-head-v13/results/decision_rag_20261015`
- Logs `/content/local_fast/lsdrag-head-v13/logs`
- Commands `lab/registry/decision_rag_20261015/common_args.json` + `scripts/run_decision_rag_stages.py --phase diagnose|teacher|train|evaluate`
- Data `/content/local_fast/lsdrag-head-v12/data` (recovered JSONL hashes frozen)
- Backup `/content/drive/MyDrive/lsdrag-backups/20261015/decision_rag_20261015`

각 JSON은 temp→atomic replace, teacher JSON/PT pair 검증 후 resume. Diagnostic은 문항별 archive로 Drive backup. 단계 완료 시 report+CURRENT+HEAD_EXPERIMENT+status+SESSION_LOG를 Git commit/push하고 archive SHA256 기록. 긴 teacher 단계도 최소5문항마다 archive한다. 프로세스 PID/launch/로그를 local 및memory에 기록하고 실제 command line을 검증한 뒤 resume한다. GPU phase는 하나씩. Shared GPU latency 우월성 주장 금지. 학습/평가 전 완료 prefix 및 contract/data/source identity를 확인한다.

## 출력 예산 진단 추가 — 현재256 run은 그대로 완료

첫10문항에서 native4/10, controlled 각각1/10의 최종 답변이256-token 제한에 걸렸다. 점수 차이에 truncation/JSON 불완전성이 섞여 있어 이 조건의 teacher로 곧바로 head를 훈련하지 않는다. 현재20문항은 변경 없이 마쳐 budget256 진단으로 보존한다. 이어 기존 AgentConfig 기본값인1024를 모든 조건에 공통 적용한 별도 새 run으로 동일20문항을 다시 진단한다. 검색/선택 prompt·도구·데이터·순서·모델은 그대로 유지한다. 어떤 질문도 제외하지 않는다. 이는 점수 최적화 sweep가 아니라 관측된 출력 잘림을 분리하는 단일 공통 예산 검사다.

새 root `/content/local_fast/lsdrag-head-v13/results/decision_rag_20261015_1024`; args `lab/registry/decision_rag_20261015/common_args_1024.json`. Config/feature identity가 달라지므로256 teacher labels/checkpoints는1024 학습에 재사용하지 않는다. 비교 품질 판정은 유효 JSON 비율과 raw scores를 함께 보고한다.1024에서도 truncation이 발생하면 숨기지 않고 해당 한계를 보고하며 무제한 증가시키지 않는다.2단계의 teacher/heads/dev에도 동일1024를 유지한다.
