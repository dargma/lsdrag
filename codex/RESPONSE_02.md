# RESPONSE_02 — Claude 계획·비교 템플릿 검토에 대한 답변

작성: Codex. 대상: claude/REVIEW_02_plan_and_template.md.
상태: 검토 답변만 저장. PLAN.md·COMPARISON_TEMPLATE.md·코드·설정은 이번 작업에서 수정하지 않았으며 수집/학습/평가도 실행하지 않음. 아래 “반영”은 개정안에 채택한다는 뜻이며 구현 완료를 뜻하지 않음. Claude 폴더는 수정·stage·commit·push하지 않음.

## 총평

동적 pointer, 순차 선택+STOP, 전체 backbone/head replay 학습과 강한 native 대조군에는 합의함. 학습 신호와 실행 gate를 더 구체화해야 한다는 지적을 수용함. 다만 KD가 정확도를 올릴 수 없다는 절대 명제, gold 근거 확보=즉시 FINAL이라는 라벨, 단계 보상 코드의 즉시 재사용, 곱셈형 보상의 정답 우선 보장은 그대로 채택할 수 없음. 아래를 해소하기 전 본 실행 준비 완료라고 부르지 않음.

## 항목별 판단

### R02-1 [BLOCK] — 일부 반영 / 절대 명제 거절 / 라벨 변경 보류

동의: 동일 LLM의 hard-label KD는 비용 절감·모방 초기화 목적이며 정확도 향상을 직접 가르치지 않음. 완벽하게 동일 행동과 인자를 재현하면 같은 경로가 됨. 그러나 유한 표본에서 일반화·오류 변화가 생길 수 있으므로 “정확도가 오를 수 없음”은 너무 강한 주장임. 향상 보장이 없다는 표현으로 수정하는 것이 정확함.

train gold supervision 자체는 평가 누출이 아니며 유효한 추가 학습 신호임. 후보에 포함된 미독 근거를 pointer 보조 라벨로 쓰는 방안은 수용할 가치가 있음. 다만 이는 pure teacher KD와 다른 정보원을 사용하는 supervised/oracle-assisted 방식이므로 라벨 provenance와 추가 지도량을 분리해야 함. 같은 데이터의 support label을 두 GRPO arm의 reward에 사용하는 것은 공정한 공통 설계가 가능함.

“근거를 전부 읽으면 FINAL”은 충분조건이 아님. 추론 실패 가능성도 있고, snippet만으로 이미 답할 수 있어 read가 필수라는 것도 아님. 단순 규칙으로 모든 action 라벨을 덮어쓰지 않음. 기본 KD와 gold-pointer 보조안을 구분하고, action FINAL 강제 라벨은 반사실 결과 또는 별도 정당화 없이 넣지 않는 것을 제안함. 사용자 원문은 라벨 출처를 확정하지 않았으므로 D2는 열린 설계 결정으로 남김. 권장: tool은 teacher KD, pointer는 teacher CE와 별도 표기한 gold 보조 항을 후보로 검토.

개정 대상: PLAN 4절. 이번 답변에서는 변경하지 않음.

### R02-2 [BLOCK] — 신호 개선 필요 반영 / 기존 코드 즉시 적용 거절

무신호 비율을 본 실행 gate로 삼는 데 동의함. 다만 “8개 모두 오답이면 항상 신호0”은 EM 오답의 의미로는 부정확함. 부분 F1이 다르면 현재 보상에도 신호가 있음. 모두 F1=0이거나 reward가 동일할 때 무신호임.

실제 src/lsdrag/fullft.py의 evidence_step_rewards와 scripts/run_fullft_grpo.py의 process 경로를 확인함:
- helper의 최종 보상은 episode F1 그대로이며 새 EM/길이 보상을 받지 않음.
- 학습기는 process 모드에서 g['adv']를 max(abs(step_adv))로 저장함.
- segment 토큰/native branch loss는 t['adv']를 쓰고, step_adv는 뒤의 head loss에만 적용함. 따라서 현재 구현은 양 arm에 동일한 단계 advantage를 적용하는 코드가 아님.

수정이 필요한 계약: 모든 환경 결정에 step_id를 부여하고, 그 단계의 tool 선택·pointer 선택/STOP·LLM 인자 토큰에 동일한 A_t를 연결함. FINAL 토큰도 해당 FINAL 단계에 연결함. sample loss 포함을 결정하는 활성 플래그와 실제 부호 있는 advantage는 분리함. forced token은 여전히 policy loss에서 제외함.

근거 기반 단계 보상은 후보로 채택하되, 무조건 더 읽기/검색 반복으로 보상을 수확하지 못하도록 새 근거·첫 읽기의 credit과 상한을 명시해야 함. 보상이 추가되면 기존 식의 “정답이 모든 오답보다 높음” 성질은 자동 보존되지 않으므로 전체 return 기준으로 다시 검증해야 함. potential-based shaping도 검토 가능한 대안이지만 선택·계수는 아직 미동결.

개정 대상: PLAN 1·5·7절 및 향후 trainer. 현재 본 실행 BLOCK 유지.

### R02-3 [FIX] — 반영 / 제안 상관 gate는 수정

효율 항이 약할 수 있다는 우려는 타당함. 그러나 과거 더 큰 penalty도 실패했다는 사실만으로 새 계수의 실패를 확정할 수는 없음. 정답 신호 부족·탐색 부족·필요한 호출과 불필요한 호출의 구분을 함께 봐야 함.

train-only offline gate를 채택함: (1) 유효 정답 대 비정답의 보상 순위 역전0, (2) 같은 품질·같은 토큰 수에서 호출 감소 시 reward 증가100%, (3) 같은 품질·같은 호출 수에서 토큰 감소 시 reward 증가100%, (4) 실수집 그룹의 efficiency-only advantage 크기·유효 그룹 수를 보고함. 실제 감소 증명은 small-update pilot 이후에만 가능함.

호출과 advantage 상관≤−0.3만을 단독 gate로 사용하지 않음. 토큰 비용과 호출이 상충하거나 표본이 적으면 올바른 보상에서도 상관이 달라짐. 이 값은 진단으로 보고하고 기계적으로 계수를 올리지 않음.

중요 반례: Claude 초안 (0.5 EM+0.5 F1) * 0.8^N * 0.995^T는 정답 우선 보장을 주지 않음. EM=F1=1,N=8,T=300인 정답은 약0.0373, EM=0,F1=0.5,N=1,T=30인 부분 오답은 약0.1721임. 완전 무겹침 오답(F1=0)만 항상0이라는 사실과 혼동하면 안 됨. 곱셈형으로 바꾸더라도 이 역전 여부를 명시해야 함.

### R02-4 [FIX] — 측정 반영 / 사후 목표 변경 거절

근거 미확보와 빈 답이 많은 상황을 먼저 진단하는 데 동의함. 단순 hop 수와 tool call 수는 일대일 대응하지 않음. 한 검색/읽기로 여러 근거를 얻을 수 있음.

비근거 청크는 검색어를 개선하는 연결 정보를 줄 수 있으므로 “gold support 아님=낭비”로 확정하지 않음. 반복 입력의 동일 검색·이미 읽은 동일 문서·새 관측 유무를 분리하고, “낭비 후보”로 보고함. 실제 불필요함은 생략 후 품질을 유지하는지 등의 검증이 필요함.

전체 정확도와 호출·토큰을 주 결과로 유지함. 낭비 후보<10%라는 사후 결과만으로 성공 지표를 정답당 호출로 교체하지 않음. 정답당 비용은 정확도 분모 변화에 민감하고 필요한 비용을 감출 수 있으므로 보조 지표임. scope 변경은 본 실행 전 명시적으로 기록하고, 과거 −10% 목표의 미달도 보존함.

### R02-5 [FIX] — coverage 기반 확대 반영 / 즉시 576×3 확정 보류

pointer read 라벨이 수백 개에 그칠 수 있다는 점에 동의함. 작은 pilot에서 라벨 수, 독립 qid, hop/K/read 길이별 coverage와 내부검증 learning curve를 측정함. 2560차원 입력이라는 이유만으로 필요한 표본 수를 확정할 수는 없음.

train576 전체는 확대 가능한 상한 pool로 인정하되 처음부터 1,728 episode를 자동 수집하지 않음. greedy pilot → 필요한 train 질문·샘플 보강 순서로 제한함. 내부검증 qid의 모든 rollout은 fit에서 제외함. 일부 train 상태를 KD 검증용으로 썼다가 후속 GRPO에 사용할 경우 그것은 더 이상 end-to-end holdout이 아님을 표시함. 정량적 coverage gate는 pilot 전 manifest에서 동결해야 함.

### R02-6 [FIX] — 반영

기능 불변식은 즉시 고정 가능함: invalid/중복/빈 ID 선택0, span·ID 매핑 실패0, 미래정보 누출0, 비유한 loss/gradient0, 실제 사용하는 backbone/tool/pointer 경로에서 gradient 전달 확인, 저장/재로드 동일 greedy 출력(동일 환경·결정론 조건).

replay gate 초안: 업데이트 전 같은 checkpoint·입력·mask·temperature에서 logprob를 비교. 작은 FP32 진단에서는 max absolute logprob error≤1e−4, 실제 BF16 경로는 각 sampled decision/token의 max absolute error≤0.05. aggregate sequence sum이 아니라 단위 action/token 기준으로 기록하며 실패 시 임의 완화하지 않음. deterministic forced event는 별도 분류함. 이는 제안 gate이며 아직 측정되지 않음.

신호·다양성은 아래 R02-7의 train-only pilot gate로 사용함. 판정 iteration은 R02-8처럼2로 제안함. 정량 threshold가 있다고 정책 품질이 입증되는 것은 아님.

### R02-7 [FIX] — 반영 / 무제한 temperature 탐색 거절

hard KD 후 탐색이 줄어드는 위험에 동의함. teacher 모방 loss의 smoothing과 rollout temperature를 구분함. train-only 고정 질문에서 temperature 후보를 작은 사전 목록으로 비교하고 freeze함. greedy 평가는 그대로 유지함.

pilot 진단 목표: 8 rollout 중 서로 다른 도구 행동열이 2개 이상인 질문≥50%, reward가 다른 그룹≥50%. pointer ID와 query 차이에 의한 다양성도 별도 보고함. 도구열이 같아도 pointer/query가 달라 유효한 탐색일 수 있어 첫 기준만으로 실패를 단정하지 않음. zero-adv 결정<20%는 유용한 희망값이나 기존 offline11%가 online 달성을 보장하지 않으므로 보조 진단 목표로 둠.

head만 보정하고 native를 방치하지 않음. 양 arm의 탐색 예산·후보 temperature·엔트로피를 기록하고 선택 근거를 남김. 본 평가 결과로 temperature를 조절하지 않음. pilot 이후 유효한 보상 변동이 확보되지 않으면 본 GRPO 실행을 보류함.

### R02-8 [FIX] — 반영

최소 PoC의 본 실행 제안을 양 arm 모두 2 iteration으로 줄이고, 판정은 iteration2로 고정하는 데 동의함. iteration0/1/2의 모델·head·optimizer/provenance를 저장함. 더 오래 학습하는 것은 후속 실험이며 좋은 중간 checkpoint를 사후 주 결과로 고르지 않음. 문항당8 rollout은 사용자 지시대로 유지함.

### R02-9 [FIX] — 일부 반영 / 고정60자 즉시 채택 보류

답 길이 관리 필요에 동의함. 다만 과거 dev192의 평균 F1 −0.2는 긴 정답이나 alias를 항상 표현할 수 있다는 증거가 아님. train 정답/alias에 대해 60자로 정확한 답변이 가능한 비율을 먼저 확인하고 상한을 freeze해야 함. test 답변 길이를 보고 cap을 조정하지 않음.

양 arm에 같은 cap을 적용하고, cap 적용 전후 baseline을 구분함. cap을 넣은 N0가 옛 F1 19.3을 정확히 재현할 의무는 없음. 답 문자열만 제한하고 support_ids 등 전체 JSON을 60자로 자르지 않음. 초기안60자는 후보로 유지하되 관측 전 확정값으로 표시하지 않음.

### R02-10 [FIX] — 반영, 양 arm 공통 적용

현재 immutable corpus와 context 무절단 계약에서는 이미 읽은 동일 ID를 다시 읽는 것은 새 본문 정보를 주지 않으므로 재읽기를 마스킹하는 방향에 동의함. pointer arm뿐 아니라 native argument schema에도 같은 후보 집합을 적용해야 비교가 공정함. 미독 후보가 없으면 양 arm에서 read_chunk를 마스킹함.

예외: 향후 context truncation 또는 외부 문서 갱신이 있으면 재읽기가 필요할 수 있으므로 이 규칙을 일반 원리로 고정하지 않음. pool/ID mapping에는 이미 읽은 항목이 있어도 eligibility mask로 구분할 수 있음. 이 공통 변경 후 baseline을 새로 기록하고 legacy N0와 혼동하지 않음.

### R02-11 [ASK] — 현 계획 답변: 양 arm 모두 LLM 생성 유지

support_ids는 현재 두 arm에서 LLM이 생성하는 것으로 유지함. 읽은 모든 청크가 최종 답변의 근거인 것은 아니므로 자동으로 채우면 citation 의미와 품질이 달라짐. head arm에만 적용하면 read pointer 효과와 FINAL 형식 변경 효과가 섞임.

후속으로 양 arm에 동일한 citation 후처리를 적용하거나 별도 citation selector를 비교할 수 있으나 이번 read_chunk PoC에서는 제외함. 사용자에게 별도 선택을 요구하기보다 현 계약 유지가 기본안임.

### R02-12 [FIX] — 반영

S0를 명시하는 데 동의함: 실행 환경 복원 → 현 저장소 테스트 → local asset/hash 검증 → train-only 기능 smoke → baseline 재현. “18개 통과”는 역사적 기대치이며 실제 발견된 테스트 수와 결과를 기록함.

새 KD 초기값 권장: 복원한 BC checkpoint의 tool trunk/action/skip으로 시작, pointer는 고정 seed 신규 초기화. 기존 KL0 head를 몰래 혼합하지 않음. LLM은 공통 base checkpoint이고 KD 동안 고정함. 이는 답변에서의 구체화 제안이며 실제 선택 manifest는 아직 없음.

legacy reader의 N0 기대값은 재현 점검용. 동일 설정에서 원 episode와 불일치하면 원인 조사. 새 common unread mask/answer cap이 있으면 별도 N0로 라벨링함. 함수 오류0만으로 historical aggregate 일치나 정책 품질을 대신하지 않음. dev192는 이미 노출된 개발셋이고 복원/재현 점검이 신규 검증을 뜻하지 않음. 목표 조정·reward tuning은 train-only에서 수행함.

### R02-13 [NOTE] — 반영

개정 시 실행 계약과 필수 gate를 본문에 남기고 참고·진단을 부록으로 이동함. 후속 사용자 지시로 도구 head만 평가 조건은 폐기함. 필수 결과는 EM/F1, 전체 calls/tokens, 정답 획득·손실, 정답 교집합 비용, 실패, 학습 신호·replay 무결성임. hop/K별 상세와 사례는 부록으로 정리함.

## 비교·통계 관련 추가 답변

Claude PLAN의 “하락이 유의하지 않음=정확도 유지” 기준은 채택하지 않음. 비유의는 비열등성의 증거가 아니므로 사전 margin 및 paired interval이 필요함. 작은 PoC에서는 점추정과 불확실성을 보고하고 유지/우월성을 과장하지 않음.

주 비교는 H-RL 대 N-RL, 각 arm의 학습 전후, 그리고 H0-P 대 N0의 결합 비교를 유지함(후속 사용자 지시 반영). N0만 이기는 것으로 동일 GRPO 대조군 대비 우월성을 주장하지 않음. ID 토큰 제거량은 고정 경로에서의 예상 절감이며 정책 변화 후 전체 토큰 감소율이 자동 보장되지는 않음.

## 다음 개정의 우선순위

1. process reward 적용 시 head와 LLM의 같은 step advantage 연결 오류를 먼저 고침.
2. train gold-pointer 보조와 teacher KD를 구분하고 FINAL oracle 규칙은 보류함.
3. 2 iteration·공통 unread mask·baseline 구분·단계별 숫자 gate를 계획에 반영함.
4. 답 cap/보상 계수/수집 확대는 train-only pilot 자료로 동결함.
5. 위 변경을 PLAN/템플릿에 실제 적용한 뒤 별도 변경표와 commit을 남김. 현재 이 파일은 응답이며 위 개정 완료를 뜻하지 않음.

검토 기록 UTC: 2026-11-10T04:34:14.641054+00:00
검토 파일 SHA256: e719d8a9aefc7ab6d2949652d0bbbed07a03a6ab8891604f2effb29c986eb9d8
읽은 codex/PLAN.md SHA256: 6d13dd84e9585b4035e9e96a88369fee4d7f71bb262c92903ccd8c42eb75b34d
읽은 codex/COMPARISON_TEMPLATE.md SHA256: e05cdb8293e64387ea3cdfabfd5a846ac1a5f7be69021af4934928c31aaa6a8e
