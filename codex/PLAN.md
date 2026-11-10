# jev_v2.2 계획 — 선택은 head, 내용은 LLM (개정 6)

2026-11-10. 개정6: OBS_02 반영. offline 계수 자동선정 폐기, 원 계수 유지+진단. 반복 smoke 계속/중단 조건 사전 고정. 전체 실행 승인 후 S0 진행 중; gate는 유지함.
기준: REVIEW_04_plan_rev1_final.md; 이전 PLAN 89e4898e6c6a. 이 문서가 이전 PLAN/RESPONSE의 미확정 제안보다 우선함.
모델·데이터 파일 복원은 RESTORE_VERIFIED.json에서 확인됨. 실행 환경·새 runner 준비 완료를 뜻하지 않음.

## 고정 구조·비교 계약

| 번호 | 도구 | 선택 뒤 처리 |
|---:|---|---|
| 0 | keyword_search | 도구 prefix를 기존 KV에 forward → LLM keywords/top_k 생성 |
| 1 | semantic_search | 도구 prefix를 기존 KV에 forward → LLM query/top_k 생성 |
| 2 | read_chunk | pointer 선택 → 실제 ID 매핑 → chunk_ids 조립·실행, ID 생성0 |
| 3 | FINAL | LLM answer/support_ids 생성. 두 arm 모두 support_ids 생성 유지 |

pointer는 read 선택 후에만 실행. 현재 입력의 후보 title/snippet hidden을 pooling하여 같은 scoring 함수를 후보별 적용함. K는 동적이며 최신 검색5개로 제한하지 않음.
후보 ID 순서·token span을 저장하고 이미 읽은 ID는 두 arm 모두 mask. 미독 후보0이면 read 금지. 미래 본문·gold를 policy 입력에 넣지 않음.
순차 pointer+STOP: 첫 STOP 금지, 이미 고른 후보 mask, 1~8개 선택; 8개/후보 소진은 강제 종료로 STOP loss 없음. LLM 재호출 없이 선택 요약을 갱신함.
원본 Qwen3-4B-Instruct-2507(cdbee75f17c01a7cc42f958dc650907174af0554), embeddinggemma-2(914f7f89142e33e77833254d9c9b90c3cef7303b), non-thinking 유지.
shared corpus101,962/768차원; split_mixhop.json 고정: train576(각192), dev192(각64), test320(160/80/80). dev는 기존 노출셋, test는 최종 동결 전 사용 금지.
두 GRPO arm은 같은 base LLM에서 별도로 시작. head arm만 KD 초기화 비용 추가. 이전 full-FT A1/B1 또는 KL0를 시작점으로 섞지 않음.
필수 비교는 N0(native constrained), H0-P(KD tool+pointer), H-RL, N-RL 네 조건. H0-G(KD tool+LLM ID 생성)는 H0-P의 dev192 EM 또는 F1 점추정이 같은 공통 계약의 N0보다 낮을 때만 원인 진단으로 실행함. 둘 다 N0 이상이면 실행하지 않음.
조건부 H0-G는 H0-P와 동일한 base LLM·KD tool checkpoint·후보 mask·답 cap·문항·greedy 설정을 사용하고 read ID 처리만 생성으로 바꿈. H0-G vs N0는 tool/KD 변화, H0-P vs H0-G는 read pointer 전환의 영향을 진단하되 상호작용과 trajectory 변화를 함께 보고함. 선택적 사후 진단이며 필수 주 비교·성공 gate에 추가하지 않고, dev 결과로 재학습/튜닝/판정 시점 변경하지 않음.
최대 tool8/context16384/생성 budget512(기존 prefix 포함 규칙)을 공통 적용. pointer 내부 선택은 tool call이 아님. seed20261110.

## 실행 단계

단일 새 pipeline은 만들지 않음. 기존 collector/KD/evaluator/GRPO/report 스크립트를 확장하고 preflight·reward gate만 새로 작성함. 아래 명령은 수정 후 CLI 계약이며 신규 옵션은 현재 미구현; 실행 가능한 현 명령으로 해석하지 않음. 시간은 공유 GPU에서 미측정 추정치.

| 단계 | 명령(예정 CLI) | 확인할 숫자·통과 기준 | 예상 시간 |
|---|---|---|---|
| S0 복원/감사 | `python scripts/preflight_v22.py` 및 `PYTHONPATH=src:scripts python -m pytest -q tests` | 자산 hash 일치100%, 테스트 실패0(과거18개, 실제 수 기록), train smoke16의 구현 오류0 | 설치 포함30~90분 |
| S1 라벨/KD | `python scripts/collect_decision_shared_teacher.py --spec <teacher576.json>` → `python scripts/fit_decision_shared_initialization.py --spec <kd.json>` | smoke16→greedy576; ID/span 매핑100%, 불법/중복/빈 선택0; 각 hop read 검증 qid≥10, 유효 read fit qid≥100 | 수집·KD1~3시간 |
| S2 KD 평가 | `python scripts/evaluate_decision_shared_corpus.py --spec <eval_kd.json>` | train smoke8 오류0/read ID decode0; H0-P dev192 보고. EM 또는 F1이 N0 미만일 때만 H0-G 진단. dev로 재학습하지 않음 | 30~60분 |
| S3 신호/보상 | `python scripts/check_v22_reward_gate.py --spec <pilot24x8.json>` | 두 arm train24(각hop8)×8, 업데이트 없이 비교; 아래 보상·다양성 gate | 수집 포함30~90분 |
| S4 GRPO | `python scripts/run_fullft_grpo.py --arm head --pointer --manifest <head.json>` / `--arm native --manifest <native.json>` | 먼저 train16×8/2 optimizer step smoke; 통과 뒤 각각192문항×8×2 iteration. 주 판정iteration2 | smoke30~60분+각3~5시간 |
| S5 보고 | `python scripts/report_fullft.py --manifest <comparison.json>` | i0/i1/i2 dev192, 동결 i2 test320. 전체 문항·실패 포함, paired 결과 | 평가 포함1~2시간 |
### S0: baseline와 줄일 여유

legacy 설정 N0를 먼저 별도 재현하여 F1≈19.3/calls≈2.48/tokens≈107.8과 차이를 조사함. 재현 불일치는 원인을 설명하기 전 통과하지 않음. 새 unread mask/답 cap 적용 N0는 별도 기준으로 저장하며 옛 수치 일치를 요구하지 않음.
train의 정답/alias 길이, teacher read의 근거 precision/recall, FINAL 때 근거 확보, 반복 검색·재읽기·새 정보 없는 호출 비율을 측정함. 비근거 문서를 곧바로 낭비라 부르지 않음.
상한 후보 C=max(60,train576 문항별 최단 유효 정답/alias 정규형 길이의99백분위). 공식 normalize_answer 후 Unicode 문자수, nearest-rank ceil(0.99*n)으로 계산하고 원문 길이도 보고함. 빈 정규형은 별도 오류 감사. 답 문자열만 제한하며 두 arm 동일 적용.

C를 넘는 정답/alias 개수와 등록된 모든 정답 정규형이 C를 넘는 qid 수·목록을 기록함. 후자가0이면 채택; >0이면 S1 전에 상한 유지/확대의 정확도 제약을 보고하여 결정함. 해당 질문 제외·qid별 cap·test 기반 조정은 금지. 이렇게 이상치 자동 확대와 정답 제약의 은폐를 모두 피함.
### S1: 라벨 충돌 없는 teacher KD + gold-pointer 지도 (R03-1/2)

tool 초기값은 복원 BC head(head.pt SHA256 0023dd45dee8…); pointer는 seed 신규 초기화. LLM frozen, 사용하는 tool trunk/action/skip·pointer 전 층 학습. 미사용 value/success/cost는 제외.
수집 전 hop별153 fit/39 내부검증 qid를 고정(459/117), family 중복을 감사·보고. 내부검증 qid의 모든 rollout은 KD fit에서 제외. 해당 qid를 후속 GRPO에 사용하면 최종 holdout으로 부르지 않음.
N0 greedy576을 1회 수집. smoke16은 fit에서 선택하며 계약이 동일하면 본 수집에 재사용하여 중복 생성하지 않음. 실패 QA도 보존하고 dummy uniform policy_probs를 teacher target으로 사용하지 않음.
라벨 규칙은 하나: tool은 teacher action; 실제 teacher read 상태에서 미독 gold 후보가 있으면 그 후보들이 pointer 주 라벨, 없으면 유효한 teacher 선택. 한 상태에 상충하는 teacher/gold CE를 동시에 넣지 않음.
gold 후보는 저장된 후보 순서로 정렬하고 최대8개; teacher fallback은 원 순서를 유지. 오류/중복/범위 밖 teacher read는 pointer 라벨 제외, 유효 tool 라벨은 유지. 출처·제외 건수와 STOP을 기록.
gold는 train/internal-validation의 오프라인 라벨·채점용일 뿐 추론 입력에 없음. gold 전부 읽음→FINAL 강제 라벨은 사용하지 않음. 이 단계는 pure KD가 아닌 teacher KD+gold supervision으로 보고함.
loss는 질문 균형 tool CE + 질문 균형 read sequence 평균 CE(각 가중1). AdamW lr3e-4, 최대100 epoch, 내부검증 합산 CE 최소 checkpoint(동률은 이른 것), smoothing0.05는 tool에만 적용.
내부검증의 gold-present read에서 pointer support set-F1≥같은 상태의 teacher set-F1을 요구하고 teacher-fallback 일치율도 별도 보고. coverage/품질 미달이면 본 GRPO로 가지 않음; 추가 수집은 마지막 표의 규칙을 따름.
### S3: 보상과 단계 보상 채택 분기 (R03-4)

기본 terminal reward: `Q=0.8*EM+0.2*F1; R=Q*(1-0.10*T/(T+128))-0.15*EM*N/8`. official alias-max EM/F1; N은 tool 수, T는 실제 decode한 episode 전체 토큰. 공급 prefix·입력·pointer 연산은 별도 비용.
정답 R≥0.75, 비정답 R≤0.2; F1=0 오답에는 효율 보너스 없음. 실패/빈 답/잘린 JSON은 품질0, 인프라 실패는 별도 기록. reward 성분별 로그 필수.
train24×8에서 terminal-only의 `A=R-그룹평균`을 계산. forced event 제외 환경 결정 기준으로 |A|≤1e−8의 비율을 셈. 두 arm 각각 무신호 결정<20%면 단계 보상 없이 진행.
어느 arm이든 ≥20%면 두 arm에 공통 근거 단계 보상을 구현하고 동일 pilot을 재채점: `r_t=0.05*새 검색 gold 수/전체gold 수 + 0.05*첫 read gold 수/전체gold 수`, 마지막에 terminal R. 각각 처음1회만 credit, 중복0, episode 보너스합≤0.10.
그러면 전체 return은 정답≥0.75, 비정답≤0.30으로 순위 보존. gold 근거 수집 자체를 장려하는 추가 목적이며 평가의 QA 품질과 구분함. |전체 return|만으로 성공을 판정하지 않음.
단계 mode는 return-to-go에서 같은 qid의 다른7개 rollout의 같은 단계 RTG 평균을 뺌(이미 종료된 rollout RTG=0). std로 나누지 않음. 같은 단계의 tool/pointer/생성 토큰에 같은 부호 있는 A_t를 연결함.
수정 필수: 기존 process 경로의 head-only 제한·max(abs(step_adv))로 토큰 advantage를 대체하는 코드를 그대로 사용 금지. 업데이트 포함 여부 플래그와 실제 advantage를 분리. 단계 ID 단위 hand-check를 양 arm에서 검증.
순위 역전0, EM=1·같은 토큰이면 calls 감소 시 reward 증가100%, Q>0·같은 품질/calls이면 tokens 감소 시 증가100%를 검사. EM=0의 call 감점 없음/Q=0 동률을 명시. 전체 return의 비교는 단계 보너스까지 동일할 때 적용함.
각 arm에서 EM 정답≥2개이고 그 정답들의 calls가 서로 다른 그룹 수/전체 유효24그룹 수를 보고함(분자·분모 명시). <10%면 호출 절감의 직접 비교 신호가 희소하다고 S4 전에 보고하고 본 실행 진행 여부를 사용자에게 확인; 계수 자동 증폭·목표 사후 교체 금지. 이는 충분한 신호의 진단이지 학습 가능성의 필요조건/불가능성 증명이 아님.
전체 학습 reward가 다른 그룹≥50%, 서로 다른 tool 행동열≥2개인 그룹≥50%를 각 arm pilot gate로 함. temperature 후보{1,2,4}를 train pilot에서만 비교하고 두 gate를 만족하는 가장 낮은 값을 arm별 동결; payload T=1 유지.
단계 보상 후에도 무신호≥20% 또는 탐색 gate 미달이면 본 실행 중단·보고. 사용자 승인으로 아래 사전 train-only 보상 검토/smoke 반복은 허용하되, gate 기준을 낮추거나 dev/test로 온도/보상/표본을 조정하지 않음. gold 근거가 같으면 calls/token 단조성도 유지되나 서로 다른 근거량 간 효율 우선은 보장되지 않음.

### S3 추가: 가중치·보상식 사전 최적화 (사용자 승인)

목적은 긴 GRPO를 반복해서 보상을 찾는 낭비를 줄이는 것임. 저장 rollout만으로 학습 후 최적점을 증명할 수 없으며, offline 순위·신호 적합성과 실제 짧은 업데이트 결과를 구분함.

1. KD 후 두 arm의 train24×8 rollout을 고정 저장. 동일 데이터에 CPU 재채점하며 후보마다 재생성하지 않음. hop별8 qid 중 해시순5는 보정(총15), 나머지3은 확인(총9). 이 분할은 보상을 보기 전에 manifest에 기록함.
2. 첫 후보군은 정확도 Q=0.8EM+0.2F1와 토큰 스케일128을 고정하고 calls 계수 {0.05,0.10,0.15,0.20} × token 계수 {0.05,0.10,0.15}의 12개. 공통 과정 보상 모드는 원 gate에 따라 선택. 정답 최소≥0.65, 오답+과정 최대≤0.30인 범위만 허용함.
3. 후보마다 정답/오답 역전, 단조성, 무신호, 정답≥2+다른 호출 그룹, efficiency advantage 크기, 빈 답/긴 부분정답 순위를 보고함. 정답이 여럿인 그룹에서는 적은 호출→동률이면 적은 토큰을 선호하는 순위와의 일치도를 계산함. 이는 정답을 알고 하는 offline 진단이며 실행 정책 성능이 아님.
4. OBS_02 권고(a) 채택: 원 계수(.15,.10)를 유지. 12개 후보는 제약·신호·엇갈리는 정답 쌍의 수를 보고하는 진단일 뿐 자동선정/최적화에 사용하지 않음. 보정15/확인9 모두 기술통계이며 확인9로 계수를 고르지 않음. 비용 우선 순위 일치도를 높였다는 이유로 큰 호출 계수를 선택하지 않음.
5. 필요시 최대3회 총 smoke round. 각 round는 양 arm 각각 같은 fit train16×8, optimizer update 최대2, 같은 출발 checkpoint로 초기화. 기존 rollout 재사용은 checkpoint/temperature/grammar가 같을 때만 허용. 각 round 전 식·계수·이유 commit. smoke 전후 같은 train16 greedy 평가와 선택 분포 변화를 측정하며 dev/test는 사용하지 않음.
6. 각 round 필수 기록: 전후 EM/F1(0~1), calls/tokens, 8개 중 도구열>=2인 그룹 비율, sampled-event 평균 k3 KL, PPO clip fraction, clip 전 gradient norm, 업데이트 전 replay gap 및 업데이트 후 평균 abs logprob 변화.
7. 하드 중단 기준(어느 arm이든): 기존 replay/ID/유한 gradient gate 실패; post-update 평균 k3 KL>0.10; PPO clip fraction>0.50; EM 하락>=2/16; F1 하락>=0.10; 재수집8-rollout을 수행한 경우 도구열 다양성 그룹<25%. 해당 시 본 실행 금지·원인 보고. 기본적으로 추가 재수집은 하지 않고 pre-update 그룹 다양성과 greedy 행동 변화만 기록하며 post-update 다양성을 측정했다고 쓰지 않음.
8. 업데이트 무효 경보: 전후 EM/F1/calls/tokens 차이가 각각1e-6 이내이고, 같은 replay 상태의 평균 abs logprob 변화<1e-5이면 보상 변경 반복을 멈추고 gradient/업데이트 경로를 조사. 2 update에서 greedy 품질 개선이 없다는 이유만으로 학습 불가능이라 하지 않음. 분포가 움직이고 하드 gate가 통과하면 작은 smoke는 기능 검증으로 충분함.
9. 다음 reward round를 여는 조건은 아래 두 가지뿐이며 품질 점추정 EM/F1이 모두 하락하지 않아야 함. S3의 직접 호출 비교 그룹>=10%와 signal/diversity gate 통과도 전제함.
   - 호출 항: 전체 terminal reward 성분의 centered absolute 크기 합 중 호출 성분 비중<5%, greedy calls가 줄지 않음 → call 계수 .15에서 .20으로 한 번만 증가.
   - 토큰 항: 같은 정의의 token 성분 비중<5%, greedy 생성토큰이 10% 초과 증가 → token 계수 .10에서 .15로 한 번만 증가.
   성분은 Q, 음의 token cost, 음의 call cost로 분리하고 qid 그룹 평균을 각각 뺌. 분모0이면 비율 미정/무신호이며 자동계수변경 금지. 과정 보상은 별도 보고하여 terminal 비중과 혼동하지 않음. 양 조건이 동시에 맞으면 호출 항 먼저, 다음 round에서 나머지 조건 재확인. 조건이 없으면 추가 round 없이 종료.
10. round3 뒤에도 동일 비용 약신호 경보가 남거나 하드 gate가 미달이면 본 실행하지 않고 보고함. gate threshold는 낮추지 않음. 현재 허용된 변경은 위 두 bounded 계수 변경뿐이며 추가 수식 변경이 꼭 필요하면 실패 근거와 정확한 대체 식·예산을 별도 사전 기록하여 검토함. 승인된 일반 smoke 허용을 무제한 식 탐색으로 해석하지 않음.
11. 모든 smoke 가중치는 본 실행에 이어 쓰지 않음. 최종 공통 reward를 동결한 뒤 두 본 실행을 시작함. 이것은 검증된 최적 reward를 보장하지 않음.
12. 기존 사용자 확인 gate(호출 절감 직접 비교 그룹<10%), S1 coverage/quality, S3 signal/diversity gate는 유지. smoke 허용이 본 실행 gate 자동 통과를 뜻하지 않음.

### S4: 실제 전체 unfrozen 학습과 수치 gate

구현은 기존 run_fullft_grpo.py/fullft.py에 pointer·step_id·manifest 계약을 추가함. 검증된 fp32 master optimizer와 기존 replay 경로를 보존하고 새 경로의 회귀검사를 추가함. 과거 replay 약0.02는 새 pointer 경로의 통과 근거가 아님.
H-RL은 base LLM+S1 heads 전 층, N-RL은 같은 base LLM 전체를 학습. replay에서 현재 prefix hidden·후보 표현을 재계산하여 tool/pointer loss가 backbone까지 흐르게 함. detached feature로 대체 금지.
old/new logprob는 후보·선택 순서·mask·temperature·grammar 분포 동일. forced syntax/STOP은 loss 제외, grammar fallback은 실제 분포 replay가 검증된 토큰만 포함. 단계 mode의 A_t는 동일 step의 모든 sampled event에 적용.
smoke gate: ID/span 오류0, 비유한 loss/gradient0, tool/pointer/LLM 사용 경로 gradient 존재; 업데이트 전 BF16 단위 action/token max logprob 오차≤0.05, FP32 소형 진단≤1e−4. 저장/재로드 동일 greedy 출력, 실패 시 임의 기준 완화 금지.
공통 PPO clip0.2, KL beta0.02(각 arm i0 frozen reference), batch16 trajectories, LLM lr1e−6/head lr5e−5, gradient clip1, update1 pass/iteration. sampled event loss 합을 batch trajectory 수로 나누고 별도 길이 정규화 없음; gradient 규모와 길이 의존성 보고.
각 iteration 같은 train192(hop별64)×8, seed/qid 순서 공통, 총2 iteration. i0는 base/KD 해시 참조, i1/i2의 LLM·heads·manifest 보존. optimizer·master·RNG·scheduler는 현재 arm 최신 재개 상태를 원자적으로 교체하며 최종 재개 상태는 해시 확인한 Drive 백업에 보존. 마지막 i2가 주 checkpoint. pilot 가중치는 본 실행에 이어 쓰지 않음.

## 자원 운영 — 점유율보다 전체 완료 시간

04:41:41 UTC 실측: GPU RTX PRO6000 97,887MiB, 사용0/여유97,251MiB, util0%; RAM available173GiB, local-fast 여유179GB, CPU48. 순간값이며 다른 작업 점유는 기동 전 재확인함.
GPU 작업은 하나의 실행 큐로 관리함. 같은 checkpoint의 수집·평가를 묶어 가능한 범위에서 모델 load를 재사용. head/native full-FT 동시 실행 금지(각 약70GB는 과거 추정). 공유 GPU라도 여유가 충분하면 진행하며 타 작업은 종료하지 않음.
CPU는 feature 매핑·라벨·KD·보상검사·보고·Drive 백업을 수행. S1 수집 뒤 CPU KD 중 동일 base의 N0 평가를 실행할 수 있음. 데이터 의존성·gate를 건너뛰지 않음. manifest/checkpoint/prompt/문항/greedy 설정이 동일한 N0만 i0 평가에 재사용함.
full-FT 첫 smoke는 여유≥80GiB에서 시작하는 보수적 기준. 이후 단계별 실제 peak reserved/프로세스 점유량을 측정해 예상 추가 사용+max(8GiB,15%) 여유가 확보될 때만 기동. 임베딩/KV/reference/master/optimizer를 모두 포함함. 30초 간격 상태를 기록하고 문항·update 경계에서 재검사, 부족하면 자체 작업만 안전 중지/재개함.
rollout8은 통계적 그룹 크기이며 GPU 동시 batch8을 뜻하지 않음. 먼저 microbatch1로 측정하고 여유가 있으면2→4를 smoke에서 시험; 실제 episode/s가 개선되고 replay/메모리 gate를 만족할 때만 채택. gradient accumulation으로 batch16 유지. 모델을 unload하지 않은 반복 rollout→update를 우선하며 새 배치 최적화가 PoC 병목이면 기존 경로로 진행함.
현재179GB 여유에서4B optimizer/master는 대략48GB, 모델은약8GB 규모이므로 모든 iteration optimizer 복제는 금지. 다음 원자적 저장·동시 임시파일·백업 staging의 실측 합+20GiB가 없으면 저장 전에 정리/대기. 이 실험의 중복 checkpoint만 Drive byte-hash 확인 후 정리하며 마지막 검증 재개본을 먼저 지우지 않음. mutable 학습 상태를 백업 thread가 직접 읽지 않음.
GPU가 gate/복사 때문에 쉬는 시간을 별도 기록하고 CPU 작업과 겹치되, idle을 채우기 위한 불필요한 rollout은 금지. S0/S1/S3/smoke 실측 episode/s·update시간·복사시간으로 남은 ETA를 갱신함. smoke에서 이득을 입증하지 못한 packing/일괄 forward 최적화는 본 실행 중 도입하지 않음.

## 비교·판정과 남은 분기

전체 EM/F1/calls/tokens와 정답 획득·손실, 양쪽 정답의 비용을 함께 보고. H-RL vs N-RL이 주 비교이며 각 arm 전후와 H0-P vs N0도 유지(구조+KD의 결합 효과). 목표는 정확도 유지/개선+calls10% 감소+tokens 감소; 목표 미달을 다른 지표로 교체하지 않음.
비유의를 품질 유지로 해석하지 않음. paired bootstrap 95% 구간을 보고하고 비열등성은 사전 허용손실 margin 미합의 시 주장하지 않음. seed1 PoC·KD 추가 지도·공유 GPU 비용을 명시함. 기존 COMPARISON_TEMPLATE의 실험 전 빈 표는 유지하되 설정 충돌 시 이 PLAN을 우선함.
| 미확정/조건부 항목 | 언제·무엇으로 결정 |
|---|---|
| coverage 부족 시 추가 수집 | S1 gate 미달일 때만 train fit qid에 샘플1회 보강하는 별도 예산안 제출. dev/test 사용 금지 |
| 답 cap 실제 숫자 | S0의99백분위 후보·전체 alias가 막히는 qid 수를 보고 결정; 양 arm 동일, test 열람 전 동결 |
| terminal vs 단계 reward·온도 | S3의 수치 분기로 선택·manifest 기록. 아래 제한된 사전 sweep/smoke 범위에서만 검토하고 실패면 보고 |
| 호출 절감 신호 희소 | S3 직접 비교 그룹<10%면 S4 전에 보고·진행 결정. 계수 자동 변경 없음 |
| 통계적 비열등성 margin | 본 실행 전 사용자 허용 품질손실을 확정하지 못하면 우월/비열등 단정 없이 구간 보고 |

범위 밖: Q-head, 별도 critic, 메모리 agent 확장, 자동 추가 iteration. Claude 파일·코드는 이번 개정에서 수정하지 않음. 본문 시간/threshold는 사전 설계이지 측정 결과가 아님.
