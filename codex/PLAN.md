# jev_v2.2 계획 — 선택은 head, 내용은 LLM

작성일: 2026-11-10. 상태: 계획 초안, 실행 미승인/미시작. 규모·계수는 아래 제안값을 구현 전 검증하고 동결함.

## 1. 목표와 범위

MuSiQue mixed-hop에서 정확도를 유지·개선하면서 agent iteration과 생성 토큰을 줄이는 PoC.
별도 결정 모델 없이 Qwen3-4B-Instruct-2507의 hidden을 사용함. 메모리 agent 확장, Q-head, 별도 value/process reward는 이번 범위에서 제외함.

## 2. 도구와 인자 처리

| action 번호 | 이름 | 선택 이후 | 실행 인자/출력 |
|---:|---|---|---|
| 0 | keyword_search | 선택 도구 prefix를 같은 KV cache에 처리하고 LLM 생성 | keywords, top_k |
| 1 | semantic_search | 선택 도구 prefix를 같은 KV cache에 처리하고 LLM 생성 | query, top_k |
| 2 | read_chunk | 전용 pointer head 실행, 인덱스를 실제 ID로 매핑 | chunk_ids 1~8개 |
| 3 | FINAL | answer prefix 후 LLM 생성 | answer, support_ids |

도구 선택 → 인자 완성 → 실제 함수 실행 → 호출과 결과를 다음 턴 입력에 기록함.
도구 번호는 텍스트 토큰으로 생성하지 않음. 검색 prefix는 현재처럼 모델 forward를 실제로 통과시키므로 이후 생성은 선택한 도구에 조건화됨.

### 동적 청크 head

고정 Linear(hidden,5)가 아니라 후보마다 같은 scoring 함수를 적용함.
현재 상태 h와 후보 표현 c_i를 받아 s_i=f(h,c_i,선택된 후보 요약)를 계산함. 후보 5개면 점수 5개, 8개면 점수 8개임.
read_chunk가 선택된 뒤에만 실행하며, 도구별 전용 분기 자체가 action 조건임. 첫 prefill의 hidden을 재사용하는 경로를 주안으로 함.
후보 표현은 현재 입력에 실제 보이는 각 청크의 title/snippet 또는 이미 읽은 본문 구간에서 pooling함. 아직 읽지 않은 원문을 corpus에서 몰래 가져오지 않음.
후보는 현재 관측 ID의 중복 제거 목록이며 정렬과 ID 매핑을 저장함. 최신 검색 5개로 임의 제한하지 않음.
구간 매핑은 tokenizer offset으로 검증하며 실패하면 명시적 오류 처리함. 관측 후보가 없으면 read_chunk를 마스킹함.

복수 읽기는 후보 선택을 반복하며 이미 선택한 후보는 마스킹하고 STOP으로 끝냄. 첫 선택 전 STOP 금지, 최대 8개 또는 모든 후보 선택 시 강제 종료함.
선택 순서를 보존하여 기존 read 결과 순서와 연결함. 반복은 작은 head 내부 연산이며 LLM 재호출이나 환경 tool call이 아님.
STOP은 전체 agent FINAL과 다름. 번호/ID 문자열은 생성하지 않고 실행기가 chunk_ids JSON을 조립함.

## 3. 고정 자산과 데이터

- LLM: Qwen/Qwen3-4B-Instruct-2507, revision cdbee75f17c01a7cc42f958dc650907174af0554. non-thinking.
- Embedding: google/embeddinggemma-2, revision 914f7f89142e33e77833254d9c9b90c3cef7303b.
- MuSiQue answerable v1.0 train/dev, shared corpus 101,962문단 및 768차원 인덱스.
- configs/split_mixhop.json 유지: train576(각 hop192), validation192(각64), test320(2hop160/3hop80/4hop80).
- 기존 dev192는 이미 관측된 개발셋임. untouched라고 부르지 않음. test320은 계획/설정 동결 전 열지 않음.
- 원본·모델 LFS·코퍼스·벡터 해시 검증 완료. RESTORE_VERIFIED.json 참고.
- base 모델과 BC/KL0 head 복원 완료. 이전 A1/B1의 full-FT 가중치는 이번 새 비교의 시작점으로 사용하지 않음.

## 4. 1차 — teacher 라벨 생성 → head KD

LLM은 고정하고 도구 head 및 청크 head를 지도학습함. teacher는 같은 base LLM의 native constrained 경로임.

수집 제안: frozen train576에서 hop별64, 총192문항. 이 안에서 학습144/내부검증48을 qid 단위로 사전 분리함. 질문 family 중복도 audit하고 확정 목록·seed·해시를 저장함. 아직 목록 선정은 하지 않음.

각 상태 라벨:
- 현재 messages/history, qid/turn, context hash, 합법 도구 mask, teacher 선택 도구.
- read 상태에서 현재 후보 ID 목록·순서·visible token spans와 teacher가 생성한 chunk_ids.
- teacher 선택 ID를 후보 인덱스 순서로 매핑하고 마지막 STOP 추가. 8개 선택/후보 소진의 강제 종료에는 STOP loss를 추가하지 않음.
- 오류/중복 ID/매핑 실패는 별도 기록하고 pointer 라벨에서 제외. 유효한 action 라벨은 별도 유지함. 성공 QA만 남기는 필터는 사용하지 않음.
- 정답·support labels는 scorer와 사후 분석용이며 policy 입력이나 KD 정답으로 섞지 않음.

KD는 우선 hard-label distillation(CE/behavior cloning)임. teacher soft distribution을 수집하지 않았는데 KL distillation이라고 부르지 않음. native collector의 dummy uniform policy_probs를 teacher target으로 사용 금지.

loss = 질문별 균형 action CE + read 질문별 균형 pointer sequence CE. pointer 손실은 read 상태에서만 계산함. 긴 episode/많은 선택의 과대 가중을 피하도록 상태 및 sequence 평균을 명시함.
LLM frozen, 기존 tool head의 trunk/action/skip과 새 pointer의 query/key/STOP/선택요약 파라미터 모두 학습함. 미사용 value/success/cost는 학습 대상에서 제거함.
초기값·학습률·epoch는 수집 분포 확인 뒤 내부 train/validation만으로 동결함. BC 시작과 KL0 시작을 섞어 비교하지 않음.

먼저 train에서 24문항 smoke. read 라벨 수, hop별 coverage, K 분포, 선택 수, STOP 비율, 오류율을 확인함. 192문항에도 read 라벨이 부족하면 coverage 미달로 보고하며 heldout을 가져오거나 무단 확대하지 않음.
기능 gate: K=1/5/8 및 그 이상 가변 길이 처리, ID mapping 100%, 범위 밖 선택/중복/빈 선택 0, future-content leakage 0. KD imitation 성능은 별도 지표이며 QA 개선과 구분함.

## 5. 2차 — LLM + 전체 head GRPO

1차 KD head를 동일한 base LLM에 붙임. LLM 전체와 사용되는 head 구조 전체를 unfrozen함.
문항당 rollout 8개. tool 선택, pointer 선택/STOP, LLM 인자·답 생성 모두 sampled policy action으로 기록함.

중요 구현 변경:
- 현재 run_fullft_grpo.py는 head action/skip만 학습하고 수집한 detached pack으로 head loss를 계산함. 그대로 사용 금지.
- 업데이트 때 당시 prefix로 현재 LLM hidden과 후보 표현을 gradient-enabled forward로 재계산해야 함.
- tool/pointer loss도 LLM backbone까지 역전파됨을 gradient smoke로 확인함.
- 수집 시의 old logprob는 고정하고, replay 시 동일 mask·후보 순서·선택 순서·temperature·grammar 분포로 new logprob를 계산함.
- 공동 logprob는 tool + pointer의 각 선택/STOP + 생성 토큰 logprob로 구성함. 강제 구문/강제 종료는 sampled decision loss에서 제외함.
- grammar fallback에서 실제 선택한 제한 분포와 학습 분포를 일치시키거나 해당 토큰을 명시적으로 제외함.

먼저 24문항×8×1회 pilot을 제안함. gate 통과 후 본 실행 제안은 iteration당 train192×8, 4회 반복. 최종 횟수/예산/평가 시점은 실행 전에 동결함.
각 iteration 모델·head를 저장함. 결과를 보고 유리한 iteration을 최종 판정점으로 바꾸지 않음.

## 6. 3차 — head 없는 LLM GRPO 대조군

2차 결과를 이어받지 않고 같은 base LLM에서 시작함. 도구 이름·read ID·검색 인자·답변을 LLM이 선택/생성함.
현재 native constrained trie 및 observed-ID schema를 유지하는 강한 대조군임. 자유 JSON 생성 baseline이라고 부르지 않음.
LLM 전체 unfrozen, 문항당 rollout8. train qids·질문 순서·iteration·보상·context/tool/output budget·채점·평가 시점을 2차와 동일하게 함.
KD는 head arm에만 있는 추가 지도비용이므로 이를 별도 보고함. head 구조만의 순수 인과 효과라고 주장하지 않음.
동일 sampling temperature를 초기안으로 두고 실제 entropy/중복 rollout/그룹 reward variance를 기록함. native만 다른 temperature를 쓸 경우 별도 사전 보정 근거를 남김.

## 7. 정확도 우선 보상 초안

공식 scorer의 alias-max normalized EM(E)와 token F1(F)을 사용함.

Q = 0.8 E + 0.2 F
R = Q * (1 - 0.10 T/(T+128)) - 0.15 E * N/8

- E는 0/1, F는 0~1. N은 실제 tool call 수(0~8), FINAL 및 head 내부 pointer 선택은 N에서 제외함.
- agent iteration은 통상 N+1이며 학습 iteration과 구분함.
- T는 episode 전체에서 실제 autoregressive decode한 토큰 수. 검색 인자, read ID, 답변, native 도구명 분기, grammar 대체 생성 포함. runtime이 공급한 prefix와 입력·tool 결과는 별도 집계함.
- 정답일 때만 호출 감점 적용. 토큰 감점은 정확도 점수에 비례하고 상한 10%.
- 정답 보상은 최소 0.75, 비정답 보상은 최대 0.2. 짧은 오답이 긴 정답보다 높아지지 않음.
- E=F=0이면 R=0이므로 빈 답/빠른 포기에 효율 보너스가 생기지 않음.
- 모든 rollout이 틀린 그룹은 신호가 없을 수 있음. 부분 F1을 남겨 완화하지만 해결 보장은 아님. fraction_zero_adv를 필수 보고함.
- 긴 부분정답의 점수 악용은 완전히 제거되지 않음. 긴 오답/정답 포함 부정문/answer stuffing을 별도 audit함.
- 계수와 128은 제안값이며 train-only offline reward sweep·순위검사 후 동결함. dev/test로 계수를 튜닝하지 않음.
- 파싱 실패/빈 답/잘린 출력은 품질0. truncation 별도 집계. 인프라 실패는 모델 오답과 구분하여 사전 재시도 규칙 적용.

GRPO advantage 초안: A_i=R_i-mean(R_1..R_8). 그룹별 std로 나누지 않음. 항별 재표준화로 계수 의미를 없애지 않음.
PPO clip, KL, loss 정규화도 양 arm에서 같은 원칙으로 고정함. 생성 토큰 수와 head decision 수의 차이가 gradient 규모에 미치는 영향을 smoke에서 확인하고 고정 normalization을 기록함. 이 부분은 구현 전 확정 필요 항목임.

근거: MuSiQue 공식 metric은 EM/F1 및 alias-max를 제공함. ShorterBetter는 correctness를 고려한 길이 절감을 연구하고, DAPO는 overlong 처리와 길이 편향 문제를 다룸. 위 식은 논문 식의 재현이 아니라 이번 PoC용 제안임.
- https://github.com/StonyBrookNLP/musique/blob/main/metrics/answer.py
- https://arxiv.org/abs/2504.21370
- https://arxiv.org/abs/2503.14476

## 8. 사전 검증과 보류

CPU: ID/span/schema/mask 계약, 보상 순위, split disjointness.
GPU smoke: cache prefix 경로, 실제 ID dispatch, tool/pointer/LLM 모든 사용 층의 유한 gradient, 업데이트 전 replay old/new logprob 일치 허용오차를 dtype별 설정, 전체 checkpoint 저장/재로드 동일 greedy 출력.
공유 GPU latency는 통제된 speedup으로 주장하지 않음. 다른 작업을 종료하지 않음.
실험은 현재 미시작. 모델/데이터 파일은 확보했으나 Python runtime·실행 smoke 미완료.
