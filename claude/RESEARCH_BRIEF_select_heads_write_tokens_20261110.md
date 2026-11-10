# 연구 브리프: Select with Heads, Write with Tokens

작성 2026-11-10 · 용도: 다른 AI/공동연구자에게 맥락 없이 설명하기 위한 문서
상태: 아이디어 단계. PoC 1건 완료, 아래 "미확인" 표시는 아직 측정·조사하지 않은 것.

> 저장 메모: 사용자가 2026-11-10 Claude 세션에 붙여 넣은 원문을 그대로 보관함. Claude가 수정·검토한 내용 아님.

---

## 1. 한 문단 요약

LLM agent는 도구 선택, 대상 ID(읽을 chunk, 고칠 메모리 항목), 인자 내용을 모두 텍스트 토큰으로 생성한다. 우리는 이 중 **이산 선택(도구, 대상 인덱스)** 을 reader LLM의 hidden state 위에 붙인 selection head와 pointer로 옮기고, **내용(검색 쿼리, 메모리 문장)** 만 텍스트로 생성한다. 별도 decision model 없이 한 모델이 한 번의 forward에서 판단과 생성을 함께 한다. 주장하려는 이점은 (1) ID를 텍스트로 옮겨 적다 생기는 오류의 구조적 제거, (2) 결정별로 분해된 학습 신호, (3) 결정 토큰 생성 비용 제거다.

## 2. 배경과 동기

- **판단 전용 모델의 한계:** 분류기형 decision model은 "무엇을 할지"는 고르지만 인자(쿼리, 메모리 내용)를 생성하지 못한다. 실제 agent 흐름은 판단 직후 생성이 필요하므로 모델 두 개를 붙여 써야 하고, 같은 문맥을 두 번 읽는다.
- **텍스트 결정의 한계:** 결정을 텍스트로 쓰면 존재하지 않는 ID 생성, 숫자 오기, JSON/XML 형식 깨짐이 생길 수 있고, 그 step은 헛돌거나 엉뚱한 대상을 건드린다.
- **RAG에서 메모리로:** RAG는 연구가 포화 상태라, 결정 종류와 횟수가 더 많은 agent 메모리 관리로 적용 범위를 넓힌다.

## 3. PoC 결과 (완료)

- 환경: A-RAG (arXiv 2602.03442). 도구 3개: keyword_search, semantic_search, chunk_read. 원 논문은 학습 없는 프롬프팅 프레임워크.
- 우리 설정: 여기에 GRPO 학습을 얹고, 도구 선택을 텍스트 대신 MLP head 출력(action index)으로 대체. backbone과 head를 함께 학습.
- 데이터: MuSiQue 평가셋을 나눠 train/eval로 사용한 소규모 실험.
- 보상: 정답 보상 + 짧은 토큰 보상.
- 결과: 정확도 상승, 생성 토큰 약 3~4% 감소, iteration 소폭 감소.
- 관찰: 길이 페널티를 키우면 정확도가 같이 무너짐. 정답 궤적에만 길이 보상을 주는 등의 변형도 큰 차이 없었음.
- 해석(가설): MuSiQue는 2~4 hop 문제라 필요한 검색 횟수 자체가 많아 줄일 여지가 작을 수 있음. 모델의 평균 검색 횟수와 hop 수의 비교는 **미확인**.

## 4. 제안 구조

### 4.1 원칙
선택과 인덱스는 head로, 내용은 텍스트로.

### 4.2 구성 요소
```
입력(질문 + 문맥 + 후보들) → LLM backbone → hidden states
   ├─ Tool head (MLP, softmax)   : 어떤 도구/연산을 할지
   ├─ Pointer head (attention)   : 후보 중 어느 대상인지
   └─ LM head (기존)             : 쿼리·메모리 내용 텍스트 생성
```
- **Tool head:** 결정 시점(추론 텍스트가 끝난 위치의 마지막 토큰) hidden state h를 입력으로 행동 분포를 낸다.
- **Pointer head:** 문맥 안의 후보 구간(검색 결과로 나온 chunk 미리보기, 메모리 항목)마다 대표 hidden state e_i를 뽑고(구간 끝 토큰 또는 pooling), score_i = f(h)·g(e_i)로 점수를 매겨 고른다. 후보 수가 변해도 동작한다. 여러 개를 고를 때는 후보별 sigmoid.
- **생성 조건:** head가 고른 행동과 대상을 템플릿 토큰으로 문맥에 넣고, LM head는 내용만 이어서 생성한다. 기존 파서와 환경 코드를 거의 그대로 쓴다.

### 4.3 환경별 매핑

| | A-RAG (검색) | AtomMem (스트리밍 메모리) |
|---|---|---|
| Tool head | keyword_search / semantic_search / chunk_read / answer | Create / Read / Update / Delete / NOOP |
| Pointer | chunk_read 대상 (직전 검색 결과 후보 중) | Update·Delete 대상 메모리 항목 |
| 텍스트 생성 | 검색 쿼리, 최종 답 | Create·Update 내용, Read 쿼리 |
| 생성이 0인 step | chunk_read | Delete, NOOP |

AtomMem (arXiv 2601.08323, EMNLP 2026): Qwen3-8B, 문서를 4K 토큰 조각으로 나눠 순서대로 입력, 메모리 연산을 XML 태그 텍스트로 생성, SFT 후 GRPO. HotpotQA·2Wiki·MuSiQue의 장문 버전(200/400/800 문서)으로 평가. 과제당 97.6초로 MemAgent(49.7초)의 약 2배.

## 5. 기여 3개

**C1. 구조:** 도구 선택 head와 대상 pointer를 LM head와 같은 hidden state에 붙인 단일 모델. 판단과 생성을 한 번의 forward로.

**C2. 오류 분해:** 텍스트 방식의 대상 지정 오류 = 선택 오류(형식은 맞지만 엉뚱한 대상) + 전사 오류(없는 ID, 틀린 숫자, 형식 깨짐). pointer는 존재하는 후보만 고르므로 전사 오류가 구조적으로 0. 두 방식의 오류를 이 분류로 세어 전체 오류 감소를 보인다.

**C3. 구조에 맞춘 학습:**
- 정책 분해: log π = log π_tool(a|h) + log π_ptr(i|h, 후보) + Σ log π_LM(토큰)
- 결정마다 별도 advantage 부여 (도구, 대상, 내용).
- 보조 지도: 데이터셋의 근거 문단 표시로 "읽었어야 할 chunk", "근거가 다 모였으니 answer"를 값싼 라벨로 만들어 pointer와 answer 확률을 직접 학습. 희소한 최종 보상을 보완.
- 기대 효과(가설): 모든 head가 같은 표현을 공유하므로 관련성 판단 능력이 결정들 사이에 전이.

## 6. 검토

### 6.1 합리성: 높음
- 주장이 측정 가능하고 실험이 싸다. 오류율 측정은 기존 로그로 하루, frozen feature pointer 실험은 하루 이틀.
- 핵심 불확실성은 **문제의 크기**다. 판단 기준(제안): 전사 오류를 한 번이라도 겪은 문제가 5% 이상이고 그 문제들의 정확도가 뚜렷이 낮으면 동기 성립, 1% 미만이면 동기 불성립.
  - step당 2% 오류라도 문제당 도구 호출이 10회면 약 18%의 문제가 영향을 받는다(1 − 0.98^10).
  - 소형 모델(3~8B)을 RL로 학습할 때 오류가 늘 것이라는 건 추정이며 **미확인**.
- head 자체도 틀릴 수 있다. 다만 텍스트 방식과 같은 hidden state 정보를 쓰므로 선택 오류가 더 커질 이유는 약하고, 직접 라벨로 학습할 수 있다. 이것도 실험으로 확인해야 한다.

### 6.2 창의성: 중간
- 부품은 모두 기존 기술이다: pointer network, hidden-state head, constrained decoding.
- **가장 위험한 반론:** "JSON schema/constrained decoding으로 ID를 후보 집합으로 제한하면 전사 오류는 이미 0이다." 이 베이스라인 대비 이점(생성 토큰 0, 동시 다중 선택, 직접 지도 가능, calibration된 확률)을 실험으로 보여야 한다. 이 방향의 선행 연구 조사는 **미실시**.
- 도구 몇 개 중 하나를 고르는 것만 보면 head는 단일 태그 토큰과 수학적으로 거의 같다. 구조적 차이는 가변 후보 pointer와 동시 결정에서만 나온다.
- credit 분리도 텍스트 방식에서 토큰 마스크로 가능하다. head 고유의 이점이 아니다.

### 6.3 조사에서 확인된 인접 연구
- 텍스트 gate + RL: GRU-Mem (2602.10560), InfMem (2602.02704, 학습된 STOP으로 MemAgent 대비 추론 시간 평균 3.9배 단축)
- frozen backbone + head: Multi-Head Latent Control (2607.14277), When2Tool (2605.09252), Probing-RAG (2410.13339)
- hidden state로 도구 선택: Chain-of-Tools (2503.16779)
- 별도 router: Gated-Memory Routing (2609.00237)
- "joint 학습된 pointer head로 메모리 대상을 고르는 agent"는 조사 범위에서 찾지 못함(부재 증명은 아님).

## 7. 보류·제외한 아이디어

| 아이디어 | 상태 | 이유 |
|---|---|---|
| Q-head 가상 분기: 행동별 가치를 head가 예측해 트리 탐색을 rollout 없이 대체 | 보류(유력한 추가 기여) | 학습 방법 기여로 강함. Q 학습 불안정 위험 |
| 쿼리 벡터 직접 출력 (텍스트 쿼리 없이 검색) | 제외 | 연구자 판단: 인자 내용은 텍스트 유지. 생성 과정의 계산을 건너뛰는 위험 |
| 얕은 layer 결정으로 prefill 절감 | ablation | URaG, Skim&Skip(2609.01613)과 구조 유사 |
| 연산량 배분 프레임 | 폐기 | 프레임 자체는 이미 통용 |
| All-Action Forking | 학습 도구로만 | 텍스트 방식도 태그 강제로 가능, head 고유 아님 |
| agent 간 latent 통신 | 제외 | 연구자 방침 |

## 8. 실험 계획

**0단계 (학습 없음, 1~2일)**
- PoC 로그에서: ① ID·형식 오류 step 비율 ② 그런 오류를 겪은 문제 비율 ③ 그 문제들의 정확도 ④ 형식은 맞지만 엉뚱한 ID를 고른 비율
- 모델의 평균 검색 횟수와 MuSiQue hop 수 비교

**1단계 (frozen feature, 2~3일)**
- 결정 시점 hidden state와 후보 hidden state를 저장, 근거 문단 표시를 정답으로 pointer만 학습
- 텍스트로 고른 ID 대비 정확도 비교

**2단계 (joint SFT → RL)**
- A-RAG, 이어서 AtomMem(HotpotQA 200문서 설정부터)
- 비교군: ① 원본 텍스트 방식 ② 텍스트 + constrained decoding ③ frozen backbone + head ④ 제안(joint)
- 지표: 정확도(EM/F1), 오류 분해(선택/전사), iteration 수, 생성 토큰, wall-clock, pointer 정확도

**판단 지점**
- 0단계에서 오류 영향 문제 비율이 낮으면: 효율(생성 0인 step, 동시 선택)과 Q-head 쪽으로 무게 이동
- 2단계에서 ②를 못 이기면: 구조 기여 주장 불가

## 9. 종합 판단

방향은 타당하고 검증 비용이 낮다. 현재 구성만으로는 "깔끔한 엔지니어링 개선"으로 읽힐 가능성이 크다. 상위 학회 수준이 되려면 (a) 0단계에서 오류가 실제로 의미 있는 크기이고 (b) constrained decoding 베이스라인을 이기며 (c) Q-head 같은 학습 방법 기여가 하나 더 붙어야 한다.

목표: ICML 2027 (마감 2027-01-22 AoE, 전년도 기준 추정이며 공식 공지 전) · 대안: NeurIPS 2027
자원: 8B 이하 모델, GPU 노드 1개 수준 가정

## 10. 검토자에게 묻고 싶은 것

1. constrained decoding 대비 pointer head의 이점이 논문 기여로 충분한가?
2. 전사 오류가 작게 나올 경우, 이 구조를 살릴 다른 동기가 있는가?
3. pointer와 tool head를 backbone과 joint 학습할 때 예상되는 불안정 요인과 대응은?
4. Q-head를 붙인다면 Q target을 어떻게 값싸게 만들 것인가?
