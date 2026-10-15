# jev_v1.1 Astra 검토 — 연구 주장 판정과 좁힌 실험안

작성: 2026-10-15 UTC. 사용자 요청에 따른 Astra 독립 검토. 대상은 [연구 제안](JEV_ICML2027_RESEARCH_PROPOSAL.md), [PoC 기록](../lab/reports/jev_v1.0/PROGRESS.md), [학습 감사](../lab/reports/jev_v1.0/HEAD_TRAINING_AUDIT.md), 현재 `decision.py`·`hf_reader.py`·`agent.py`다. 코드 읽기와 논문 본문 검토만 수행했다. 학습, GPU 실험, 논문 재현, git 변경 작업은 수행하지 않았다. 아래 방법과 수치는 실행 결과가 아니라 설계·판단 기준이다.

## 1. 판정

**현재 PoC의 제한적 해석은 APPROVED. 기존의 '멀티모달 지연 효용을 작은 head로 학습'을 독창적 알고리즘으로 내세우는 안은 REJECT. 다음의 좁힌 가설을 확인하는 2주 pilot은 CONDITIONAL. ICML 기여가 확보됐다는 승인은 아니다.**

핵심 문제는 충분히 가치 있다. 긴 행동 설명을 생성하지 않고 tool/answer를 고르면서, 증거를 조금 더 확보하면 풀릴 문제를 너무 일찍 포기하지 않게 하는 것이다. 하지만 작은 head, multimodal routing, 미래 value, 같은 prefix에서 분기한 supervision을 합친 것만으로는 기존 연구의 조합을 넘었다고 보기 어렵다.

추천을 다음처럼 좁힌다.

> **현재 증거가 다음 검색에 주는 효과를 답변에 직접 주는 효과와 구별하는 학습 신호가, 같은 비용과 데이터의 일반 value learner보다 빠른 제어기의 새로운 문서·개체에 대한 일반화를 개선하는가?**

우선 **증거 접근을 query generator와 answer reader에서 교차 통제하는 실험**을 한다. 결과가 재현되고 예측 가능한 경우에만 그 대조를 auxiliary supervision으로 사용하는 작은 controller를 훈련한다. 추론 시에는 관측된 현재 상태를 한 번 읽고 행동 하나만 선택한다. 미래 branch·증거 제거·추가 reasoning을 실행하지 않는다.

이 구체안도 아직 novelty가 확정된 방법이 아니다. 일반 Q-learning을 다른 이름으로 바꾸지 않으며, 개입 실험·차분 loss·multitask learning 자체도 기여로 주장하지 않는다. 기여 후보는 **어떤 학습 신호가 기존 terminal-return supervision에 없는 유용한 구분을 제공하는지와, 그 신호가 실제 순차 제어에서 작동한다는 근거**다.

## 2. 현재 PoC와 구현의 정확한 해석

### 2.1 승인할 수 있는 것

- Frozen Gemma의 마지막 정규화 hidden에서 4-action linear head를 읽고, 동일 호출의 prefill KV를 사용해 선택된 tool의 인자나 최종 답변을 생성하는 구현이다. [HFReader](../src/lsdrag/hf_reader.py)의 hook·prefill·prefix 경로가 이 설명과 일치한다.
- [상태 복원](../src/lsdrag/decision.py)은 현재 assistant target을 넣기 전에 상태를 산출한다. prompt hash, tool/result 대응, 지원하지 않는 이력의 거부가 있다. 읽은 코드에서 명백하게 정답 또는 미래 assistant 답을 입력에 넣는 경로는 발견하지 않았다. 전체 원자료의 무누출을 새로 인증한 것은 아니다.
- 45개 고유 성공 질문·197턴으로 head를 학습했고, 새 20문항의 기록에서 전체 시간 7.273→5.048초를 관측했다는 보고는 유지할 수 있다. EM 1→2의 유일한 개선은 출력 형식 차이이며, 정확도 향상 근거로 쓰면 안 된다.

### 2.2 표현과 구현에서 바로잡을 것

1. **한 번의 prefill은 작은 모델 한 번이 아니다.** 전체 VLM과 이미지 encoding을 실행한다. 현재 호출은 `logits_to_keep=1`로 vocabulary logits도 계산하고, norm hook은 decode 중에도 마지막 hidden을 복사한다. head FLOPs만으로 시스템 비용을 설명하지 않는다.
2. **KV 재사용은 현재 호출 안에서다.** iteration 사이에는 `chat()`이 전체 누적 messages를 다시 prefill한다. 'agent 전체에서 cache를 증분 재사용한다'고 쓰면 틀린다. 이를 개선한다면 모든 controller의 공통 backend 개선으로 분리한다.
3. **prefill 계측에는 preprocessing이 포함되지 않는다.** `prepare()`가 timer보다 먼저 실행된다. 전체 `reader_wall_seconds`와 agent wall time에는 포함되므로 전체 지연 지표는 유용하지만, 세부 분해에는 preprocessing·이미지 전달·파싱 등 잔여 시간을 따로 둔다.
4. **4-action head는 일반적인 tool router가 아니다.** 고정 `ACTIONS`는 keyword/semantic/read/FINAL이고 V2의 image_search/figure_read는 다루지 못한다. 지원 action mask와 schema 일치 검증은 후속 구현 전 필수다. 없는 schema를 찾는 경로의 예외도 점검해야 한다.
5. **FINAL은 답변 후 종료다.** 독립적인 abstain/quit 행동이 학습된 것은 아니다. 새 실험도 우선 `ANSWER_AND_STOP` 하나로 정의한다. abstention을 추가하려면 정답 불가능 문항·보상·평가를 새로 정의해야 하므로 기본 범위에 넣지 않는다.
6. **성공 궤적만 imitation하는 것은 '최적 행동' label이 아니다.** 성공 중 불필요한 검색도 양성 label이 될 수 있고, 실패·복구 불가 상태에서 멈출 학습 신호가 없다. 197/197 일치율은 이 선택된 행동을 외울 수 있다는 결과다.
7. **높은 action probability는 성공 확률이 아니다.** 기존 FINAL≥0.9인데 EM=0인 다수 사례와 모순되지 않는다. bounded output을 제공한다는 요구와 보정된 정답 확률을 제공한다는 요구는 다르다.

새 구현 전에 상태 hash에 text·이미지 bytes·tool schema·prompt·모델/processor 설정을 포함하고, 실제 실행 action과 fallback 전 선택 action을 구분해야 한다. 이 검토는 코드를 수정하지 않았다.

## 3. 원문으로 확인한 독창성 경계

다음은 이번 검토에서 primary 문서를 직접 연 결과다. 저자 주장과 이 저장소의 재현 결과를 구분한다.

| 원문 | 실제 중복 범위 | 남는 검증 질문 |
|---|---|---|
| [Stop-RAG §3–4](https://arxiv.org/html/2510.14337v1) | full-width forward-view Q(λ), 미래 return, prefix별 STOP/CONT 보상. 자체 데이터 수집에는 반복 답변 채점과 무신호 상태 필터링도 있다 | multi-action 확장·2-hop target 자체는 신규 방법이 아니다. 동일 데이터의 일반 future-value 학습이 충분한가? |
| [Calibration Is Not Control §2–4](https://arxiv.org/html/2606.21399v1) | failure confidence와 intervention advantage를 구분하고, 동일 prefix branching으로 action utility와 regret를 평가한다 | 'confidence 대신 utility', 'counterfactual branch 학습'은 이미 존재한다. 새로운 관측/학습 신호가 무엇인가? |
| [General Decision Models / InnerJev §6](https://arxiv.org/html/2610.03935v1) | option-label의 first-token readout과 reasoning-to-readout self-distillation | head·bounded decision·CoT 압축은 새 기여가 아니다. 같은 reader의 first-token baseline을 이기는가? |
| [MC-Search §2.1 식(2)–(3), §3](https://arxiv.org/html/2603.00873v1) | evidence 제거의 context utility와 downstream sub-question의 entity overlap을 이용한 navigational role을 구분한다. Search-Align은 chain process supervision이다 | navigation 효과를 처음 발견했다고 할 수 없다. entity-overlap proxy를 넘어 module별 실제 접근 개입이 제어 학습에 추가로 유용한가? |
| [CLIMB §3](https://arxiv.org/html/2610.03421v1) | fixed complementary micro-pool, MMR, critic 및 confidence에 따른 refinement | 모든 종류의 상보성·정지 문제를 이미 해결한 것은 아니지만, 상보적 증거와 종료의 결합 자체는 새롭지 않다 |
| [UniversalRAG 초록](https://aclanthology.org/2026.acl-long.177/) | modality·granularity routing, heterogeneous corpora, unified embedding의 modality bias 문제 | 멀티모달 임베더+router 조합만으로 부족하다. 본 검토에서는 초록 이상 전체 방법을 재감사하지 않았다 |
| [CUE-R §3, §6.8](https://arxiv.org/html/2604.05467v1) | single-shot RAG의 remove/replace/duplicate evidence intervention, multi-axis utility, 두 근거의 비가산적 효과 | evidence ablation이나 pair synergy를 새 평가 원리라고 할 수 없다. 순차 검색의 controller generalization과 연결해야 한다 |
| [VAD §3](https://arxiv.org/html/2607.28590v1) | 같은 student prefix에서 evidence-present/removed teacher view로 supervision을 구성하는 visual attribution distillation | '시각적 반사실을 학습 때만 사용한다'는 넓은 주장도 중복. 여기서 필요한 target은 next-token correction이 아니라 특정 acquisition의 value 차이다 |
| [Active Acquisition for Multimodal Temporal Data 초록](https://arxiv.org/abs/2211.05039) | 비용에 따라 관측 modality를 선택하는 오래된 active acquisition 문제와 cross-modal reasoning | 비용을 고려한 modality 선택을 새 문제라고 할 수 없다. adaptive policy의 실제 state dependence를 보여야 한다 |

InnerJev §5.3은 빠른 typical episode와 성공률 감소·tail 악화를 함께 보고한다. Table 3과 바로 아래 본문에는 성공률·지연 수치가 서로 다른 부분이 실제 존재한다. 방향성만 인용하거나 table/version을 특정해야 하며 수치를 섞으면 안 된다. [원문 §5.3](https://arxiv.org/html/2610.03935v1)

새 개입안의 독창성 검색은 충분히 끝나지 않았다. causal mediation, active feature acquisition, evidence attribution, representation invariance, counterfactual policy learning에는 훨씬 넓은 선행이 있다. 검색에서 확인한 [TAP-RAG](https://arxiv.org/html/2607.18917v1)의 task-conditioned policy와 selective visual inspection도 대안의 중복 감사 대상이다. 이를 '관련 논문이 없으니 새롭다'로 해석하지 않는다.

## 4. 먼저 구분해야 하는 두 현상

초안의 '첫 evidence는 답변을 개선하지 않지만 두 번째와 합치면 개선한다'에는 다른 원인이 섞여 있다.

- **검색을 가능하게 함:** 첫 이미지로 entity를 알아야 다음 text query를 만들 수 있다. 마지막 답변에는 첫 이미지가 다시 필요하지 않을 수 있다.
- **답변에서 함께 필요함:** 두 개의 이미지를 비교하거나 이미지 속 속성과 텍스트 속 사실을 결합해야 한다. 둘 중 하나가 없으면 마지막 답변도 어렵다.

그 밖에 단순히 더 많은 문서, 답변 생성의 변동, context 길이, query generator의 우연한 성공으로도 pair gain이 생긴다. 따라서 `U(s_ab)>U(s_a)`만으로 cross-modal complementarity를 판정하면 안 된다. text→text·image→image의 동등 깊이 대조도 있어야 한다.

그리고 **현재 얻은 evidence가 다음 검색에 중요하다는 것**과 **아직 얻지 않은 evidence의 가치를 지금 예측할 수 있다는 것**은 별개다. 아래 개입은 전자를 직접 다룬다. 후자의 첫 행동 결정까지 개선한다는 주장은 별도 closed-loop 실험을 통과해야 한다. 이를 한 문장으로 합치지 않는 것이 이번 범위 축소의 핵심이다.

## 5. 구체적인 최소 실험: 모듈별 증거 접근의 2×2 대조

### 5.1 실험 단위와 입력 경계

고정된 behavior mixture가 실제 검색한 evidence를 받은 직후, 아직 그 evidence를 사용해 새 assistant 발화를 만들기 전을 분기점으로 삼는다. 공통 과거 상태를 `B`, 새로 받은 관측을 `e`라 하자. `S₁=B+e`, `S₀=B`다. `e`는 정답이나 gold chain으로 골라서는 안 된다. 예를 들어 각 문항의 사전 고정된 첫 두 acquisition boundary, 반환 순서 첫 번째 unique item처럼 **결과를 보기 전에 정한 관측 규칙**을 사용한다. 실패·무효·무이득 상태도 남긴다.

두 상태의 이미 소비한 acquisition 횟수·남은 실행 budget·legal action mask는 같게 둔다. `S₀`는 검색을 실행하지 않은 과거 상태가 아니라 **같은 boundary에서 e의 접근만 제거한 상태**다. 별도의 intervention 정답표시나 성공 여부 flag를 controller 입력에 넣지 않는다.

검색 행동 `a`는 modality 이름만이 아니다. `g_a(S)`라는 고정 query generator, corpus, retriever, top-k, 실제 evidence 전달 규칙까지 포함한 **acquisition kernel**로 고정한다. 모든 controller가 이 kernel을 공유한다.

각 `S_j`로부터 query를 생성하고 실제 검색해 `D_j=R_a(g_a(S_j))`를 얻는다. final Reader에는 질문·공통 배경 `B`·선택된 `e`·실제 반환 `D_j`만 전달하고 utility `U_ij`를 측정한다. `i`는 Reader의 `e` 접근, `j`는 query generator의 `e` 접근이다.

| final Reader 입력 | query generator에 e 없음 → D₀ | query generator에 e 있음 → D₁ |
|---|---|---|
| e 없음 | U₀₀ = score(Reader(B,D₀)) | U₀₁ = score(Reader(B,D₁)) |
| e 있음 | U₁₀ = score(Reader(B,e,D₀)) | U₁₁ = score(Reader(B,e,D₁)) |

이 표는 **module-access intervention의 결과**다. natural direct/indirect causal effect, 모델 내부의 참 reasoning mechanism, 순수한 navigation/integration의 유일한 분해라고 부르지 않는다.

### 5.2 반드시 막아야 하는 숨은 정보 경로

- query generator가 쓴 query, CoT, intermediate answer를 final Reader에 전달하지 않는다. 그렇지 않으면 e를 Reader에서 숨겨도 query에 복사된 내용으로 다시 전달된다.
- 검색 결과는 실제 corpus에서 반환된 내용만 쓴다. query를 덧붙인 tool wrapper, generated summary, query-dependent 제목/설명, echo field를 제거한다. 원문 ID/hash·실제 content를 기록한다.
- `D₁`에 `e`와 같은 정보가 원래 corpus 문서로 다시 나타날 수는 있다. 이는 검색 경로를 통한 재획득이지 독립적인 direct-effect 증명은 아니다. 동일 항목/중복 내용 재검색률과 copied-content 진단을 별도 보고한다.
- 분기 전에 이미 e의 내용이 assistant text·caption·OCR·다른 document에 복제되었다면 단순 이미지 삭제는 유효한 'e 없음' 개입이 아니다. 최초 노출 경계를 우선 사용하고 residual information을 기록한다.
- 이미지와 caption을 함께 제거한 개입은 evidence-package 효과다. image-only 제거, caption-only 제거, text-only 치환 대조 없이 이를 **시각 정보만의 효과**라고 부르지 않는다.
- 삭제로 context 길이·위치·image count도 달라진다. 삭제와 유사 크기 distractor 치환을 구분하고 결과의 일관성을 점검한다. 원본 상태를 자연분포 평가에, 개입 상태를 별도 진단에 사용한다.

모듈 입력을 분리하려면 기존 full-message reader와 다른 입력 경계가 생긴다. 실제 controller 평가에서도 동일 경계를 적용하거나, 이 표를 별도의 진단으로만 보고해야 한다. 진단 환경의 효과를 원래 agent에 바로 전이된다고 쓰면 안 된다.

### 5.3 측정할 대조와 한계

단순한 관측 대조는 다음과 같다.

```text
N₀ = U₀₁ − U₀₀          # Reader에는 e 없이, query generator의 접근만 변경
I₀ = U₁₀ − U₀₀          # 검색 결과를 D₀로 고정하고 Reader의 접근만 변경
J  = U₁₁ − U₁₀ − U₀₁ + U₀₀
U₁₁ − U₀₀ = N₀ + I₀ + J
```

마지막 등식은 산술 항등식이다. 새로운 정리나 causal identification theorem이 아니다. 반대 기준점의 `N₁=U₁₁−U₁₀`, `I₁=U₁₁−U₀₁`도 함께 보고해 기준점 의존성을 숨기지 않는다. 음수 효과도 그대로 보존한다.

EM의 단일 0/1 차이는 기대 utility가 아니다. 주요 정답 지표와 의미상 답변 판정 규칙을 사전 고정하고, 필요하면 고정된 stochastic decoder의 독립 trial을 사용한다. greedy decode를 여러 번 반복해 독립 표본 수가 늘었다고 세지 않는다. 반복 비용이 크면 deterministic pipeline의 realized outcome이라는 범위를 명확히 둔다.

`D₀=D₁`인데 동일 Reader 입력의 score가 다르면 generation noise 또는 실행 비일관성부터 조사한다. query가 달라졌다는 사실만으로 유용한 검색이 발생했다고 판정하지 않는다.

## 6. 구체적인 학습안: 일반 value learner에 접근 대조 supervision 추가

### 6.1 표현과 행동

행동 집합은 우선 `ANSWER_AND_STOP`, `TEXT_ACQUIRE`, `IMAGE_ACQUIRE`다. 각 acquisition이 실제로 전달하는 evidence와 비용을 명확히 정의한다. 원래 keyword/semantic/read와는 다른 인터페이스이므로 원형 A-RAG와 공통 인터페이스 baseline을 따로 기록한다.

표현은 같은 frozen VLM prefill의 마지막 hidden `h`, 이미 관측한 evidence slot의 pooling `z_e`, 현재 남은 tool/image/context budget 및 step의 작은 metadata로 제한한다. pooling을 추가하면 같은 token positions·metadata·연산을 모든 learned baseline에 제공한다. retriever embedding은 후보 표현이며 **이미지 내용을 읽은 Reader 표현과 동등하다고 가정하지 않는다**. 선택되지 않은 tool 결과, gold support, 미래 query·answer·U값은 입력에서 제외한다.

작은 공유 MLP trunk 위에 action return head와 두 개 정도의 auxiliary contrast head를 둔다. 처음에는 마지막 hidden만 사용하는 구현과 비교하고, 새로운 attention/GNN/memory encoder를 동시에 만들지 않는다. 현재 코드의 norm hook을 확장할 때 선택된 token pooling의 메모리·시간도 계측한다.

### 6.2 target을 정확히 정의하기

`S_i`에서 바로 답하는 score를 `V_i`라 하고, acquisition a 뒤 바로 답할 때의 비용 보정 target을 다음처럼 둔다.

```text
y_i(a)    = U_ii(a) − λ c_i(a)
y_i(STOP) = V_i − λ c_i(STOP)
d_a       = [y_1(a) − y_1(STOP)] − [y_0(a) − y_0(STOP)]
```

`c`는 같은 하드웨어·backend에서 측정한 query generation, retrieval, evidence 전달, 다음 prefill, 최종 answer 등 **현재 이후 비용**이다. 이미 사용한 비용은 sunk cost이고 남은 budget에는 영향을 준다. 단위는 score와 초를 연결하는 λ로 명시한다. λ는 development에서 고정한 작은 grid를 쓰고 최종 QA–latency curve를 별도로 제시한다.

`U₀₁`, `U₁₀`은 일반 acquisition kernel에서 자연스럽게 얻은 transition이 아니다. 이를 `Q(S₀,a)` 또는 `Q(S₁,a)`의 정상 action return로 잘못 넣으면 안 된다. off-diagonal 셀은 접근 대조 target과 진단에만 쓰며, 해당 terminal context의 answer supervision으로 사용하는 baseline은 별도로 명시한다.

최소 loss는 다음과 같다. ρ는 Huber 등 사전 선택한 regression loss다.

```text
L_return = Σ_i,a ρ(Qθ(S_i,a) − y_i(a))
Aθ(S,a)  = Qθ(S,a) − Qθ(S,STOP)
L_pair   = Σ_a ρ([Aθ(S₁,a) − Aθ(S₀,a)] − d_a)
L_access = Σ_a [ρ(nθ(S₁,e,a) − N₀(a)) + ρ(jθ(S₁,e,a) − J(a))]
L        = L_return + α L_pair + β L_access
```

`L_pair`는 기존 return target의 차분을 재강조하므로 그 자체가 새로운 정보는 아니다. **정답·정책 label의 추가 정보는 off-diagonal 실행에서 온다.** `L_access`는 같은 공유 trunk에 그 신호를 주고 배포 시 auxiliary 출력을 요구하지 않는다. 이것이 낮은 데이터 조건에서 유용한 inductive bias라는 가설을 검증하는 것이다. 대조 예측만 좋아지고 QA–latency가 좋아지지 않으면 제어 방법의 성공이 아니다.

이 최소안은 **현재 얻은 evidence를 이용한 다음 행동**에 초점을 맞춘다. 두 단계 future return까지 넣으려면 일반 baseline과 제안안 모두 동일한 continuation policy·horizon·data를 받는다. 후속 행동을 noisy observed best-of로 선택한 값을 참 기대 최적 return로 쓰지 않는다. 질문/source fold 밖에서 학습한 continuation policy를 고정하고 평가 branch를 새로 실행하거나, 명확히 realized oracle target이라고 제한한다. 2주 pilot에서는 우선 한 단계 접근 대조가 학습되는지 확인하는 편이 낫다.

### 6.3 배포와 bounded output

배포는 legal action에 대해 `argmax Qθ(S,a)`를 선택하고, 필요한 query 인자 또는 최종 답변만 공통 VLM이 생성한다. 매 step 예산을 mask한다. softmax(Q/τ)를 노출할 수 있지만 이는 **policy preference**이며 정답 확률·최적 행동 확률이라고 부르지 않는다. τ 보정도 이 의미를 자동으로 바꾸지 않는다.

선택된 action의 prefix를 같은 prefill KV에 붙이는 현재 방식을 재사용한다. 별도 작은 encoder를 추가해 reader prefill까지 절약한다는 변경은 서로 다른 연구축이므로 이 단계에 넣지 않는다.

## 7. 기여를 인정하려면 필요한 가장 강한 비교

| 비교군 | 동일하게 맞출 것 | 이 비교가 무너지면 |
|---|---|---|
| 같은 VLM의 생성형 controller | reader/retriever/context/최종 형식/인자 생성/실행 backend | 전체 시스템 이득을 주장하지 못함 |
| constrained first-token readout | action semantics, 실제 single-token label, option 순서, 선택 후 같은 native prefix 경로 | 긴 action 설명 제거 이상의 이득이 불분명 |
| imitation linear/MLP | 상태 표현·파라미터 예산 | 단순 강한 classifier의 효과일 수 있음 |
| query-only 및 fixed budget/항상 modality 정책 | 같은 query, acquisition 예산, Reader | state adaptation을 입증하지 못함 |
| immediate-return 및 multi-action future-value learner | 같은 lawful branch returns·horizon·cost·입력·학습 횟수 | '미래를 고려했다' 수준으로 기존 방법과 겹침 |
| 일반 value + 동일 paired-state 데이터 | 제안과 같은 S₀/S₁ 및 factual returns | 일반적인 data augmentation만으로 충분할 수 있음 |
| 일반 value + 같은 전체 GPU 수집 예산의 추가 자연 branch | off-diagonal 수집에 든 비용만큼 보완 | 추가 supervision 예산 효과일 수 있음 |
| 제안의 β=0, α=0, 단순 evidence-removal target, 임의 보조 target | architecture·optimizer·데이터 고정 | 2×2 접근 대조의 특수한 가치가 불분명 |

동일 데이터와 동일 수집 비용 비교는 서로 다르다. off-diagonal label의 구조를 쓰지 않는 baseline에 동일 원자료를 제공하는 비교와, 그 비용으로 자연 branch를 더 모으는 비교를 모두 포함한다. 제안만 더 많은 미래 label을 받게 하고 algorithm 우월성으로 해석하면 안 된다.

Stop-RAG의 multi-action 이식은 **adaptation**으로 명명한다. 원본 STOP/CONT 실험과 이식된 text/image action 실험을 혼동하지 않는다. 가능한 외부 method는 하나의 task-compatible baseline부터 가져온다. UniversalRAG/CLIMB/MC-Search를 모두 대형 재현하는 범위는 한 GPU·3개월 조건에 과하다.

## 8. 누출·식별·평가의 must-fix

1. **분할 단위:** 질문의 모든 trajectory·branch·S₀/S₁·2×2 셀은 같은 split이다. 가능하면 source/document/entity 연결 성분 단위로 나눈다. 동일 corpus의 검색은 허용할 수 있지만, 새로운 source 일반화 평가와 구분한다. giant component 등으로 분리가 어렵다면 한계를 보고한다.
2. **선택 편향:** 성공 사례·positive interaction·gold bridge만 뽑은 학습으로 population 효과를 주장하지 않는다. natural sample의 prevalence를 먼저 보고하고, hard subset을 oversample했다면 전체 분포와 분리한다. MC-Search 자체의 chain 필터링 성격도 고려한다.
3. **일반화 단위:** 수천 state가 있어도 원 질문 50개면 독립 사례 50개다. 문항 또는 source cluster로 uncertainty를 계산하고, state 단위 bootstrap으로 좁은 CI를 만들지 않는다.
4. **branch oracle 편향:** noisy max는 위로 편향될 수 있다. regret는 observed branch regret인지 추정 기대 regret인지 명명하고, 한 번의 최고 실현 결과를 실현 가능한 최적 정책이라고 부르지 않는다.
5. **폐루프:** prefix classification/return regression이 좋아도 실제 새 정책의 상태 분포가 다르다. 독립 질문에서 정책을 처음부터 실행한다. 필요하면 train partition에서 실패 포함 정책 혼합으로 한 번만 새 상태를 수집해 모든 baseline에 같은 추가 budget을 준다.
6. **모듈 coupling:** action prefix가 query 품질·답변 형식을 바꾸면 순수 선택기 효과가 아니다. 그 변화가 있다면 시스템 효과와 선택 효과를 구분하고, 공통 forced-action replay도 일부 수행한다.
7. **멀티모달 필요성:** 원 이미지, image 제거, caption/OCR만의 비교를 통해 실제 시각 정보가 필요했는지 확인한다. metadata만으로 entity를 알 수 있는 task를 visual reasoning 성공으로 세지 않는다.
8. **근거 oracle:** gold evidence만 주는 Reader와 gold chain·중간 답까지 주는 Reader는 다른 oracle이다. 후자를 근거 recall만 해결한 upper bound로 해석하면 안 된다. 초기 진단은 gold evidence만의 능력을 우선 본다.
9. **품질 지표:** EM을 주 지표로 쓸 때 normalized answer와 format-validity를 함께 기록한다. 의미상 정답 판정은 사전 고정한 rubric으로 blind audit하고 자동 judge의 오류를 일부 검토한다. 여러 지표 중 좋아진 것만 선택하지 않는다.
10. **지연:** 평균·p50·p95, timeout·fallback을 모두 포함하고 cache 정책·실행 순서를 통제한다. 작은 pilot의 p95는 불안정하다고 명시한다. controller 연산 시간뿐 아니라 end-to-end wall time이 주 지표다.

## 9. 2주 반증 계획과 명확한 종료 판단

200–300문항은 **현상 탐색과 feasibility**를 위한 범위이지 일반화·비열등성의 충분한 표본 수가 아니다. 최종 test는 pilot과 분리하고, final test를 보기 전에 본 연구의 표본 수·허용 품질 손실·분석 방법을 정한다.

| 시점 | 해야 할 일 | 통과 조건 또는 중단 이유 |
|---|---|---|
| 1–2일 | GPU 모델/VRAM/가용 시간 확인. 30–50문항에서 natural rollout, gold-evidence-only Reader, first-token/생성형 runtime 확인 | Reader가 evidence를 받아도 못 풀거나 visual/context budget이 안 맞으면 controller 연구를 멈추고 기반 task를 바꿈 |
| 3–5일 | 사전 고정 200–300문항의 train/development pilot. 관측 boundary에서 full 실패 포함 2×2 기록, same-modality 대조 일부 | `N₀` 또는 action advantage 변화가 noise·metadata·query echo로 설명되는지 검사. 수십 개의 독립 사례조차 없으면 확대 전 중단 검토 |
| 6–8일 | cached features로 일반 value, paired-state augmentation, 제안 auxiliary objective 비교 | query-only보다 상태가 유용하지 않거나 접근 대조 target이 held-out source에서 예측되지 않으면 특수 head 중단 |
| 9–11일 | development 질문에서 closed-loop QA–latency 비교, 첫 acquisition/후속 acquisition 효과 분리 | 임의 미래 branch oracle만 좋아지면 실패. 일반 value 또는 first-token이 같은 frontier면 novelty 경로 중단 |
| 12–14일 | label GPU-hour 포함 비용 재산정, 실패 audit, 문헌 추가 감사, 진행/중단 결정 고정 | 표본 확대·두 번째 reader 검증이 남은 일정에 맞을 때만 본 연구 진행 |

다음 수치는 **권장 pilot gate 예시**이며 측정된 결과나 필수 통계 기준이 아니다. runtime smoke 뒤 본 pilot 결과를 보기 전에 확정한다.

- natural state에서 의미 있는 접근 대조·행동 preference 변화가 충분히 자주 발생해야 한다. 예컨대 200–300문항 중 독립 사례 30개 미만이면 source 일반화 학습을 주장하기 매우 어렵다. 부족할 때 positive subset만 새로 꾸려 전체 현상처럼 보이지 않는다.
- 비용 상한에서 얻을 수 있는 observed oracle 이득이 2–3 QA percentage point 이하라면 그보다 큰 방법 기여를 기대할 근거가 약하다. oracle 자체가 noisy max라면 이 수치도 낙관적이다.
- 제안이 일반 value보다 나아 보이더라도 pilot CI가 넓으면 '성공 확정' 대신 추가 표본에서 검증할 가설로 남긴다. 효과의 방향·크기·사례·비용을 함께 보고 결정한다.
- 최종 목표가 예컨대 QA 손실 ≤2pp에서 latency ≥20% 절감이라면, **2pp 비열등성을 판별할 표본 규모가 가능한지** 먼저 계산한다. 20문항이나 200문항에서 p>0.05라는 이유로 비열등성을 주장하지 않는다.

### 결과별 다음 선택

| pilot 결과 | 권고 |
|---|---|
| generic future-value가 충분하고 빠름 | 사용자 목적의 실용 controller는 완성할 가치가 있다. 새로운 ICML 알고리즘이라는 주장은 철회 |
| 접근 대조 현상은 강하지만 auxiliary 학습은 이득 없음 | 재현 가능한 진단 연구 가능성을 검토. CUE-R·MC-Search 대비 새로운 발견과 충분한 외부 검증 없이는 별도 논문 기여로 확정하지 않음 |
| 현재 evidence 이후에는 이득, 첫 acquisition에서는 이득 없음 | 'evidence-conditioned continuation'으로 제목·claim을 좁힘. 장기 계획 능력으로 포장하지 않음 |
| one-token이 이미 같은 품질·속도 frontier | head 연구를 멈추고 시스템 PoC를 정리. memory/CoT를 추가해 주제를 살리지 않음 |
| Reader/corpus가 bottleneck | reader 또는 데이터 한 번 교체 후 재진단. 둘 다 불충분하면 이 task에서 control 연구 중단 |
| visual read의 비용만 압도적 | text/OCR→image 관측 선택을 대안으로 검토하되 TAP-RAG·active acquisition과 독창성 재감사. 주력과 병행 구현하지 않음 |

## 10. GPU 한 대·약 3개월의 현실성

작은 head 학습은 가능성이 높지만 본 연구의 비용은 VLM branch 수집이다. GPU가 Blackwell이라는 아키텍처 이름만으로 VRAM·동시 모델 적재·속도를 확정할 수 없다. 기존 기록의 약 22.5GB peak는 그때의 특정 모델·이미지/context 설정에서 나온 값이다. 새로운 하드웨어 적합성의 보증이 아니다.

다음과 같이 비용을 먼저 세어야 한다.

```text
총 GPU-hours ≈
  (자연 rollout 호출 수 × 평균 호출 시간
   + branch query 호출 수 × 평균 query 시간
   + branch answer 호출 수 × 평균 answer 시간
   + 반복 평가 호출 비용) / 3600
```

예를 들어 250문항×2상태×2 tool이면 1,000개의 state-action 대조다. 각 대조의 query 2회·answer 4회로 약 6,000 VLM 호출이며, stop-answer와 원 rollout 비용은 별도다. 호출 평균이 가정상 5초면 약 8.3 GPU-hour, 20초면 약 33.3 GPU-hour다. 이는 **직렬 실행 산술 예시**이며 실제 처리량 추정이 아니다. 이미지 preprocessing, context 증가, 실패 재시도, stochastic trials, 데이터 준비가 더해진다. retrieval cache 재사용과 평가 warm/cold 정책은 명시한다.

권장 범위는 frozen reader 하나·frozen retriever 하나·주 데이터 하나에서 시작하고, 11월에 핵심 가설이 통과하면 12월에 다른 reader 또는 외부 corpus로 핵심 비교를 옮기는 것이다. 큰 MC-Search 전체 corpus를 첫날부터 구축하거나 retriever와 VLM을 공동학습하지 않는다. 작은 corpus를 쓴 경우 distractor 정책·검색 난이도·일반화 범위를 공개한다.

10월 말까지 go/no-go, 11월에 주 방법/강한 baseline 확정, 12월 전반에 외부 검증, 12월 후반에 분석·집필을 권한다. 1월은 사용자의 목표 일정이며 공식 ICML 2027 마감을 본 검토에서 새로 확인하지 않았다. 제출 가능성과 채택 가능성을 약속할 수 있는 근거는 현재 없다.

## 11. 최종적으로 남겨야 할 논문 메시지와 미검증 사항

검증을 통과한 경우에만 사용할 수 있는 메시지는 다음 정도다.

> 작은 controller가 현재 evidence의 단순 답변 관련성만으로 다음 검색을 판단하면 실패하는 조건을 계측하고, query-generation과 answer-reading의 evidence access를 분리한 학습 대조가 동등 비용의 일반 value learning보다 source 일반화와 폐루프 QA–latency를 개선함을 보였다.

현재는 이 문장의 **어느 개선 결과도 확보되지 않았다.** 새 실험을 수행하지 않았고, 제안 loss의 효능·최적 표현·실제 effect prevalence·corpus 접근성과 라이선스·single-GPU 전체 비용·외부 일반화는 미검증이다. literature search도 exhaustive하지 않다. 2×2 대조는 원리를 설명하는 정확한 실험 설계 후보이지만 새 causal theorem이 아니며, auxiliary learning은 검증할 후보 방법이다.

기존 초안의 좋은 점은 성능·독창성을 과장하지 않고 강한 generic value baseline을 요구한다는 점이다. 이번 검토에서 바꾼 핵심은 **'짧은 미래 가치를 학습한다'는 넓고 이미 알려진 방법을 주력 기여에서 내리고, 관측된 evidence가 다음 acquisition의 가치를 어떻게 바꾸는지 식별하고 학습하는 더 좁은 실험으로 옮긴 것**이다. 이 좁은 안마저 강한 baseline을 넘지 못하면 주력 알고리즘 가설을 종료하는 것이 맞다.
