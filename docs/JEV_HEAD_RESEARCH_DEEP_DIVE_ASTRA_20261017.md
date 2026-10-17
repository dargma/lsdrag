# Astra 심층 조사: 작은 head가 Native를 넘어설 수 있는 연구 조건

조사 기준일: **2026-10-17 UTC**. 원문 방법·부록을 중심으로 읽은 핵심 12편과 보조 자료를 구분했다. 검색으로 모든 선행연구의 부재를 증명할 수 없으며, 아래 차이는 독창성의 증명이 아니라 검증할 가설이다. 논문 성능은 저자 보고이며 재현하지 않았다.

**현재 상태: 사용자 HOLD 유지. 이번 작업은 문헌 조사와 제안 문서 작성뿐이다. 새 GPU 추론·학습·평가·프로토콜 실행을 하지 않았다. 기존 100 평가 질문은 후속 학습·튜닝에서 제외한다. 아래 예산과 gate는 제안이며 동결·승인·실행된 프로토콜이 아니다.**

## 판단과 추천

현재 결과는 유용한 실행 최적화의 출발점이다. 그러나 **작은 hidden-state head, frozen LLM 주변의 경량 정책, 반사실 실행 결과로 더 좋은 선택을 증류하는 원리 모두 선행연구가 있다.** 이들을 함께 쓴다는 이유만으로 강한 novelty를 주장하기 어렵다.

가장 중요한 새 확인은 두 가지다. LAnR v2의 부록에 있는 실제 control head는 affine projection+sigmoid이며 추가 LLM forward가 없다. FORGE는 frozen host 주위의 작은 router를 offline action enumeration과 품질·비용 target으로 학습한다. 따라서 지금 방향의 공헌은 구조 이름보다 **어떤 결정에 얼마나 적은 supervision을 써서, 어떤 강한 Native 기준보다 왜 더 좋은 실제 품질–비용 곡선을 얻는가**에서 나와야 한다. [LAnR §3/B.1](https://arxiv.org/html/2604.17866v2), [FORGE §3](https://arxiv.org/html/2609.34358v1)

요청한 논문별 별도 정리는 [LAnR 상세 노트](papers/LANR_2604_17866_20261017.md)와 [FORGE 상세 노트](papers/FORGE_2609_34358_20261017.md)에 저장했다.

추천 flagship은 **모든 온라인 결정을 head가 수행하면서, 오프라인 반사실 결과로 품질과 미래 비용을 개선하는 정책**이다. Native는 학습 참고와 강한 비교군이며 온라인 기본 fallback으로 호출하지 않는다. 먼저 최적화된 Native가 기존 시간 이득을 얼마나 재현하는지 검사하고, 자기 정책이 방문한 상태에서 대안을 실행해 head를 개선한다. 인자·답변은 같은 frozen Qwen이 계속 생성한다. 이는 기존 통합 head+generator의 개선이며 별도 decision-only 모델 실험이 아니다.

최신 사용자 교정: 한 번만 개입하고 이후 Native로 돌아가는 설계는 속도 이득과 주방법으로서의 매력이 약하다는 이유로 flagship에서 제외했다. 그 방식은 선택적 원인분리 진단으로만 남긴다. 이 교정은 연구 방향 수정이며 실험 재개 승인이 아니다.

단순 imitation+20% latency 절감은 제품상 충분히 가치가 있을 수 있다. 논문상으로는 최적화 Native·간단한 규칙·동일 데이터의 일반 cost-sensitive 학습보다 나아야 한다. 이 기준을 넘지 못하면 **실용적 시스템 개선은 남지만 강한 새 연구 기여는 없을 수 있다.** 수락 확률이나 최초 발명을 추정하지 않는다.

## 현재 증거가 요구하는 질문

저장 JSON과 코드에서 확인한 현재 구조는 frozen Qwen3-4B-Instruct-2507의 마지막 token hidden 하나에 독립적인 linear action/read head를 붙인 것이다. 공유 trainable trunk는 없고 CE를 합산한다. 학습은 train24의 82상태, validation8의 29상태다. 독립 질문 수는 상태 수보다 작다. [학습 코드](../scripts/run_improvement_v3.py), [feature 추출](../src/lsdrag/text_integrated.py)

새100에서 Native/head F1은 .311133/.315705, EM은 .19/.20이다. 5승·90동률·5패이고 exact 정답 3개를 잃고 4개를 얻었다. Native 형식 오류4/head0이며 오류 없는96의 사후 차이는 F1 -.005654다. 전체100이 주 분석이고 오류 없는 subset은 원인 해석의 단서일 뿐이다. 시간 1.3366→.9783초는 공유 GPU 관측값이다. 호출2.26→2.39와 입력3914→4106은 증가했다. [원 결과](../lab/reports/text_integrated_20261016/frozen_head_generalization100_20261017.json)

이로부터 다음 세 질문이 나온다.

1. 시간 이득 중 **학습 head만이 만드는 부분**은 얼마인가? 고정 구문 batch 처리와 cache 재사용은 Native에도 제공해야 한다.
2. head가 Native와 다르게 고른 상태에서, **어떤 차이는 무해하고 어떤 차이는 정답을 잃게 하는가?** 모든 action disagreement에 같은 CE 벌점을 주는 것이 적절한가?
3. 대안을 실행하면 나아지는 상태가 실제로 있고, **그 차이를 현재 관측만으로 학습할 수 있는가?** oracle 최대값은 learnability 증거가 아니다.

기존 contraction probe는 train 정답5/24, nonterminal 단축 gate 실패였다. 그 제한된 상태·후속 정책·짧은 horizon은 전체 행동 최적화 데이터가 아니다. 기존 read-selection도 품질 손실을 냈다. 따라서 중간 transition 증류나 인자 생성을 대체하는 새 decoder부터 시작할 이유가 없다. [기존 probe](../lab/reports/text_integrated_20261016/contraction_probe_20261017.json)

## 핵심 문헌 12편: 이번에 실제 읽은 범위와 방법 차이

아래 날짜는 최초 공개/읽은 버전 날짜다. `방법 확인`은 해당 절의 원문을 읽었다는 뜻이며 전체 논문 재현·증명 검증을 의미하지 않는다. 모든 접근일은 2026-10-17이다.

| 원문·버전 | 실제 읽은 범위와 핵심 방법 | 우리에게 남는 질문 |
|---|---|---|
| [LAnR](https://arxiv.org/html/2604.17866v2), 2026-04-20 / v2 05-07 | §2–3, B.1, D. `[PRED]` hidden으로 query vector와 retrieve/stop을 얻는다. 부록의 head는 단일 affine+sigmoid, 추가 forward 0. contrastive/NTP/control 공동 학습, gold evidence 포함 여부로 제어 label 구성 | same-prefill/linear head 자체는 선행. 기존 외부 툴 계약 유지와 frozen backbone의 **행동별 결과차 학습**이 실제 장점을 주는가? |
| [TabAgent](https://arxiv.org/html/2602.16429v1), 2026-02-18 / v1 | §3, 실험 설정, 비용·feature 부록. 성공 trace를 구조·텍스트 feature로 만들고 synthetic coverage를 보강해 약50M classifier로 shortlist 대체 | 생성형 유한 선택을 classifier로 바꾸는 주장도 선행. 마지막 hidden의 이점은 metadata-only classifier와 비교해야 함 |
| [Stop-RAG](https://arxiv.org/html/2510.14337v1), 2025-10-16 / v1 | §3–4, offline 구성과 식13–14/Algorithm1. full trajectory 각 prefix에서 답변을 채점하고 STOP/CONT Q(λ) 학습, 답변 반복8회. 현재 정답도와 미래 가치를 구분 | FINAL-only 개선이면 가장 직접 비교 대상. 우리의 target은 여러 tool의 **공통 후속 정책 아래 실행 결과**이며 full Q* 학습이라고 부르지 않음 |
| [Partial Answer Quality and Utility](https://arxiv.org/html/2609.16453v1), 2026-09-15 / v1 | §3–4, window regressor와 stopping 규칙. 중간 답변 품질 및 인접 iteration 품질 변화, trajectory-derived predictor | 해당 partial utility는 현재 iteration의 변화량이다. 이를 곧바로 임의 대안 행동의 미래 causal advantage와 동일시하면 안 됨 |
| [GRASP](https://arxiv.org/html/2607.10463v1), 2026-07-11 / v1 | §3, Appendix B/C. keyword/semantic/read/final, 답변·gold reading·상보검색·효율 보상, Qwen2.5-3B full-parameter GRPO | 행동 공간도 이미 가깝다. backbone 동결·작은 branch 예산·minimal thinking의 장점을 비용까지 입증해야 함. 논문 점수와 우리100을 직접 비교하지 않음 |
| [FORGE](https://arxiv.org/html/2609.34358v1), 2026-09-28 / v1 | §2–4. frozen host, 약269K router, evidence form×thinking 설정, arm enumeration→Boltzmann KL→GRPO. Full feature에는 fresh query host probe4회, Lite에는 없음 | **frozen+small router+반사실 outcome+quality-cost**도 선행. ours는 episode 내부 순차 action/same-prefill이다. 차별화 여부는 실제 효용과 sample scaling에 달림 |
| [ProAct](https://arxiv.org/html/2602.05327v1), 2026-02-05 / v1 | §3, §4.2. 환경 MCTS를 compact explicit reasoning으로 SFT, 온라인 RL과 Monte-Carlo critic. 실용 critic은 저렴한 random-policy rollout surrogate 사용 | 환경 분기를 증류해 teacher 행동을 넘는 일반 원리는 선행. 자연어 인자 생성이 필요한 우리 환경에서는 random rollout이 게임처럼 싸거나 유익하다는 보장 없음 |
| [Probing-RAG](https://arxiv.org/html/2410.13339v2), 2024-10-17 / v2 2025-02-27 및 [NAACL 2025 논문](https://aclanthology.org/2025.findings-naacl.181/) | §3, dataset/size 분석. 이미 생성한 rationale+answer의 여러 layer hidden을 mean-pool하고 correctness label로 retrieval gate 학습 | pre-answer 마지막 hidden에서 즉시 제어하는 조건과 다름. 현재 correctness를 미래 추가검색 가치로 간주하지 말아야 함 |
| [AggreVaTe / NRPI](https://arxiv.org/pdf/1406.5979), 2014-06-23 / v1 | §2–3, Algorithm1/2와 한계. learner 방문 상태에서 한 행동을 탐색한 뒤 expert/current policy로 이어가 cost-to-go classifier 학습 | proposed branch→advantage→head 원리의 직접 선행. 전문가 suffix가 학생에게 너무 낙관적일 수 있다는 한계도 이미 설명됨 |
| [SPIBB](https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf), ICML 2019 proceedings | §2, Algorithm/Theorem2와 count-based bootstrapping. 자료 부족 state-action에서는 baseline policy를 보존 | 보수적 fallback의 근거가 되지만 연속 hidden·24질문·ensemble disagreement에 유한 MDP의 보장을 자동 이식할 수 없음 |
| [Learning to Defer](https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf), ICML 2020 proceedings | §3–4, cost-sensitive surrogate와 expert-relative rejector. expert 결과도 학습 정보로 사용 | head confidence보다 **head 사용 시 손실 대 Native 비용/오류**를 학습해야 함. sequential policy용 보장으로 직접 확대하지 않음 |
| [DENSE](https://arxiv.org/html/2609.21423v5), 2026-09-18 / v5 10-08 | §2–3, REFIT reset 규칙, shortcut compilation 부록. evidence-grounded nested tree를 feedback으로 만들고 reset한 별도 attempt에서 평가 | 정책 head나 짧은 실행 sequence를 직접 학습하는 방법이 아니다. 이름의 trajectory distillation만 보고 우리 중간 transition 압축 성공 선례로 쓰면 안 됨 |

**보조 확인:** [Skill-RAG v4, 2026-08-07](https://arxiv.org/html/2604.15771v4)의 §2–3도 확인했다. correctness probe가 실패를 감지하면 prompt-based rewrite/decomposition/focusing/exit를 고른다. 단순 ‘failure hidden+recovery routing’만으로 차별화하기 어렵다. [TurnOPD v2, 2026-09-28](https://arxiv.org/html/2607.05804v2)는 turn-aware on-policy distillation의 학습 효율 연구다. 해당 방법·실험 절을 확인했으며 우리 inference primitive-call 압축과 구분한다.

[OPID v1, 06-25](https://arxiv.org/abs/2606.26790v1), [SOD v3, 08-18](https://arxiv.org/abs/2605.07725v3), [PANDO v2, 05-26](https://arxiv.org/abs/2605.24785v2)는 이번에는 초록·버전 메타데이터를 screening했다. 각각 hindsight skill supervision, step divergence에 따른 distillation 조절, online skill library/cache 효율을 다룬다. 원문 전체 방법을 검토했다고 주장하지 않으며 현재 flagship의 직접 baseline으로 억지로 넣지 않는다. 기존 AnySearch/MetaRAG/Better Together 조사 내용은 [이전 문서](JEV_RELATED_RESEARCH_UPDATE_20261017.md)에 보존하고 이번 확인과 구분한다.

## 먼저 할 gate 0: 학습 없는 빠른 Native가 시간 이득을 재현하는가

새 실험을 재개한다면 **branch 라벨 수집보다 앞서 이 검사를 추천한다.** 원래 Native의 무제약 full payload 생성은 가장 강한 효율 기준이 아니다. JSON 안정화·고정구문 일괄처리·동일 KV 재사용을 양쪽에 제공해야 한다. [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)와 [XGrammar](https://arxiv.org/abs/2411.15100)는 강한 실행 baseline의 근거다. 문법 masking이 자동으로 모든 고정토큰의 model forward를 생략한다는 뜻은 아니다. skip/fast-forward는 해당 구현의 실제 지원과 경계를 별도로 확인한다.

제안하는 `N-fast`는 다음 순서다. 이는 아직 구현·검증되지 않았다.

1. 기존 prompt의 동일 prefill에서 LM head logits를 읽어 Native serialization의 tool 시작/FINAL 시작 분기를 고른다. prefix를 넣기 전에 FINAL 선택 가능성을 유지한다.
2. tool 분기라면 공통 구문을 batch 처리하고, 합법 tool 이름이 처음 갈리는 token 위치에서 Native logits로 선택한다. 하나의 first-token score로 이름을 식별할 수 없는 경우 trie 분기를 더 진행한다.
3. 선택이 결정되면 남은 고정 구문을 batch 처리하고 같은 cache에서 인자를 생성한다. FINAL도 동일 schema·예산으로 생성한다.
4. mode 선택·subword·공백·다른 합법 serialization·cache position을 감사한다. constrained greedy는 unconstrained Native와 정책이 달라질 수 있으므로 `Native와 정확히 동일`은 검증할 가설이다. forced common prefix가 잘못된 경로를 강요한다면 더 보수적인 native-name boundary decoding을 함께 둔다.

비교군은 원 Native(N-full), N-fast, 현재 BC head+같은 executor(H-BC)다. 기존 probe의 Native 경로는 cache 복제 후 선택구문을 다시 처리했으므로 최적 N-fast의 latency 기준으로 쓰지 않는다. Native를 끝까지 생성한 뒤 payload를 버리고 인자를 재생성하는 비용도 무료로 취급하지 않는다.

**가장 싼 제안 규모:** 기존 train8에서 이미 고정한24상태를 그대로 사용한다. 각 상태에서 N-full/N-fast/H-BC 고정 상태 생성을 정책마다1회, 이어 같은 train8의 세 정책 rollout1회(24episode). 총 state-generation72회+24episode, load 제외20분/생성80k/입력prefix1.5m의 hard cap. cap 도달은 미완료로 기록하고 쉬운 질문만 남겨 통과시키지 않는다. 이것은 새 test가 아니라 구현·투자 판단용 train 진단이다. 이 최소규모로 안정적인 시간 판정이 안 되면 불확실로 남기며 통과/실패를 강제하지 않는다.

기능 gate는 cache reuse/argument 의미/handler 결과/FINAL 유효성을 확인하며, 불일치 하나도 조용히 제외하지 않는다. Native 동일성은 정상 Native 출력에 대해 검사하고 오류 수정은 별도로 센다. timing은 warmup 후 조건순서를 교대하고 같은 고정 상태·동일 출력 경로의 비용도 분리한다. 공유 GPU에서 안정적 통제 측정이 안 되면 비용 forward 수까지만 판단한다.

**효율 kill 기준 제안:** N-fast가 품질을 유지하면서 원 N-full→H-BC 시간 이득의80% 이상을 재현하고 H-BC 잔여 절감이5% 미만이면, ‘학습 head가 속도 이득의 핵심’이라는 방향은 종료한다. 80%/5%는 투자 기준이며 유의성 기준이 아니다. 이것은 반사실 정책 개선 가설 자체를 죽이는 결과는 아니다. 그 경우 후속 학습은 **추가 선택 품질 개선 가능성**이 있어야 정당화되며, 20% 속도 headline을 미리 약속하지 않는다.

## 아이디어 1 — 추천: 모든 결정을 head로 유지하는 반사실 정책 개선

**반증 가능한 가설 H1:** frozen backbone의 same-prefill hidden으로 매 turn 행동을 고르는 작은 head가, Native action imitation 대신 **자신이 방문한 상태에서 수집한 행동별 미래 품질·비용**을 학습하면, N-fast와 같은 데이터 예산의 BC 보강보다 나은 품질–비용 곡선을 만들 수 있다. 온라인 제어는 처음부터 끝까지 head이며 Native 선택기를 매번 호출하지 않는다.

Native가 QA 절대상한은 아니지만, label을 Native 행동 하나로만 만들면 체계적인 초과 성능을 학습할 신호가 약하다. 반대로 같은 상태에서 read와 search를 실행한 뒤 끝까지 진행해 얻은 실제 결과차는 Native가 선택하지 않은 행동의 장점을 알려줄 수 있다. 현재 90/100 QA 동률과 소수의 치명적 차이는 모든 action disagreement에 같은 벌점을 주는 CE와 downstream 목적의 불일치를 시사한다. 이100의 개별 실패·정답은 후속 학습에 재사용하지 않는다.

### 입력·출력·label·학습

- 입력 `φ(s)`는 같은 prefill의 normalize한 마지막 hidden, 공개 scalar5개(turn, 남은 budget, 마지막 결과수, 직전 empty 여부, 관측 ID 수), 마지막 tool one-hot과 legal mask다. feature 구성은 구현 전 manifest로 고정한다. gold answer/support, 미래 tool 결과, branch 답변은 입력에서 제외한다.
- 출력은 기존4 action이다. action 이후의 인자·FINAL 답변은 같은 frozen Qwen과 cache에서 생성한다. 첫 모델에서는 read 인자 선택을 켜지 않는다. Native online fallback도 기본 정책에 넣지 않는다.
- `π_k`는 iteration k의 **동결된 현재 head 정책+고정 generator**다. 첫 iteration의 π0는 기존 BC head다. πk로 train 질문을 실제 rollout해 상태를 모으고, 같은 상태에서 합법 action을 바꾼 뒤 **πk로 suffix를 완료**한다. label은 `Q^{πk,g,H}(s,a)`이며 전역 Q*가 아니다. H는 원 전체8 tool budget에서 남은 양이다. 짧은 H=2를 사용하면 별도 진단이고 전체 정책값과 혼합하지 않는다.
- 품질 target은 action별 F1, EM, support/error를 분리 기록한다. 비용 target도 remaining tool calls, full-prefill 입력, decoder forwards, prefix batch, wall time을 분리한다. primary advantage는 `ΔF1(s,a)=F1(a→πk)-F1(πk(s)→πk)`다. Native 행동은 참조 비교로만 추가하며 label의 정답으로 고정하지 않는다.
- 최소 학습은 동일 feature의 작은 linear quality/cost regressor와 action head다. 질문별 가중치가 같도록 **평균 MSE+L2(초기β=1)**로 action별 ΔF1와 normalized remaining decoder/input cost를 맞춘다. action policy는 예측 quality advantage를 우선하고, 사전고정 품질 tolerance 안의 행동에서 비용이 적은 것을 고른다. 또는 같은 자료의 cost-sensitive CE로 action logits에 바로 증류한다. 둘을 큰 새 구조로 합치지 않고, 처음에는 linear regression 대 hard best-action CE의 간단한 비교로 충분하다.
- 데이터가 없는 action을 큰 predicted value만으로 선택하지 않는다. legal mask와 train coverage 조건을 적용하고, unsupported state/action에서는 **기존 BC head**를 유지한다. 이것은 Native fallback이 아니므로 Native selector decode를 추가하지 않는다. BC anchor의 장점/한계도 별도 ablation으로 확인한다. 품질 tolerance·cost weight를 test에서 고르지 않는다.

최초 비용 목적은 F1/EM을 희생해 FINAL만 고르는 보상이어서는 안 된다. train label에서 비용 우대는 F1·EM이 유지되고 관측 근거가 악화하지 않는 후보에만 부여하는 보수적인 teacher target을 쓴다. 실제 새 정책에서도 premature FINAL, zero-tool wrong answers, exact 손실, support 저하를 모두 보고한다. gold gating은 훈련 label 생성 규칙이며 inference에서 gold를 조회하는 규칙이 아니다. 품질·비용 예측이 틀리면 여전히 collapse할 수 있으므로 gate가 필요하다.

### 정책이 바뀌는 문제를 피하지 않고 측정한다

πk의 suffix로 얻은 target이 πk+1의 가치를 뜻하지 않는다. 따라서 고정 상태 ranking이 좋아졌다는 이유로 전체 개선을 선언하지 않는다. πk+1을 **episode 처음부터 끝까지 실행**해 품질·비용과 새로운 방문 상태를 확인해야 한다. 통과하면 다음 iteration에서 πk+1을 새로 동결하고 그 상태·suffix로 branch를 수집한다. 각 label에 policy/checkpoint/generator/horizon hash를 붙인다. old/new labels를 같은 Q target처럼 섞지 않으며 old data는 BC anchor나 별도 정책조건부 자료로만 사용한다.

이 원리는 AggreVaTe/NRPI와 approximate policy iteration의 계열이다. 새로운 RL 원리라고 주장하지 않는다. 차별화 가능성은 **Native-free 온라인 실행을 유지하는 작고 싼 정책을, 제한된 환경 분기 예산으로 실제로 개선할 수 있는가**에 있다. 마지막 hidden 하나가 Native의 전체 KV+추가 decode 계산과 동등하지 않으므로 표현력 한계를 인정한다. 학습이 안 되면 큰 MLP나 더 많은 RL이 자동 해답은 아니다.

기존 contraction도 head suffix였지만 짧은 horizon·상태·후보 구성의 제한이 있으므로 새 full-horizon target과 바로 합치지 않는다. Native suffix로 바꿔 재명명하는 것도 금지한다. 한 번 개입 후 Native/πk로 복귀하는 실험은 target 차이를 분리하는 **선택적 진단**이다. 그것을 주방법의 성능으로 보고하거나 현재 all-head26.8% 관측 시간 감소를 유지한다고 약속하지 않는다.

### 가장 작은 판별 cycle — 제안, 실행 아님

Gate0 뒤에 **기존 train24**만으로 한 cycle을 제안한다. 기존 BC π0로24 rollout을 얻고 질문당 첫 tool 뒤 상태1개를 outcome-blind로 선택한다. 합법4 action을 각각 실행하고 공통 π0 suffix로 끝내므로 최대96 branch suffix다. 첫 cycle에서는 active-state mining을 추가하지 않는다. train4의 reset/cache 기능검사 비용도 이 cap에 포함한다.

atomic-question family 연결성분을 기준으로4fold cross-fit한다. 부모의 CPU audit상 train24의 atomic48개가 모두 달라24 singleton이므로18학습/6audit가 가능하다. 각 fold에서18질문의 branch로 학습한 candidate를 나머지6질문의 **전체 autonomous episode**에서 평가한다. 기존 BC가 원train24로 학습된 상태이므로 이는 ‘처음 보는 질문의 총성능’ 검증이 아니라 **추가 counterfactual 학습의 조건부 일반화 진단**이다. fresh 평가라고 부르지 않는다.

최대 비용은 기존-head roll-in24+branch96+out-of-fold4개 정책(primary, hard-label CE, metadata-only, BC coverage 보강) 각24 rollout+N-fast comparator24=240 episode/suffix 작업이다. 기존BC의roll-in을동일계약baseline으로재사용한다. Native imitation 보강 control에 필요한 동일 상태 selector 최대24회도 별도 기록하고 같은 token/wall cap 안에서 지불한다. load 제외30분/생성120k/입력prefix2m, context16384/output512/전체8 tool+FINAL을 제안한다. 초과·truncation·오류는 그대로 저장하고 미완료를 통과로 처리하지 않는다. 추가 cycle은 자동 시작하지 않는다.

**기능 gate:** 독립 branch reset/hash, legal action, 동일 generator·cache 계약, 미래정보 배제. 이 실패를 품질 실패와 섞지 않는다.

**학습 투자 gate:** (a) 기존π0 대비 F1≥.25인 합법 대안이 서로 다른 train 질문 최소4개, 그 중 nonFINAL 최소2개. 전부 FINAL이면 stopping 문제로 좁힌다. (b) OOF **전체 rollout**에서 BC 대비 F1 총합 개선>0, exact 신규획득−손실≥2, 최소2fold 양의 개선, exact 손실≤1. (c) 비용은 N-fast 또는 BC 대비 늘어날 수 있으므로 품질과 함께 Pareto를 보고, 원래 all-head를 썼다는 이유로 같은 speedup을 가정하지 않는다. (d) metadata/rule·같은 branch hard-label CE·같은 비용의 Native-label BC 보강이 동일하면 특별한 head/target 공헌을 주장하지 않는다. 숫자는 투자용 파일럿 기준이며 통계적 성공 기준이 아니다.

1cycle 통과 후에만 최대1회 추가 iteration을 검토한다. 그때는 π1이 방문한 상태를 재수집하고, **같은 branch budget**에서 uniform 상태선택 대 공개 empty/repetition/head-margin 기반 선택을 비교할 수 있다. 적어도 절반은 uniform으로 남겨 쉬운/불확실 상태에만 편향되는 것을 막고 sampling rule/확률을 기록한다. 새 데이터 배분법이 uniform branch나 단순 BC coverage 보강보다 효율적인지가 하나의 구체적 연구 가설이다. 하지만 아직 그런 배분법을 새 알고리즘으로 발명·검증한 것은 아니다.

### 강한 기여가 되려면

FORGE는 frozen host의 사전 routing과 outcome/cost target을, LAnR는 same-hidden 제어를, AggreVaTe/NRPI는 cost-to-go 정책개선을 이미 제시했다. 우리의 조합도 점진적일 수 있다. 남는 후보 기여는 **같은 prefilling을 공유하는 all-head 실행에서, 소량의 자기방문상태 반사실 supervision이 왜 Native 모사보다 효과적인지, 어떤 state에 branch 예산을 써야 이득이 나는지, 최적화 Native보다 실제 Pareto가 나아지는지**다.

oracle gap이 없거나, advantage가 family 밖에서 학습되지 않거나, N-fast/규칙/동일데이터 일반학습과 동률이거나, 개선 때문에 interaction이 길어져 비용이 더 커지면 주장의 힘이 사라진다. 이런 경우 추가 head·복잡한 RL을 계속 붙이지 않고 **강한 논문 방향은 중단하고 시스템 최적화 결과를 보존**한다. 성공해도 첫 claim은 해당 frozen 모델·도구·분포에서 작은 정책의 비용 포함 개선이며, 전역 최적·보편적 안전·모든 agent로의 일반화를 주장하지 않는다.

## 아이디어 2 — 데이터 효율 가설: 어떤 상태에 반사실 예산을 쓸 것인가

**가설 H2:** 같은 수의 실제 branch 실행을 쓰더라도, action 모방 자료를 무작정 늘리거나 상태를 균등하게 분기하는 것보다 **실패복구·선택경계 상태를 함께 커버하는 수집**이 all-head 정책을 더 잘 개선한다. H1과 다른 대형 모델을 추가하는 방향이 아니라, H1이 처음 성공했을 때 조사할 데이터 효율의 가설이다. 온라인은 여전히 모두 head이며 Native 기본 fallback은 없다.

왜 필요한가: 현재 독립 train 질문은24개이고 단순 state 수는82개다. 비슷한 쉬운 상태를 여러 번 수집하면 label 수만 늘고 정답을 잃는 선택경계는 그대로일 수 있다. 반대로 실패 상태만 고르면 편향된 정책을 배울 수 있다. 현재100의 실패 질문은 선택 규칙에도 사용하지 않으며, 실제 배포 관측에서 계산 가능한 empty/repetition/BC margin만을 후보 신호로 쓴다.

입력은 πk rollout에서 얻은 공개 상태와 head의 action margin이다. 출력은 **다음 오프라인 수집에 쓸 state/action 목록**이고 inference tool 선택은 기존 head가 한다. 수집한 label·학습 loss는 H1의 downstream F1/EM/cost 또는 같은 label의 cost-sensitive CE다. 효과의 target은 선택된 상태 자체의 oracle 최대값이 아니라, 같은 예산으로 학습한 head의 별도 질문 전체 rollout 품질–비용이다.

처음에는 작은 자료에 또 acquisition neural head를 학습하지 않는다. 사전고정 점수 규칙과 uniform을 비교한다. 예를 들어 budget의 절반은 질문당 균등 상태를, 나머지는 empty-result/repetition/낮은action margin 상태를 outcome-blind로 선택한다. 질문당 state 상한2를 두고, 같은 질문의 상태가 train/audit 양쪽으로 퍼지지 않게 한다. 미래 정답을 보고 효과 큰 branch만 선택하면 실전 수집정책이 아니다.

**최소 추가 판별 제안:** H1 cycle 통과 뒤, 같은 train24의 현재π1 전체 trajectory에서 두 방식이 각각24state를 선정하고 각state 합법4action을 실행한다. 최대192suffix(중복state/action은 동일hash일 때 재사용), 두 수집방식 각각 label예산96을 맞춘다. load 제외30분/생성160k/입력prefix3m 상한이며 rollout/selector overhead도 포함한다. 추가 실험이므로 H1의96branch 안에 무료로 포함했다고 세지 않는다. 후보의 부족·중복으로 실제 예산이 달라지면 사용량과 결과를 함께 보고한다.

비교군은 uniform branch, 위 혼합 선택, 같은 총 수집비용의 Native-label BC coverage 보강이다. 학습 capacity/step/seed를 맞추고 family cross-fit의 전체 episode를 비교한다. **계속할 조건 제안:** 혼합 선택이 uniform 대비 exact net gain≥2 또는 F1≥.03을 얻으면서 비용이 악화하지 않고, 최소2fold에서 개선되며 BC보강도 이겨야 한다. 수치는 투자 기준이다. 추가 확인평가 전 독립 성능 개선이라고 주장하지 않는다. 개선이 한 질문에만 의존하거나 uniform과 같으면 수집법의 특별한 공헌은 접는다.

차별화 후보는 '반사실을 쓴다'가 아니라 **작은 분기 예산을 어떤 상태에 배분해야 작은 all-head 정책의 후회비용을 줄이는가**다. active learning·AggreVaTe의 exploration 논의와 겹치므로 이 역시 원리 자체의 최초성은 없다. 단순 empty/margin 규칙이 충분하면 복잡한 acquisition learner를 만들 필요가 없다. 여러 예산과 task에서 재현되는 데이터 효율이 없으면 H1의 구현 요령에 머문다.

Native-relative deferral는 실용 비교로는 남길 수 있지만 사용자 교정에 따라 주방법과 추천3방향에서 제외한다. 기존 learning-to-defer/SPIBB의 위험 제어 아이디어가 곧24질문에서의 안전 보장은 아니다. 독립 Bernoulli 가정에서 손실0/8의 one-sided95% 손실확률 상한도 약31%,0/24도약12%이며 같은 질문의 여러 상태는 독립표본이 아니다. 현재 작은자료에서 ensemble/LCB를 사용하더라도 휴리스틱으로만 해석한다.

## 아이디어 3 — 후순위: read 후보의 ‘생성 생략 가치’를 학습

**가설 H3:** 정적 slot 번호의 teacher exact-match 대신 현재 관측 후보 내용과 downstream 결과차를 학습하면, 유용한 read selection에서만 인자 생성을 생략할 수 있다. 현재 ON/OFF 실패는 이 가설을 입증하지도 완전히 반증하지도 않지만, action 개선보다 우선할 근거는 약하다.

action을 `read_chunk`로 고정한 상태만 사용한다. 같은 prefill에서 **이미 관측한 candidate snippet/body span**의 hidden을 pool한 표현과 마지막 hidden을 작은 공유 scorer에 넣어 실제 후보 ID 또는 GENERATE를 고른다. ID는 pointer 주소이며 의미 표현은 연결된 관측 텍스트에서 얻는다. causal LLM에서 ID token이 snippet보다 앞에 있으면 그 ID hidden은 뒤의 내용을 보지 못하므로 ID token만으로 후보 내용을 읽었다고 가정하지 않는다. 검색 결과에 없는 원문 body는 feature로 추가하지 않는다. candidate 순서를 섞어도 동일 ID 선택이 유지되는지 별도로 검사하며 추가 LLM 후보분석 호출은 없다. 과거 slot4 train 양성0 문제를 단순 slot-label CE로 반복하지 않는다. token span을 저장하지 않은 과거 feature는 이 방법의 입력으로 바로 쓸 수 없다.

각 관측 후보를 read한 뒤 공통 N-fast suffix, 그리고 Qwen이 인자를 생성해 read한 뒤 같은 suffix를 실행한다. label은 후보별 F1/EM/support/cost이며 `GENERATE`와의 paired regret이다. loss는 candidate-pair outcome 차이를 가중한 ranking logistic+L2, 후보가 모두 동률이면 GENERATE에 대한 비용차만 별도로 분석한다. generation을 무조건 정답으로 label하지 않는다. 생성으로 고른 ID도 틀릴 수 있어 downstream 기준으로 더 나은 후보를 배울 원리가 있다. 반면 관측에 없는 미래 ID를 candidate에 추가하면 누출이다.

가장 싼 제안은 기존 train24 안에서 outcome-blind로 정한 read 가능 상태를 질문당1개, 최대12개만 사용하는 것이다. 관측 후보≤4+GENERATE, 최대60 suffix/15분/생성50k/입력1m. train 질문이12개를 못 채우면 그대로 coverage 부족으로 기록하고 새 질문을 자동 추가하지 않는다. top-rank unread/직전 hit/Native generated ID/기존 slot head와 같은 상태에서 비교한다. 모든 후보를 매번 실행하는 oracle는 deploy baseline이 아니다.

**계속할 조건 제안:** 품질·EM·support를 유지하면서 인자 decode를 줄이는 candidate가 서로 다른 질문 최소4개에서 있고, family cross-fit에서 GENERATE 대비 exact 손실0 및 최소4회 생성 생략, 전체 episode 비용≥5% 절감이 있어야 한다. 숫자는 투자 기준이다. read가 드물거나 ID가 짧아 전체 비용효과가 작으면 종료한다. 현재20에서8회 선택으로 평균 생성3.55토큰만 줄었던 현실을 먼저 고려한다.

차별화 후보는 ‘현재 argument 생성 비용 대 이후 interaction 결과’의 제한적 비교다. observed pointer+fallback 자체는 알려진 구성이고 FORGE/Stop-RAG의 utility routing 원리와도 겹친다. 다른 툴 계약으로 전이하지 못하거나 단순 top-rank 규칙이 같은 곡선을 만들면 독립 기여는 약하다. 검색 query를 새 decoder로 생성하는 방향으로 자동 확대하지 않는다.

## 분기 데이터·오프폴리시·평가 원칙

같은 reset 상태에서 모든 합법 action을 실제 실행하는 것은 그 상태의 후보 효과를 비교하기 좋다. 그러나 결과는 질문별 corpus·현재 관측·greedy argument generator·후속 정책에 조건부다. 마지막 hidden은 관측의 압축 표현이며 Markov sufficient state라는 보장이 없다. deterministic greedy 반복은 같은 출력의 반복일 뿐, 독립적인 uncertainty 샘플을 만들어 주지 않는다.

기존 logged trajectory에서 선택되지 않은 action의 값을 추측해 label로 채우지 않는다. Native 행동은 거의 deterministic하므로 support가 없는 action의 IPS/DR 추정은 식별되지 않는다. 실제 분기로 local action coverage를 얻어도 새로운 head가 방문할 state coverage 문제는 남는다. outcome을 본 뒤 ‘유망한 상태’만 고르면 selection bias가 생긴다. 첫 tool 뒤처럼 outcome-blind 상태 규칙, 질문/family split, 선택확률·탈락 사유를 남긴다.

훈련용 gold는 branch reward 계산에만 사용한다. teacher rationale의 미래 정답·공통 최종 evidence를 input prefix로 이식하지 않는다. branch마다 cache/history/tool 상태를 독립 복제하고 reset hash를 확인한다. invalid action·empty search·truncation도 outcome이며 사후 삭제하지 않는다. 대안별 maximum은 노이즈와 다중 비교로 낙관적이므로 oracle uplift와 cross-fit/실제 rollout uplift를 함께 제시한다.

기존 validation8은 이미 여러 번 살핀 개발자료라 새로운 독립 확인평가로 부르지 않는다. 부모의 이번 CPU audit에서 validation atomic16개 중1개가 train atomic과 공유됨도 확인했다. 첫 small gate 통과 후에만 별도의 미노출 질문 집합을 사전 선택하고 질문/family/manifest/질적 기준을 동결한다. 새100은 계속 학습 제외다. 확인 규모는 최소 관심 효과·paired disagreement·분산에 기반해 정한다. 또100을 쓰면 충분하다는 가정은 하지 않는다. 현재100의 F1 차이 구간이 약±5%p이므로 0.5%p 정도의 개선을 확정하기에는 거칠다.

## 강한 baseline과 원인 분리

| 비교 | 알아낼 것 | 공정성 조건 |
|---|---|---|
| N-full vs N-fast | schema/cache/serialization 최적화 | 동일 모델·prompt·tool·budget, 오류와 출력 변화를 별도 기록 |
| N-fast vs BC head + 동일 executor | 학습 selector의 순수 비용·품질 | Native selector 비용 포함, forced prefix 수·경계 기록 |
| BC vs hard oracle-label CE vs regret 학습 | downstream target이 필요한가 | 같은 branch label·capacity·학습/튜닝 예산 |
| hidden vs metadata-only vs 고정 규칙 | head가 문맥 의미를 쓰는가 | 규칙은 train에서만 고정, 실패 사례별 예외 추가 금지 |
| FINAL-only vs 4-action 개선 | early-stop 외 tool 선택 가치가 있는가 | 같은 후보 예산·suffix·평가 qids |
| no-defer vs confidence-defer vs regret-defer | fallback이 제공하는 가치 | Native 실행·gate feature 비용 포함 |
| learned policy vs 같은 추론비용의 Native 반복/추가검색 | compute를 더 주면 같은 개선인가 | 예산 안에서만, 여러 답변의 선택기 비용까지 포함 |

global budget을 맞추되 tool call 수를 무조건 같게 강제하면 정책의 cost 이득을 없앨 수 있다. 따라서 (1) 같은 상한에서 품질·실제 비용, (2) cost-matched Pareto, (3) 고정 상태·고정 출력의 execution microbenchmark를 구분한다. 더 많은 branch label을 쓴 head와 data 없는 Native의 온라인 비용을 비교하더라도 offline 총 수집시간·GPU시간·에너지 대리지표·break-even 사용량을 보고해야 한다.

`T = prefill + selector + prefix_batch + argument_decode + answer_decode + tools + scheduling`으로 분해한다. 절감률은 이 합에 대한 값이며 token 총합만으로 FLOPs나 latency를 대체하지 않는다. 현재 긴 답변/긴 context/더 비싼 tool이 생기면 fixed syntax의 비중이 줄 수 있다. 반대로 interaction 하나를 줄이면 다음 prefill·tool·decode까지 절약할 수 있지만 아직 그 성공을 보이지 못했다.

실제 serving에서는 batch1뿐 아니라 concurrency4/16, 짧은/긴 context, p50/p95, throughput, GPU memory, scheduler overhead를 측정할 후보가 있다. 이는 향후 검증 범위이며 지금 실행하지 않는다. vLLM·prefix cache·지원되는 speculative decoding을 양쪽에 똑같이 적용한다. [공식 speculative decoding 문서](https://docs.vllm.ai/en/latest/features/speculative_decoding/)도 효과가 workload/model/hardware에 의존함을 명시한다. custom head가 batching을 깨뜨리거나 host-device sync를 늘리면 batch1 이득이 사라질 수 있다. 지원되지 않는 조합은 실제 한계로 기록하고 가상의 최고속도 Native와 비교하지 않는다.

## 일반화와 중간 step distillation의 위치

현재 장기 성공 trajectory와 반복 가능한 nonterminal shortcut 자료가 부족하다. 2-hop 평균2~3 tool에서 ‘여러 reasoning step을 하나로 증류’하는 주장을 먼저 세우지 않는다. DENSE의 retry feedback, PANDO의 skill library, TurnOPD/OPID/SOD의 on-policy token 학습은 서로 다른 대상이다. primitive calls, LLM turns, generated tokens, retry attempts 중 무엇을 줄이는지 먼저 정해야 한다.

전이 후보는 순서대로 (a) frozen head의 MuSiQue3-hop horizon shift, (b) 같은3tool을 유지하는 HotpotQA/2Wiki의 question/corpus shift, (c) 별도의2–4B non-thinking checkpoint에서 **방법을 재학습해** 재현하는 model shift다. 다른 hidden dimension의 모델에 같은 head weight를 그대로 옮기는 것은 기본 가정이 아니다. task와 model을 한 번에 바꾸지 않고 어느 축에서 실패하는지 구분한다. 더 긴 tool 환경은 nonterminal shortcut이 실제 발견된 뒤의 후보이며 이번에는 추가 실행 범위가 아니다.

향후 성공 기준은 ‘논문 하나가 요구하는 고정 점수’가 아니다. 먼저 기능이 맞는지, 그 다음 독립질문에서 품질–비용이 나아지는지, 마지막으로 단순 방법과 선행기법으로 설명되지 않는 재현 가능한 기여가 있는지를 별도로 판단한다. gate0에서 대부분의 효과가 N-fast로 설명되고 counterfactual target도 규칙/CE보다 낫지 않으면 **그 결과를 정직하게 시스템 최적화로 정리하고 head novelty 주장을 접는 것**이 합리적이다.

## 저장·추적

- 기계판독 ledger/추천/held 상태: `lab/reports/text_integrated_20261016/astra_head_deep_research_20261017.json`.
- 실제 확인한 로컬 근거: frozen100 JSON, contraction JSON, 기존 보고서·Astra 방향 리뷰·관련연구 업데이트, `scripts/run_improvement_v3.py`, `src/lsdrag/text_integrated.py`.
- 이 문서 작성은 runtime 코드·기존 결과·checkpoint·split을 바꾸지 않는다. 제안에는 아직 qid manifest·코드·worker가 없다.
- Git·Drive 저장과 memory 갱신은 부모 작업이 수행한다. 자동 실험 시작은 계속 꺼져 있다.
