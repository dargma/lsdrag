# Jev와 공개 decision model: 원문·코드·벤치마크 조사

조사일2026-10-24. 이번 작업은 공개 원문/모델 카드/배포 코드/벤치마크 원본의 검증이다. 외부 모델을 우리 RAG 환경에서 새로 실행한 결과가 아니다. 모델 카드 수치, 커뮤니티 평가, 우리 실측을 구분한다. 다운로드한 공개 소스는 local_fast/source/decision_model_research_20261024, revision/SHA는 lab/reports/decision_model_research_20261024/sources.json에 저장했다. 외부 모델 weight는 이번 조사에서 다운로드하지 않았다.

## 핵심 결론

공개 typed decision model은 입력 상태와 선택지를 받아 빠르게 판단하는 모델이다. 이를 long-horizon 정책과 동의어로 쓰면 안 된다. Typed model도 정당한 decision model이며, 우리가 추구하는 것은 그중에서도 **reader와 계산을 공유하면서 미래 답변 품질과 비용을 최적화하는 task policy**다.

작은 head 추가 자체, 출력 토큰 없이 선택하는 구조, calibration 자체는 독창성의 충분한 근거가 아니다. 논문 주장은 공유 reader의 실제 연산 절감, 현재 정책에서 연속된 행동의 결과를 학습하는 방법, 독립 평가에서 품질/비용 동시 개선에 두어야 한다. 아직 우리 결과는 이를 입증하지 못했다.

## 현재 공개 벤치마크를 같은 버전으로 비교

Hugging Face Space multimodalart/jev-decision-index revision `0b5ecaf2396891e76c13a19aa17b874b59abcc37`의 data/index.json 직접 파싱. 생성시각2026-10-22T20:10:35Z, label Decision Index0.4.1,229개 행. Full은 private same-skill60% + private new-domain40%를 public scale로 변환한 값이다. 이전0.3.2의20/50/30배합과 혼용하지 않는다. source 내부 panel_id/v03라는 필드명이 남아 있어도 edition/공식formula를 우선한다. methodology의 옛 generated_utc와 최신edition 날짜가 다른 점도 원본 그대로 보존했다.

| 모델 | Full Index0.4.1 | Public skill | 우리 관점 |
|---|---:|---:|---|
| Perplexity Decider v1.2,27B명칭 |67.98|67.74|대형 외부 참고, 당장 주 비교에는 과도한 크기|
| D3 v3.1.0,27B명칭 |67.15|65.45|대형 외부 참고|
| Jev1.14 |60.24|60.70|hosted API 기준, 내부 학습 공개 부족|
| Perplexity Decider Flash v1.2,4B명칭 |54.29|60.06|동급 외부 판단기 우선 후보|
| D3-mini,4B명칭 |49.97|54.90|두 번째 외부 후보|
| D3-nano,2B명칭 |34.30|42.22|크기2–4B를 엄격히 지킬 때 대안|
| Liquid d1-3B |31.12|48.99|다른 backbone 참고; 이점은 직접 측정 필요|
| Strands Decider2B |20.72|25.41|학습 레시피 참고; 우선 품질 대조는 후순위|

이 값은 F1도 percentage accuracy도 아니며 RAG 목표와 직접 비교 불가하다. 신뢰구간/동률 그룹이 있으므로 점수 차이를 통계적으로 확정된 우열로 읽지 않는다. private 구성은 우리가 재현한 것이 아니다. Public과 Full 사이 차이는 분포 전이 경고이지 오염의 증거가 아니다.

원본: https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/0b5ecaf2396891e76c13a19aa17b874b59abcc37/data/index.json ; 추출본 lab/reports/decision_model_research_20261024/benchmark_snapshot.json.

## 구조와 학습: 가져올 것과 가져오면 안 되는 것

| 대상 | 검증한 공개 근거 | 우리에게 주는 의미 |
|---|---|---|
| Jev | state+typed questions→structured answers/probabilities. 제조사는 RLCD를 소개하지만 이번 원문에서 재현 가능한 전체 학습법은 확인하지 못함 | RL이라는 이름만으로 multi-turn return 최적화를 추정하지 않음 |
| D3 | nano/mini 코드·config에 code readout과 타입별 확률출력. nano의255-code readout, Qwen3.5-2B base 확인 | 고정4action head보다 범용 선택지를 읽는 능력이 있음. 우리 Qwen3-4B에 weight를 그대로 이식할 수 없음 |
| Perplexity Decider | v1.1은 별도readout와 full-attention층의 noncausal mode. v1.2는 question marker별 readout. Flash는27B teacher 확률분포 증류와 weight averaging | soft distribution 학습 참고. noncausal backbone 경로를 reader의 causal생성/cache 경로와 동일시하면 안 됨 |
| Kev | LoRA+pointer head; 작은모델은 supervised CE, 선택지섞기, calibration. 27B는 full finetune | 공개 typed model 성과가 head-only frozenLLM 충분성의 증거는 아님. backbone adaptation을 구분해야 함 |
| Liquid d1 | single-forward 결정; 공개 학습 설명은 backbone finetuning/merging, 데이터shortcut 수정·선택지순서변경 | 빠른 readout의 실용성 참고. 제시된장비/입력의 latency를 우리 환경으로 이전하지 않음 |
| Strands | Qwen3.5-2B LoRA+readout, checked paraphrases 및 gold일치 teacher증류, calibration | 학습 데이터·정답목표의 품질이 중요. 카드의 MuSiQue 판단 정확도는 우리 open-corpus QA F1과 다른 task |
| SII-Decision (초기 문헌명 InnerJev) | reasoning 후 답 분포를 reasoning없는 첫토큰 readout으로 self-distillation, fullrank/LoRA 배포 | inference reasoning을 줄이는 유망한 teacher 방식. 과거 reasoning선택의 모방은 미래행동결과 학습과 별개 |
| LLaRP | frozenLLM 둘레에 trainable inputbridge,actor,scalarV; actual rollout/PPO. 로컬에서 원본critic/GAE를 격리실행 검증 | head 정책의 가능성 근거지만 embodied환경/대규모경험이며 우리의 cachedfixedfeatures+48문항과 다름 |

Primary sources:
- Jev: https://docs.typesafe.ai/introduction ; https://typesafe.ai/blog/introducing-system-one-models-and-jev
- D3: https://huggingface.co/vllm-sr/d3-nano ; https://huggingface.co/vllm-sr/d3-mini
- Perplexity: https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b ; https://huggingface.co/perplexity-ai/pplx-decider-v1.2 ; https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2
- Kev: https://github.com/jaredpalmer/kev ; https://github.com/jaredpalmer/kev/blob/main/kev/train.py
- Liquid: https://www.liquid.ai/blog/d1-open ; https://huggingface.co/LiquidAI/d1-3B
- Strands: https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21
- SII: https://github.com/amazingljy1206/SII-Decision
- LLaRP: https://arxiv.org/html/2310.17722v2 ; https://github.com/apple-aiml-research/ml-llarp

Availability: anonymous HF API and firstweight HEAD returned200 for Perplexityv1.2/Flash/Strands; metadata gated=false. Perplexity cards still say private/authentication: record this discrepancy, do not infer full runtime success from HEAD. D3nano/mini/Liquid/Kev metadata also nongated. Pin revisions before execution. D3mini and Flash nominal4B each have ~4.54B total parameters: if user2–4B constraint is strict actualtotal, choose D3nano2.21B or d1-3B3.12B instead of silently treating4.54B as4B.

## 가장 직접적인 외부 경고: local decision과 episode success

JEVal 논문의 현재 HTML은11,257개 bilingual판단과 agent분석을 함께 제공한다. tau-bench에서 같은 생성LLM에 Jev actionselection을 붙인 조건은 overall success77.6→66.1%, P50시간267.1→160.6초, P95695.5→816.0초였다. 각 task1회 실행한 해당 harness의 연구자 보고이며 우리가 재현한 수치가 아니다. 빠른 선택과 긴 trajectory의 성공은 별개라는 직접적인 사례다. 최신paper/model 이름 SII-Decision과 초기검색초록 InnerJev가 달라 모델 식별을 주의한다.

Source: https://arxiv.org/html/2610.03935 (Section5.3/Table3). 추론: 우리도 단일step선택정확도·평균속도만으로 성공을 주장하면 안 된다. 확률이 높다고 높은Q값인 것도 아니다; calibration과 expectedreturn은 다른 학습목표다.

## 우리 결과와 연결

첫 critic36fit은 내부12에서 actionmean보다 품질gap/비용 예측 모두 나빴다. 동일구조72fit, 새12monitor의 검증 결과:

| 모델 | action-gap MSE↓ | remaining-token MAE↓ | gap sign accuracy↑ |
|---|---:|---:|---:|
| 기존36critic |.259428|23.5232|.79630|
| 확장72critic |.269107|20.1002|.71914|
| fit72 actionmean |.225459|20.3383|.83642|

37informativepairs/9questions로 coverage충분, 사전gate실패. 비용은개선됐지만 품질판별이악화됐다. 이것은 대규모로 늘리면 해결된다는 증거도, frozenreader불가능의 증거도 아니다. 학습 수를 더 늘리는 것만을 다음 기본전략으로 삼지 않는다. 새actor/QA평가없음; H3유지. 독립재계산·checkpointreplay·split·Drivearchive 검증통과. 근거 lab/reports/decision_future_critic_probe72_20261024/verification.json.

## 벤치마킹을 실제 정책 검증으로 연결하는 최소 설계

1. 공개typed모델 후보는 처음부터 전부 실행하지 않는다. 엄격2–4B 범위라면 D3nano를 첫 외부대조로 고정. nominal4B까지허용되는 비교표에는 Flashv1.2와D3mini를별도로표시. JevAPI는 모델자체weights공개실험과분리하며, 인증/사용요금 설정을검증하지않았으므로 실행완료로쓰지않는다.
2. 저장된 train-only preaction상태에서 same4legalactions를 제시한다. 입력은 사용자질문+실제로관찰한toolhistory뿐; gold,미래F1,남은suffixcost를넣지않는다. 도구인자선택기까지동시에변경하지않는다. 선택지순서불변성은 작은고정샘플한번으로확인.
3. 기존 all-actionH3suffix결과로 선택한행동의terminalF1/regret/cost를조회한다. 이는 one-intervention진단이며 closed-loop성공으로표현하지않는다. H3행동일치율은정답률이아니다.
4. 살아남는후보하나만 native동일reader/argumentgeneration/corpus에서짧은fullrollout. 두모델의inputprefill,시간,VRAM까지포함. 외부decision모델이출력token0이라고 totalcompute0은아니다. 공유GPU결과는기능/품질확인; 통제된latency우위주장은보류.
5. 학습방법 비교는 예정된relativequality1회와stateencoder+policy공동학습1회로제한. 기존failedcritic을teacher로승격하지않는다. 공개모델softlabels는prior/warmstart일수있지만 rolloutoutcome을대체하지않는다.
6. long-horizon가설을강화하려면 train소수에서1step intervention과2step coordinated intervention을비교한다. H3suffix에서불리했던첫행동이두번째행동변경후유리해지는지확인. 이두턴probe는새아이디어이며구체manifest미동결/미실행; 큰worldmodel실험아님.

성공정의: frozen basicGRPO대비F1+.05 및generated<=.70동시, correct-retained절약/유용한계속탐색증거, 별도사전고정독립확인. reserved80보호. 베이스LLM교체나LoRA는현재mainarm에암묵도입하지않는다.

## 연구 주장 권고

현재 방어가능: shared-prefill frozen-reader interface에smallactionhead를연결하고counterfactualfutureoutcomes를학습·검증중이다.
현재 방어불가: head만으로일반decisionintelligence를획득했다; actionreadout의확률이critic이다; selection과완성된답변이단일forward에서동시생성된다; typedleaderboard우위가장기정책우위다; accuracy와token목표달성했다.

목표 논문주장: 같은reader표현과cache를재사용하는작은정책모듈이실제다중턴결과로학습되어, 별도decisionLLM의중복prefill없이정확도와비용을개선한다. 이문장은앞으로의검증목표이며현재결론이아니다.
