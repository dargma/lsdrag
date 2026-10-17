# 도구 head 일반화·정교화 실험 초안 — Astra 검토 전

2026-10-17. 사용자 “해봐”로 일반화·정교화 실험이 승인됐고, 직후 “계획 Astra 검토 및 여러 방향 비교”를 요청했다. 새 GPU worker는 아직 시작하지 않았다. 아래는 검토 대상 초안이며 최종 실행 protocol은 아니다. 새 준비 코드 `/content/local_fast/lsdrag-text-v1/drafts/text_head_generalization_20261017.py`는 아직 smoke/검증되지 않았다.

## 기존 증거

- frozen Qwen3-4B-Instruct-2507, keyword/semantic/read/FINAL, train24의82상태, validation8의29상태, head seed1.
- 통합 checkpoint: `/content/local_fast/lsdrag-text-v1/results/improvement_v3_merged_20261017/head.pt`, SHA256 `534da7b64150878869850cf7ccb735ec0fa8fbd643ddd0a74db19afeea4ebaef`.
- 기존 평가20: Native F1.29/EM.25, head+always-generated args F1.304286/EM.25. 1 partial-F1승,19동률. 도구step2.20→2.35. 원실행 관측latency1.273→.979s이나 controlled timing 아님.
- 후속 동일20 ON/OFF: OFF F1.304286/EM.25, read 선택ON F1.254286/EM.20. 인자tokens43.6→40.2;8/15read에서생성생략;0승19동률1패. 양쪽step2.35. OFF latency1.013s/ON.951s은 관측값.
- read 선택6고정validation상태slot모방정확도2/6;teacher 선택가능5/6,예측선택2회중slot정답1. 검색인자선택 학습은 없음.
- 생성tokens89.75→60.55,decoder forward산출86.55→59.55지만 생성+강제prefix89.75→91.10,입력tokens3928.5→4266. 전체계산감소입증아님.
- 가까운 문헌은 LAnR(동일hidden latent retrieval/stop), GRASP(같은3tool/RL/turn효율), Stop-RAG(미래검색가치), TabAgent(생성결정classifier대체). 최신9~10월utility/budget 논문까지 [문헌 정리](JEV_RELATED_RESEARCH_UPDATE_20261017.md)에 있음.

## 제안했던 순서

1. frozen 현재head로 미사용MuSiQue2-hop100문항 native/head(always-generated args) paired 평가.
2. 재학습 없이3-hop30문항 native/head 전이 평가.
3. 기존train24만 사용해 linear와 작은MLP head 비교. 학습문항8/16/24 curve는 CPU imitation 보조지표. 새평가 결과로checkpoint선택 금지.
4. 기존validation8만으로 calibration을 정하고 사전고정확률threshold에서 Native로fallback하는 정책. fallback의중복prefill/이미생성한payload를비용에포함. 확률을실제정확도로가정하지않음.
5. 고정관측state에서경로반복/구간계측으로latency원인진단. 다른GPU작업이있으면통제speedup주장불가.

문항은모든기존노출qid와분해하위질문family를제외하고hash순서로동결. 새로운test는문항수/한도/primary/secondary/code·model·prompt·head hash를먼저저장. 평가중튜닝/성능조기중단없음. 질문본문·공개paragraph만inference입력,정답/근거는채점용분리. latency는model load/corpus사전embedding/채점/저장제외episode wall과단계별관측값.

## 사용자 최신 검토 요청

위 실험을 지금할지, 아니면 다음 방향에 집중할지 여러 관점에서 비교해야 한다.

- A: 도구 선택 head 정교화/일반화. classifier/imitation/안전한fallback 중심.
- B: 인자 생성까지 커버. 후보선택·GENERATE gate,인자필드/값생성,검색어의의미와품질까지개선. 기존read slot실패의coverage/표현/label/value원인분리.
- C: head를통해interaction축소. diffusion step distillation과비슷한고수준발상이나 retrieval은환경의새관측이필요하며단순solver timestep생략과동일하지않음. teacher trajectory중간search/read를생략/합치고질문정답·근거를유지하는지검증. 단순조기종료와새로운관측을얻는단계압축의차이필요.

검토 항목: 과학적질문/선행연구중복/주장가능범위/품질위험/데이터·정답label요구량/반사실target문제/현학습규모와실행가능성/실제walltime절감/다음최소판별실험/중단·확대기준. 전체연구로드맵과가장먼저할1가지,동시추진하지말아야할가지도제시. 사용자결정이필요한본질적선택과agent가정할수있는운영선택을구분.

새실험자동기동은Astra검토전미실시. 기존연구/모델/결과는보존하며RL·step압축이이미구현/실행된것으로표현하지않는다.
