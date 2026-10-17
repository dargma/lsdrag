# Frozen 도구 head 일반화100 평가

2026-10-17 사용자 “ok” 실행 승인. 현재 Qwen3-4B-Instruct-2507 non-thinking과 v3 head를 고정한다. Native(전체 원래 도구/인자/답변 생성) 대 head_generate(도구만 head 선택, 인자/답변은 같은LLM의 aligned cache 생성). read 인자 선택·재학습·튜닝·fallback·3hop·RL 없음.

## 사전 동결

목록·모델·head·prompt·dataset·실행sourcehash: `lab/registry/text_integrated_20261016/generalization100_manifest_20261017.json`. 기존Git lab/memory/docs와local-fast 결과의MuSiQue qid를보수적으로수집(예약질문포함). 그질문들의train/dev원본분해하위질문과겹치는후보도제외. 2hop answerabledev1252중eligible1099, SHA256(generalization100_20261017:qid)순서에서서로하위질문이겹치지않게100선택(선택중겹침20skip).100질문/200서로다른하위질문/기존qid불중복 CPU검증통과. 이배제로평가대상은원래dev전체와분포가달라질수있다. 가용기록노출감사이며모델pretraining노출은배제불가.

정답·support는채점만사용, 모델입력은질문·publicparagraphs·실제도구결과. 100결과를보며checkpoint/프롬프트/규칙/샘플선택을바꾸지않는다. 이후개선은train자료에서만하고별도새질문평가가필요하다.

## 실행

100질문×2조건=200episode. 짝수문항Native→head/홀수head→Native. maxoutput512/context16384/기존최대8tool+FINAL/greedy. corpus벡터는질문마다local-fast에만생성하고양쪽동일벡터사용;queryembedding은실제tool실행비용에포함. 모델load제외worker30분/Qwen생성150000token/prefill입력+forcedprefix3000000tokenhardcap. 다음호출상한예약으로budget초과를예방. 품질에따른조기중단없음;운영상한도달시미완료로기록하고임의질문교체없음. 공유GPU허용/다른작업종료금지.

주평가pairedF1/EM. 보조근거F1,실제toolcalls/interactionturns(종료·실패시도포함),schema오류,성급한종료·검색대신읽기등분기양상,생성/forcedprefix/입력/prefill/deriveddecoderforward,관측episode walltime. Native의combined생성은인자토큰을별도계측하지않으므로argument_tokens=0을실제인자생략으로해석하지않음.

시간은queryembedding을포함한episode wall. 모델load/문서사전embedding/채점/저장/백업제외. 문서준비시간은별도기록. 순서교대만으로GPU간섭·warmup·trajectory차이를통제하지못하므로speedup입증으로해석하지않음. paired5000bootstrap seed20261017의F1/EM차이구간은불확실성기술용이다. 사전비열등성margin/검정력설정없으므로비열등성확정주장금지. Nativeformat실패도모든문항주분석에보존한다.

실행: `PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /content/local_fast/lsdrag-text-v1/experiment-env/bin/python -u scripts/run_frozen_head_generalization100.py --output /content/local_fast/lsdrag-text-v1/results/frozen_head_generalization100_20261017`

완료뒤결과·원문항별실패·원본archive·memory를Git/Drive에보존한다. 자동새학습/3hop확대없음.
