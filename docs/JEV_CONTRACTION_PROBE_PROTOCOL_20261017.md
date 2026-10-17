# 제한된 interaction 축소 가능성 진단 — 실행 protocol

2026-10-17 사용자 “그래”로 Astra 권고의 작은 진단 실행 승인. 기존 train24/validation8을 그대로 사용하며 새 학습·RL·인자 decoder·새 test는 없다. 기준 모델/head/prompt/도구는 기존 v3와 동일하다.

## 동결 입력과 비교

`lab/registry/text_integrated_20261016/contraction_probe_manifest_20261017.json`에 질문별 첫 실제 tool 결과 직후 1상태, source/message/history/model/prompt/head/code 해시를 GPU 실행 전에 저장했다. teacher 미래 suffix는 inference 입력으로 사용하지 않는다. branch마다 prefix를 다시 prefill하여 KV를 격리한다.

같은 상태에서 head 정상 진행 / native name-boundary selector+공통 aligned generator / 즉시 FINAL / head 최초 선택과 다른 합법 tool 최대2개(keyword→semantic→read)를 비교한다. 이후는 전부 frozen head+generated args이다. read는 실제 관측 ID만 허용한다. 모든 branch는 남은 tool 최대2회+FINAL이다. head/native의 첫 suffix tool 뒤 B=1용 FINAL을 별도로 생성한다. 과거 최종 답을 당겨 쓰지 않는다.

Native selector는 공통 prefill에서 greedy로 tool 이름이 구별되는 지점까지 생성한다. selector용 KV는 복제하고 공통 generator는 원 cache를 사용한다. 발생한 routing token·복제시간·이후 일괄 syntax 처리비를 모두 센다. 따라서 syntax가 중복 처리되는 진단 selector이며 최적화된 Native 생산 경로/전체 Native 정책과 같지 않다. 무료 selector 또는 인과적 speed baseline으로 해석하지 않는다.

최초 train8(기존 split 첫8질문)은 Native full output과 동일 action의 aligned 값/이름 파싱/원 cache 길이 보존을 검사한다. 단 한 건의 기능 불일치도 확대 전에 중단한다. 이8질문의 품질은 통과 gate가 아니며 나쁜 결과 때문에 샘플을 바꾸지 않는다. 구조 검사4개 CPU 통과(분기별입력 불변/중간snapshot격리/미관측ID차단/예산 및name구분). 첫 fixture는 실제 tool history를 메시지에 싣지 않아 실패했고 fixture를 실제 메시지 계약에 맞춰 수정했다.

## 한도와 계측

최대160 branch+16 path checks+64 B1 answer probe=240건. 모델 로딩 후30분, Qwen 생성120000token, Qwen full-prefill입력+forcedprefix2000000token 중 먼저 닿으면 미완료로 기록한다. 호출 전 최대 토큰 예약으로 cap 초과 방지. 각 호출 max512/context16384. 문항 순서는 기존train24 후validation8, 조건은 head/native/FINAL/다른tool이며 순서교대가 없으므로 latency causal claim 금지. 공유 GPU 타작업 종료 금지.

생성/forced token, prefill 수/길이, decoder 경로 usage, 실제 tool walltime, suffix walltime, decision turn, primitive call, EM/F1, 관측된gold근거recall/읽은근거recall/citation ID유효성을 기록한다. corpus 사전 embedding은 저장 index를 사용하며 model load/채점/저장은 suffix 시간에서 제외한다. embedding query 비용은 tool 시간에 포함한다. 전체trajectory 도구/turn수는 공통prefix1회를 포함하지만 prefix walltime은 과거 teacher 기록이다. 과거 prefix와 새 suffix의 합을 새 통제된 end-to-end latency라고 하지 않는다.

## 분석과 후속 gate

train/validation 분리 집계. Oracle는 gold 채점으로 시험한 branch 중 정답·F1·관측근거recall을 보존하는 더 싼 branch를 고르는 특권 상한이다. 학습 정책 성능이 아니다. baseline EM정답 최소5train질문, 그 정답subset에서 per-question품질/근거비하락·정답보존 및 총 primitive call 또는 decision turn20%절감, 독립질문 최소3개의 절감 사례를 후속 투자 기준으로 둔다. baseline그대로를 선택할 수도 있어 oracle품질보존 자체는 성능증거가 아니다. 중간transition압축 주장에는 nonterminal대체 성공3질문이 따로 필요하다. FINAL-only이득이면 early stopping으로 기록한다. 모두 오답인 문항의 빠른 오답은 긍정적 압축 사례로 세지 않는다. 비용 성분이 서로 상충하면 전체효율판정은 보류한다.

validation8은 이미 노출된 확인 진단이다. 임계값/branch/샘플을 바꾸지 않으며 비열등성·새로운독립일반화 주장을 하지 않는다. 결과가 좋아도 자동으로 새 head/RL을 시작하지 않는다. 2hop100/3hop30 확인평가는 이후 과제다.

## 명령

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /content/local_fast/lsdrag-text-v1/experiment-env/bin/python -u scripts/run_text_contraction_probe.py --output /content/local_fast/lsdrag-text-v1/results/contraction_probe_20261017
```

이 protocol이 [Astra 검토](JEV_HEAD_DIRECTION_ASTRA_REVIEW_20261017.md)의 다음 진단을 구체화한다. 문헌독창성/stepdistillation학습이 이미 검증된 것으로 표현하지 않는다.
