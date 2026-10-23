# 질문 분리 행동 가치 예측 probe

2026-10-23 사용자 실험 계속 승인. CPU-only bounded 분석, 새 GPU rollout/정책 학습 실행 없음(회귀 head fitting은 실행).

## 가설과 고정 설계

같은 hidden에서 고정4행동의 기대utility를 예측할 수 있는가? 각 합법action별 ridgeρ1 signed regression, 타깃mean(F1−.03cost). 0이하clip을 하지 않는다. 입력표현만 비교: last(primary),question-end,observation-end,last16mean,observationmean,32contextbinmean. 기준은 action별상수평균 및 metadata(turn,legalmask,BC4확률) 회귀, historical BC행동. 후보별동적표현/attention/MLP/RL은이번범위밖.

고정round0 67state/24train질문. 기존family6fold를그대로사용하며 heldout family의모든state/outcome을fitting에서제외. action별train통계로표준화하고 질문균등가중. 같은상태의기존2suffix평균을target으로사용. reference가전체outcome으로학습된round1은제외. 추가6suffix를모은12상태의outcome은학습에사용하지않고보조평가에만사용. 기존BC/reference는train24teacherlabel을본적있으며집계결과도이미봤으므로pristineunseen평가가아닌retrospective representation probe.

주지표: 질문균등 OOF선택행동utility−BC utility. 보조:행동선택regret,legalaction평균Q MSE,추가6회12상태선택gain. fixed OOF questionbootstrap4000(seed20261029),재학습/수집seed불확실성미포함. 주조건last가BC및metadata둘모두대비95%하한>0일때만전진gate통과. aux표현의최대값을새주조건으로바꾸지않는다. 하이퍼파라미터검색없음.

OOF선택은heldout결과를읽지않으며 평가만기존결과표로진행. 이후suffix는고정reference이므로새head전체rollout품질증거가아님. 결과가음수면target은있어도현재표현/추정기/데이터로예측하기어렵다는판정이며전체hidden불가능론으로확대하지않는다. 다음실험은결과와한계에따라별도protocol.

실행: PYTHONPATH=src /content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/run_decision_value_probe.py

동결 manifest: lab/registry/decision_value_probe_20261023/manifest.json. source/input SHA 검증 후 실행.
