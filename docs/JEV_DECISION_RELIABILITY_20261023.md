# 고정 reference 행동 가치 반복 검증

2026-10-23. 사용자 계속 지시로 실행. 새 정책 학습이 아닌 train-only 신뢰성 진단.

## 사전 고정

pilot2 round0 train24에서 identity hash(seed 문자열 reliability-20261028)로12질문 선택, 각 질문의 기존 상태 중 identity hash로1상태 선택. outcome/gap은 선택에 사용하지 않음. 이전 aggregate 결과를 본 retrospective 데이터라는 한계는 유지. 기존 reference_round0.pt 그대로, Qwen3-4B-Instruct-2507 non-thinking, 기존 prompt/생성/도구를 유지. 각 합법행동당 새6suffix, 총252episodes, 실행1800초(load제외), 생성300000/input12000000 상한. 동작: keyword_search/semantic_search/read_chunk/FINAL. 새수집 seed20261028, 상태당100 간격, repeat offset; 같은 state/repeat의 action 간 시작 RNG를 맞춤. 분기 길이가 달라 이후 RNG까지 완전히 대응하는 것은 아님.

## 측정

utility=F1−.03cost. 주 측정은 새repeat0:3 평균으로 행동을 고르고 repeat3:6에서 BC대비 utility gain을 계산, 양쪽 방향을 교환해 평균하는 cross-half gain. 기준 BC는 기존 historical head의 동일 state argmax. 질문균등평균과 question bootstrap95%CI4000(seed20261028), 행동 순위 일치, 전체6회 평균으로 선택한 낙관적 gain을 함께 보고한다. FINAL대비 최고 nonFINAL의 gap 및 nonFINAL내 행동차이도 별도 보고. 기존2회 자료와 새6회 최고행동 일치는 보조 진단이다. 동률은 action index순이며 tie 민감도를 고려해 실질 차이0.05 이내도 별도 확인한다.

## 해석과 다음 단계

cross-half gain CI가0을 포함하면 학습 신호가 안정적으로 확인됐다고 주장하지 않는다. 12개 state/고정reference라 전체정책의 독립 seed 재현이나 보편적 효용 증거가 아니다. FINAL 뒤 답변 생성은 greedy이므로 FINAL 반복 동일은 독립 확률 표본 증거가 아니다. 새수집으로 reference변동과 suffix샘플잡음을 분리하는 제한적 진단. 같은 결과로 selector/구조/학습법을 튜닝하지 않으며 완료 후 coverage와 noise를 보고 다음실험을 설계한다.

실행: PYTHONPATH=src /content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/run_decision_reliability.py

Manifest: lab/registry/decision_reliability_20261023/manifest.json. 원본: /content/local_fast/lsdrag-text-v1/results/decision_reliability_20261023. GPU별 통제 latency 주장 없음. 실패·cap은 기록하며 성공만 골라 버리지 않는다.
