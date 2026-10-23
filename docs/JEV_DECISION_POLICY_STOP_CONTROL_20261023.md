# 순차 실험 1: 종료 전용 대조군 — 2026-10-23

사용자 “그래 차례차례진행하자”로 승인. 첫 단계는 Stop-RAG-inspired shared-feature control이며 원 논문 재현이 아니다. 현재 모델과 네 도구 계약은 그대로다. 새 Qwen 후보로 교체하지 않는다. 다음 선형value·독립 seed 검증은 이 단계와 별도 기록한다.

## 판별 질문

전체 행동정책의 비용 절감은 종료 판단만 개선해도 얻는가? 계속하기로 했을 때 기존 BC의 합법 non-FINAL argmax 도구를 바꾸지 않는 gate와 비교한다.

## 동결 조건

- 기존 train24/val8, pilot2 round0/1의67/72상태·990분기 결과 재사용. 새 수집비용0이며 재사용 원래 수집비용도 보고.
- 최종 품질−.03·남은비용 target, 동일 reference suffix, 질문별 동일 가중치. Two rounds,100epoch/round,lr.001,anchor KL.02,entropy.01. Threshold/model selection 없음.
- Binary gate는 historical BC의 CONT최대logit/FINAL logit을8로 나눈 baseline에 128tanh+linear skip residual을 학습. Residual 출력은0으로 초기화하고 trunk는 기존 BC MLP에서 가져온다. 초기 greedy 행동은 historical BC와 동일. Frozen BC tool selector는 buffer이며 optimizer가 바꾸지 않는다.
- 계속이면 정해진 BC 도구 하나, 종료면 FINAL. 둘 외 행동확률은0. cap8은 동일. 고정도구선택은 학습전후와 실행 중 확인.
- 전체 정책은 pilot2의 사전 주조건 MLPcost checkpoint를 그대로 사용한다. Test40에서 사후 성능이 더 좋았던 linear head로 바꾸지 않는다.
- 평가: native / historicalBC / fullMLPcost / stop-only, val8+**fresh family-disjoint60**, 총272episodes. 과거test100/20/40과 다른 노출 질문·family 제외. 새60도 평가 후 튜닝금지.
- seed20261025 단일; binary gate 초기화와 전체 policy 초기화가 달라 순수 행동공간 하나만 바꾼 인과분리로 주장하지 않는다. 초기 도구보존은 정확히 검증한다.
- Reference Q는 CONT 이후 전부stop-only로 실행한 Q가 아니다. 실제 평가에서는 stop-only가 모든 단계의 종료를 제어하므로 distribution shift가 존재한다. 비용감소만으로 성공 주장하지 않는다.

## 평가·상한

F1/EM, calls, generated/input/supplied-prefix tokens, proxycost, errors, capFINAL 및 paired F1CI. Primary: stop-only−BC, full−stop-only. 품질 비열등성 허용차를 사후 정의하지 않는다. 공유GPU 관측시간은통제speedup 아님.

CPU14검사 통과. 기동시 추가로139학습상태의 초기BC argmax parity, 학습후 nonFINAL 선택 불변, input/data/source SHA를 확인. Feature inputs에 gold는 없으며 score target만train정답을 사용.

Model load후1800초, 생성400k/input8m,8tools,512output,context16384. Output `/content/local_fast/lsdrag-text-v1/results/decision_policy_stop_20261023`; log `/content/local_fast/lsdrag-text-v1/logs/decision_policy_stop_20261023.log`.

```
PYTHONPATH=src /content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/run_decision_policy_stop.py --output /content/local_fast/lsdrag-text-v1/results/decision_policy_stop_20261023
```

[Frozen manifest](../lab/registry/decision_policy_stop_20261023/manifest.json), [PPO와 현재 학습 수식](JEV_DECISION_POLICY_MATH_20261023.md), [related work/후속 순서](JEV_DECISION_POLICY_RELATED_WORK_20261023.md).
