# 행동 후 최종 정답 가능성 기반 본 실험

사용자가 인사이트2~3개를얻어본실험으로진행하라고승인했다. API사용없음. 현재GRPO를완료하고추가candidate하나만검증한다. 설계전략은공유표현,실제후속EM결과,정답효율이다. 기존구조스윕/종료bias를재개하지않는다.

## 입력과 학습

현재GRPO수집192경로는모두업데이트전value_aux정책에서새로샘플링된다. 그훈련batch의FP32사전행동상태와실제선택행동을사용한다. label은그행동후최종EM과남은모든생성토큰(이후강제FINAL포함)이다. 이미관측한과거토큰은잔여비용에서제외한다. 관측하지않은행동의결과를만들지않는다. 이자료의continuation정책은value_aux이며업데이트후GRPO가아니다.

기존48질문순서앞32fit/뒤16calibration을동결한다. calibration결과로재학습/epoch선택/threshold조정을하지않는다. 기존base는과거학습에서질문을봤으므로완전독립일반화주장금지; 새outcomelabel의질문분리만성립한다.

동결된value_aux의128hidden에서4개sigmoidEMreadout과4개positive log(1+remainingtokens)readout을학습한다. 총1032파라미터,base/normalization은완전히동결. 선택행동에만 BCE+smoothL1,질문균등가중,100AdamW.001seed20261086,gradclip1. 미관측행동을0으로labeling하지않는다. 원policy선택확률을정답확률로사용하지않는다.

## 먼저 예측의미 확인

훈련자료에서행동별상수EM확률(질문가중평균)과상수log비용(질문가중중앙값)을baseline으로동결한다. fit에서각행동이5개이상질문,calibration에서2개이상질문에실제관측되어야한다. calibration에서EM Brier와log비용MAE가각baseline보다모두낮아야정책실행평가로넘어간다. 이gate는작은개발상의실현가능성검사이며확률보정/인과성/일반화보장이아니다. 실패하면결과를보존하고GPU평가/재튜닝없이종료한다.

## 실제 행동선택

현재state에서reference=기존value_aux의argmax. 합법적후보중예측EM이reference보다0.05이상높고예측잔여토큰이reference의90%이하인경우에만최저예측비용행동으로대체한다. 없으면reference와동일logits를반환한다. 출력q는4행동합이1일필요없는독립결과점수다. 확률이잘맞는다고가정하지않고위gate와실제rollout으로검증한다. 고정margin은본계획의설계선택이며MS의미공개알고리즘이아니다.

## 평가·중단·보존

현재GRPO전체pipeline이끝날때까지새GPU평가를대기한다. gate통과시동일dev64전후128회, 기존reader/도구/인자생성/코퍼스동일,reference64의과거value_aux원본정확재현검사. 사용자최신지표:전체평균과EM정답평균토큰둘다보고. 성공screen:F1/EM유지이상+두평균감소+공통정답총토큰비증가+오류0. 정답문항구성변화에의한착시를분리한다. reserved80/final80보호. seed/계수/구조추가탐색없음.

CPUgate실패든rollout완료든fit/checkpoint/labels/원본을Drive에백업하고Git에작은기록/보고서저장한다. 한후보의결과로끝내며추가모델이나API를호출하지않는다. 코디네이터 `scripts/continue_qwen35_outcome_rerank.py`,registry `lab/registry/qwen35_outcome_rerank_20261025/`,status `results/qwen35_outcome_rerank_20261025_pipeline/status.json`.
