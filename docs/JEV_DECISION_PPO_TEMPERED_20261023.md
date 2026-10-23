# 탐색성을 확보한 PPO 대 동일배치 보상 가중 모방학습

2026-10-23. 사용자 지속실험 승인. 1seed bounded functional/quality pilot. 선행CPU회귀진단은relative target으로해결되지않았고last train/heldout오차격차가컸다. PPO성공을전제하지않고직접검증한다.

## 동결 조건

Qwen3-4B-Instruct-2507 nonthinking, 기존4action/생성기. pilot2 reference_round0 (기존BC temperature8)을anchor/초기actor로사용. same checkpoint, MLP trunk 및normalizer고정,action/skip/value만학습. value zero초기화로actor공유trunk변화없이critic학습. epsilon=0으로tempered정책자체에서sample;과거epsilon mixture에의한gradient약화와구분. PPO clip.2,AdamW1e-4,4epochs,batch32,gradnorm1,KL anchor.02,entropy.01,valueMSE.5. 총3iterations×train24×2=144on-policyepisode. 매iteration PPO가수집하며behavior logprob parity검사. sampledforcedFINAL은actor학습제외,그비용은MCreturn에포함.

return=terminalF1−.03remainingcost,discount1,advantage=MCreturn−수집시critic;배치표준화. GAE아님. RWR은같은배치에서weight=exp(clip((return−batchmean)/.1,−5,5))를평균1정규화하여weighted CE. 같은초기화/epoch/minibatch/regularizer. PPO clip 효과외에도목표형태가다르므로알고리즘단일요인분해는아님. PPO수집정책으로RWR도학습하는비대칭을명시하며독립on-policyRWR비교라고하지않는다.

## 평가/예산

fresh40은기존노출qid와sharedsubquestionfamily를제외해hash순서로동결. 4조건 historicalBC/temperedanchor/PPO/RWR×40=160eval. 총304episodes,load제외1800초,생성400000/input12000000상한. 모든설정/finaliteration고정,validation선택없음. 신규test40은이후튜닝금지.

주비교 PPO−temperedanchor: pairedF1/utility 및questionbootstrap4000(seed20261033),call/token/error/cap보고. PPO−RWR/historicalBC는보조. 품질과utility모두95%하한>0일때만강한전진신호로판정하며단일seed한계유지. 평균개선만으로효율정책채택안함. 통제latency주장없음. 완전한최신방법벤치마크나신규성증거아님.

원본run /content/local_fast/lsdrag-text-v1/results/decision_ppo_tempered_20261023
명령 PYTHONPATH=src /content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/run_decision_ppo_tempered.py
동결manifest lab/registry/decision_ppo_tempered_20261023/manifest.json

## 평가 결과 열람 전 독립 학습 seed 반복 동결

첫실행평가outcome/summary열람전에seed20261034를추가동결. 동일train24/test40/hyperparameters와초기checkpoint,독립on-policy수집·update RNG. 첫worker종료후순차실행. 별도run decision_ppo_tempered_seed2_20261023,동일runner를import하여REG/OUT만별도namespace로설정. 304episodes상한동일. 각seed를별도보고하고합산은같은40질문의seed평균 paired차이로계산(80독립질문아님). seed2만잘나왔다고채택하지않는다. 실험전체상한608episodes.
