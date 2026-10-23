# PPO96: 질문과 trajectory 확대 실행

2026-10-23 사용자명시승인:24/48단계건너뛰고바로96,질문당trajectory도4회로증가. 기존24+새72train,새family-disjoint40test. 다른문항크기/입력위치/구조/GRPO비교는이번에추가하지않음.

## 동결 조건과 예산

96 unique questions×4 stochastic trajectories×3iterations=1152학습episodes. 평가historicalBC/temperedanchor/PPO/RWR×40=160,총1312. 최대load제외3600초,생성1600000/input40000000. seed20261037하나. Qwen3-4B-Instruct-2507 nonthinking,4tools/actions 기존동일. PPO초기head는기존train24teacher 기반temperature8 reference이며checkpoint/trunk/normalizer고정. PPOaction/skip/value만학습. 최종iteration2 평가고정,earlybest선택없음.

PPO clip.2,AdamW1e-4,4epochs,minibatch32,gradnorm1,KL.02,entropy.01,valueMSE.5,MCreturn=F1−.03remainingcost,discount1. 수집시oldcriticbaseline으로advantage표준화. epsilon0,정책자체에서sample. 매iterationbehaviorlogprobparity검사. RWR도PPO가수집한같은배치를사용,기존exp가중CE/.1temperature/동일regularizer. 수집정책비대칭유지한계명시.

질문수≠업데이트수. 한trajectory의controllable결정들이학습state가되고,iteration별4×ceil(state수/32) optimizersteps가발생한다(각learner). 결과에실제state수와update수를기록한다. 질문다양성/trajectory/총update가함께증가하므로순수데이터량원인분리실험아님. 과거24실험과새40평가집합도다르므로점수차이로96의우월성을직접주장하지않음. 초기BC는96문항teacher를새로학습한baseline이아님.

주비교새40에서PPO−temperedanchor F1/utility paired95%CI,둘다하한>0일때강한신호. PPO−RWR/historicalBC보조. 실제calls/input/generated/error/cap분리보고. 통제latency주장없음. 새test는이후튜닝금지. dataset/source/checkpoint SHA 동결manifest검증.

원본 /content/local_fast/lsdrag-text-v1/results/decision_ppo96_20261023
명령 PYTHONPATH=src /content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/run_decision_ppo96.py
manifest lab/registry/decision_ppo96_20261023/manifest.json

기존사례분석은새96확대요청이우선하여pending. 이미본평가사례를새train72선정에사용하지않음(hash/중복family제외만사용).
