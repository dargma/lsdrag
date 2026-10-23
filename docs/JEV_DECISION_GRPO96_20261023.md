# GRPO96 직접 비교 실행

2026-10-23 사용자명시승인:불필요분석축소,GRPO바로실행및실제weight검사. 72state근거진단은뒤로미룸(미실행). 동일train96/group4/3round=1152on-policytrajectory. 기존temperedanchor부터시작하여PPO96과학습자료량·초기화·학습률·epoch/clip/KL/entropy를유지. critic은사용/학습하지않음. 새family-disjoint80질문에서anchor/PPO96/GRPO를같이평가240회,전체1392. 평가질문확대와동일집합비교로이전서로다른40평가문제해소. 기존test재튜닝없음.

## 방법

각질문4trajectory의R=F1−.03전체cost를그룹평균/표준편차(+1e-6)로표준화한A를해당trajectory의모든controllableaction에부여. 같은보상이면A=0. trajectory별길이역가중으로각trajectory동일비중;PPO96은state균등이므로이차이도명시한다. action선택확률ratio의clippedsurrogate(.2),정확한4actioncategoricalanchorKL.02,entropy.01. minibatch32,4epochs,AdamW1e-4,gradnorm1. 행동/skip만학습,trunk/backbone고정. GAE/valueMSE없음. 같은utility라도그룹내작은cost차이가std정규화로증폭될수있어groupzero/variance를별도진단. 논문GRPO의head-level adaptation이며전체LLMtokenGRPO재현이아님.

동일seed20261037:첫수집초기policy와RNG는PPO96과맞추고이후는각학습policy에서독립on-policy수집. 후속경로차이로동일state/update수보장안함. 기존grammar오류는RecoveryReader로모든GRPO학습/새평가에서episode실패로기록,성공할때까지재시도안함. PPO96학습완료checkpoint고정·재학습안함.

## 판정

주비교GRPO−PPO96F1/utility pairedbootstrap4000;둘하한>0이면강한우위신호. anchor보조. 단일seed80문항한계,실행시간은통제비교아님. tool/input/generated/error/cap별도보고. 최대load제외3600초,생성1600000/input40000000,총1392episode.

실행 PYTHONPATH=src /content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/run_decision_grpo96.py
원본 /content/local_fast/lsdrag-text-v1/results/decision_grpo96_20261023
manifest lab/registry/decision_grpo96_20261023/manifest.json
참고: https://arxiv.org/html/2402.03300v3 (GRPO), https://arxiv.org/abs/1707.06347 (PPO).
