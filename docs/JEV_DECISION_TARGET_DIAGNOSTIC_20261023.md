# 데이터 양과 상대 이득 target 진단

2026-10-23. 사용자 지속 실행 승인. 기존train24/67state의family6fold 재사용, 같은ridgeρ1. Last hidden 및metadata, absoluteQ 및Q−Q(BC), 학습질문25/50/100%(fold별5/10/20) 비교. Hash순서3seed20261030/31/32로nestedsubset,고정평가fold. 원래2suffix만 사용. 36개CPU fitting조건, GPU추론없음. 실행전 source/inputhash 동결.

주목적은학습곡선과target정의진단이며최고결과를confirmatory성공으로채택하지않는다. 주지표질문균등BC대비선택utilitygain; train/heldout MSE진단,subsetseed범위. target별MSEscale가다르므로target간MSE단순비교금지. 100%subsetseed는동일자료라독립seed재현아님. 절대Q100%결과는직전probe와parity검사. family누출금지. 전부이미열람한trainpool이므로탐색적결과. PPO/GRPO성과증거아님.

실행: PYTHONPATH=src /content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/run_decision_target_diagnostic.py

후속: 작은PPO비교조건설계. 현재회귀의성공을전제하지않으며on-policy수집과실제rollout평가를별도수행. 과거노출test튜닝금지.
