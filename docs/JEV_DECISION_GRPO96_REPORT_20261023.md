# GRPO96 결과 — 2026-10-23

학습1152+새80×3평가240=1392episodes. GRPO 대 PPO의 F1/utility CI하한>0 gate: False. 단일 학습 seed.

| 정책 | F1 | utility | 호출 | 생성token | 오류episode | cap FINAL |
|---|---:|---:|---:|---:|---:|---:|
| tempered_anchor | 0.3211 | 0.2190 | 2.388 | 61.30 | 0 | 0 |
| ppo | 0.3427 | 0.2366 | 2.450 | 66.29 | 0 | 0 |
| grpo | 0.3343 | 0.2396 | 2.212 | 56.25 | 0 | 0 |

## Paired 비교

- GRPO−tempered_anchor f1: +0.0132, 95% CI [-0.046785714285714285, 0.0750178571428572].
- GRPO−tempered_anchor utility: +0.0206, 95% CI [-0.03944644553571428, 0.08180390808035715].
- GRPO−ppo f1: -0.0083, 95% CI [-0.09083333333333334, 0.07542708333333337].
- GRPO−ppo utility: +0.0029, 95% CI [-0.07945273906250001, 0.0871909134375].

## 학습 및 검증

원본1392개재집계,결정4608개확률합/합법성,동결headhash검사통과. 각iteration수집당시logprob와update전logprob parity통과.

- Iteration0: 1284학습state, diagnostics={'grpo': {'loss': -0.022475232620064806, 'changed_vs_anchor': 0.014797506853938103, 'ratio_clip_fraction': 0.14018690586090088}}
- Iteration1: 1281학습state, diagnostics={'grpo': {'loss': -0.029653611286293444, 'changed_vs_anchor': 0.04918032884597778, 'ratio_clip_fraction': 0.31928181648254395}}
- Iteration2: 1239학습state, diagnostics={'grpo': {'loss': -0.023107871031149838, 'changed_vs_anchor': 0.10815173387527466, 'ratio_clip_fraction': 0.23486682772636414}}

실제 optimizer updates: 484.

같은 질문4trajectory의전체utility를group-standardize하고trajectory길이역가중. actor action/skip만학습,trunk고정,critic없음. PPO와state weighting도달라알고리즘단일원인분리불가. 강제행동분기자료없음. 모든평가조건에RecoveryPolicyReader적용.

현재관측: GRPO는PPO보다생성/입력/호출평균이작지만F1과utility차이CI는0포함. 정확도동등성·우월성·통제latency절감입증아님. 평가80은이후튜닝금지.

[동결 protocol](JEV_DECISION_GRPO96_20261023.md)
