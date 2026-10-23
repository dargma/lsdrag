# PPO96×4 trajectories 결과

총1312episodes,학습수집1152+fresh40×4평가160. strong signal gate: False. 단일seed,공유GPU통제latency없음.

| 정책 | F1 | utility | 호출 | 생성token | 오류episode | cap FINAL |
|---|---:|---:|---:|---:|---:|---:|
| historical_bc | 0.3307 | 0.2302 | 2.325 | 62.67 | 0 | 0 |
| tempered_anchor | 0.3307 | 0.2303 | 2.325 | 62.35 | 0 | 0 |
| ppo | 0.3907 | 0.2883 | 2.325 | 66.42 | 1 | 0 |
| rwr | 0.3907 | 0.2814 | 2.475 | 71.15 | 1 | 0 |

## 평가 복구 기록

{'preserved_existing': 1222, 'missing_completed': 90, 'original_failed_cell': ['2hop__19007_60935', 'ppo'], 'failure_score_retained': True, 'original_failed_attempt_extra_cost_unavailable': True, 'no_retraining': True}
학습1152회완료후평가70개시점에grammar mismatch로중단. 기존1222개episode는bytehash보존,누락90개만실행. 첫실패cell은재실행성공여부와무관하게F1=0으로보존. 해당cell비용은재실행proxy이고최초실패시도추가비용은알수없어총실제비용을완전히집계한것은아님. 1312는정규집계episode수이며실패한원시도를포함한총attempt수와다르다.

## Paired 비교

- PPO−historical_bc f1: +0.0601, 95% CI [-0.034912701586032255, 0.15850793016126882].
- PPO−historical_bc utility: +0.0581, 95% CI [-0.03650662791533388, 0.1564611993675863].
- PPO−tempered_anchor f1: +0.0601, 95% CI [-0.034912701586032255, 0.15850793016126882].
- PPO−tempered_anchor utility: +0.0580, 95% CI [-0.036511420415333874, 0.15627151124258631].
- PPO−rwr f1: +0.0000, 95% CI [0.0, 0.0].
- PPO−rwr utility: +0.0069, 95% CI [0.0014243887499999987, 0.013647496875000002].

## 학습 및 검증

원본1312개재집계,결정4468개확률합/합법성,동결headhash검사통과. 각iteration수집당시logprob와update전logprob parity통과.

- Iteration0: 1284학습state, diagnostics={'ppo': {'loss': 0.11815391932954876, 'changed_vs_anchor': 0.03037383034825325, 'ratio_clip_fraction': 0.1993769407272339}, 'rwr': {'loss': 0.8116793571630629, 'changed_vs_anchor': 0.061526477336883545, 'ratio_clip_fraction': 0.32710278034210205}}
- Iteration1: 1308학습state, diagnostics={'ppo': {'loss': 0.1088129336599306, 'changed_vs_anchor': 0.06574923545122147, 'ratio_clip_fraction': 0.16743120551109314}, 'rwr': {'loss': 0.7303506948235559, 'changed_vs_anchor': 0.12308868765830994, 'ratio_clip_fraction': 0.3707951307296753}}
- Iteration2: 1337학습state, diagnostics={'ppo': {'loss': 0.10752329923811235, 'changed_vs_anchor': 0.10845176130533218, 'ratio_clip_fraction': 0.23036649823188782}, 'rwr': {'loss': 0.7057239629683041, 'changed_vs_anchor': 0.19147345423698425, 'ratio_clip_fraction': 0.3462976813316345}}

실제optimizer update수(각learner): 496. 질문96개와update수는다르다. 이전24문항실험과는수집량/update수/평가질문도달라점수차이만으로데이터량의순수효과를주장하지않는다.

PPO가수집한동일배치를RWR도사용한다. 같은데이터이나수집정책은PPO에맞춰져있어독립on-policy학습법의공정대결은아니다. BCtemperature8로탐색성을주고trunk고정/action·skip·critic만학습했다. full-backbone RL이나구조탐색성과가아니다. 새test40은이후학습/튜닝금지.

학습식: return=F1−.03×remainingcost, advantage=(return−oldcritic)의배치표준화. PPO clipped surrogate(clip.2)+.5valueMSE+.02anchorKL−.01entropy. RWR은exp(clip((return−mean)/.1,−5,5))정규화weight의CE와동일regularizer. Monte Carlo return이며GAE아님.

[동결 protocol](JEV_DECISION_PPO96_20261023.md)
