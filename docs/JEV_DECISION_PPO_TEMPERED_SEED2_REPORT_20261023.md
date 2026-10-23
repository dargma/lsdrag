# Tempered PPO 대 동일배치 RWR 결과

총304episodes,학습수집144+fresh40×4평가160. strong signal gate: True. 단일seed,공유GPU통제latency없음.

| 정책 | F1 | utility | 호출 | 생성token | 오류episode | cap FINAL |
|---|---:|---:|---:|---:|---:|---:|
| historical_bc | 0.3618 | 0.2638 | 2.300 | 56.90 | 0 | 0 |
| tempered_anchor | 0.3618 | 0.2637 | 2.300 | 57.25 | 0 | 0 |
| ppo | 0.4659 | 0.3681 | 2.250 | 60.75 | 0 | 0 |
| rwr | 0.4830 | 0.3815 | 2.300 | 66.78 | 0 | 0 |

## Paired 비교

- PPO−historical_bc f1: +0.1040, 95% CI [0.026254180602006693, 0.2037625418060201].
- PPO−historical_bc utility: +0.1044, 95% CI [0.027441524421451133, 0.20348392036835747].
- PPO−tempered_anchor f1: +0.1040, 95% CI [0.026254180602006693, 0.2037625418060201].
- PPO−tempered_anchor utility: +0.1045, 95% CI [0.027632870973383496, 0.20355642027777776].
- PPO−rwr f1: -0.0172, 95% CI [-0.09864130434782607, 0.06566770186335404].
- PPO−rwr utility: -0.0133, 95% CI [-0.09534558933618012, 0.06845616877329194].

## 학습 및 검증

원본304개재집계,결정1016개확률합/합법성,동결headhash검사통과. 각iteration수집당시logprob와update전logprob parity통과.

- Iteration0: 155학습state, diagnostics={'ppo': {'loss': 0.07214045030996204, 'changed_vs_anchor': 0.012903225608170033, 'ratio_clip_fraction': 0.1225806400179863}, 'rwr': {'loss': 0.8144267983734608, 'changed_vs_anchor': 0.20000000298023224, 'ratio_clip_fraction': 0.3677419424057007}}
- Iteration1: 166학습state, diagnostics={'ppo': {'loss': 0.08094918289377044, 'changed_vs_anchor': 0.054216865450143814, 'ratio_clip_fraction': 0.2469879388809204}, 'rwr': {'loss': 0.6517547561476628, 'changed_vs_anchor': 0.22289155423641205, 'ratio_clip_fraction': 0.3313252925872803}}
- Iteration2: 169학습state, diagnostics={'ppo': {'loss': 0.11190890319024523, 'changed_vs_anchor': 0.06508875638246536, 'ratio_clip_fraction': 0.18934911489486694}, 'rwr': {'loss': 0.777458514397343, 'changed_vs_anchor': 0.17159762978553772, 'ratio_clip_fraction': 0.44378697872161865}}

PPO가수집한동일배치를RWR도사용한다. 같은데이터이나수집정책은PPO에맞춰져있어독립on-policy학습법의공정대결은아니다. BCtemperature8로탐색성을주고trunk고정/action·skip·critic만학습했다. full-backbone RL이나구조탐색성과가아니다. 새test40은이후학습/튜닝금지.

학습식: return=F1−.03×remainingcost, advantage=(return−oldcritic)의배치표준화. PPO clipped surrogate(clip.2)+.5valueMSE+.02anchorKL−.01entropy. RWR은exp(clip((return−mean)/.1,−5,5))정규화weight의CE와동일regularizer. Monte Carlo return이며GAE아님.

[동결 protocol](JEV_DECISION_PPO_TEMPERED_20261023.md)
