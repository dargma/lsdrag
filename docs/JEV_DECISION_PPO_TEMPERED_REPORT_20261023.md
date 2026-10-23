# Tempered PPO 대 동일배치 RWR 결과

총304episodes,학습수집144+fresh40×4평가160. strong signal gate: True. 단일seed,공유GPU통제latency없음.

| 정책 | F1 | utility | 호출 | 생성token | 오류episode | cap FINAL |
|---|---:|---:|---:|---:|---:|---:|
| historical_bc | 0.3618 | 0.2638 | 2.300 | 56.90 | 0 | 0 |
| tempered_anchor | 0.3618 | 0.2637 | 2.300 | 57.25 | 0 | 0 |
| ppo | 0.4783 | 0.3873 | 2.050 | 61.38 | 0 | 0 |
| rwr | 0.4780 | 0.3858 | 2.075 | 62.35 | 0 | 0 |

## Paired 비교

- PPO−historical_bc f1: +0.1165, 95% CI [0.0014957264957264987, 0.24427350427350425].
- PPO−historical_bc utility: +0.1235, 95% CI [0.01076598219551283, 0.2502155756944445].
- PPO−tempered_anchor f1: +0.1165, 95% CI [0.0014957264957264987, 0.24427350427350425].
- PPO−tempered_anchor utility: +0.1236, 95% CI [0.01088666649572651, 0.2503288463194445].
- PPO−rwr f1: +0.0003, 95% CI [-0.03124999999999999, 0.03214285714285714].
- PPO−rwr utility: +0.0014, 95% CI [-0.027569566249999993, 0.03228399214285714].

## 학습 및 검증

원본304개재집계,결정980개확률합/합법성,동결headhash검사통과. 각iteration수집당시logprob와update전logprob parity통과.

- Iteration0: 163학습state, diagnostics={'ppo': {'loss': 0.11643589272474249, 'changed_vs_anchor': 0.0061349691823124886, 'ratio_clip_fraction': 0.12269937992095947}, 'rwr': {'loss': 0.6723324403865263, 'changed_vs_anchor': 0.20858895778656006, 'ratio_clip_fraction': 0.4110429286956787}}
- Iteration1: 154학습state, diagnostics={'ppo': {'loss': 0.0652912568533793, 'changed_vs_anchor': 0.08441558480262756, 'ratio_clip_fraction': 0.15584415197372437}, 'rwr': {'loss': 0.7570564791560173, 'changed_vs_anchor': 0.2532467544078827, 'ratio_clip_fraction': 0.40909090638160706}}
- Iteration2: 154학습state, diagnostics={'ppo': {'loss': 0.07727033914998174, 'changed_vs_anchor': 0.20129869878292084, 'ratio_clip_fraction': 0.4025973975658417}, 'rwr': {'loss': 0.635782778263092, 'changed_vs_anchor': 0.20779220759868622, 'ratio_clip_fraction': 0.48701298236846924}}

PPO가수집한동일배치를RWR도사용한다. 같은데이터이나수집정책은PPO에맞춰져있어독립on-policy학습법의공정대결은아니다. BCtemperature8로탐색성을주고trunk고정/action·skip·critic만학습했다. full-backbone RL이나구조탐색성과가아니다. 새test40은이후학습/튜닝금지.

학습식: return=F1−.03×remainingcost, advantage=(return−oldcritic)의배치표준화. PPO clipped surrogate(clip.2)+.5valueMSE+.02anchorKL−.01entropy. RWR은exp(clip((return−mean)/.1,−5,5))정규화weight의CE와동일regularizer. Monte Carlo return이며GAE아님.

[동결 protocol](JEV_DECISION_PPO_TEMPERED_20261023.md)
