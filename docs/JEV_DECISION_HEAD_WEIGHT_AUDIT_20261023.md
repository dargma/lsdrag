# 실제 head weight 및 출력 검사

실제.pt파일을CPU로열어전체tensor shape/NaN/Inf/norm/최댓값/초기대비차이검사. 기존train67hidden에forward하여합법확률/합1/포화/행동분포확인. 품질이나이해능력검증을대체하지않는다.

| head | finite | trunk/normalizer 동일 | maxp 중앙값 | maxp>.99 | 행동별수 | anchor대비변경 |
|---|---|---|---:|---:|---|---:|
| anchor | True | True | 0.734 | 0.000 | [23, 14, 10, 20] | 0 |
| ppo96 | True | True | 0.736 | 0.045 | [25, 18, 6, 18] | 7 |
| rwr96 | True | True | 0.737 | 0.045 | [23, 21, 8, 15] | 11 |

행동순서keyword/semantic/read/FINAL. 금지logit−1e9는mask이므로폭주로해석하지않고legal값만통계. success/cost출력층은현재훈련대상아니어서초기값유지,성공확률로해석금지. critic은finite여도예측력이약할수있음: PPO96 on-policy explained variance는후기round에서약−.001/−.007이었다. Critic영향과actor가중치정상여부를구분한다.

## 학습층 변화

| head | tensor | norm | delta norm | max abs |
|---|---|---:|---:|---:|
| ppo96 | action.weight | 0.18490 | 0.05236 | 0.02221 |
| ppo96 | action.bias | 0.01654 | 0.00948 | 0.01263 |
| ppo96 | skip.weight | 0.31143 | 0.23088 | 0.01494 |
| ppo96 | skip.bias | 0.01003 | 0.00948 | 0.00856 |
| ppo96 | value.weight | 0.15626 | 0.56118 | 0.02984 |
| ppo96 | value.bias | 0.03528 | 0.03773 | 0.03528 |
| rwr96 | action.weight | 0.20671 | 0.08080 | 0.02582 |
| rwr96 | action.bias | 0.01954 | 0.01806 | 0.01203 |
| rwr96 | skip.weight | 0.34011 | 0.26401 | 0.01400 |
| rwr96 | skip.bias | 0.01798 | 0.01805 | 0.01635 |
| rwr96 | value.weight | 0.15493 | 0.56053 | 0.03180 |
| rwr96 | value.bias | 0.03743 | 0.03989 | 0.03743 |

현재파일은손상/NaN/전체한행동붕괴같은명백한garbage징후가없다. Weight범위정상만으로학습의유용성을주장하지않으며실제heldout품질과함께판단한다. Critic은zero초기화했으므로표의anchor대비변화에는reset효과가포함됨.
