# 질문 분리 행동 가치 예측 결과

retrospective train-family-held-out head fit conditional on teacher-trained frozen BC/reference; not pristine unseen end-to-end evaluation

동일67상태/24질문/6family fold,동일ridgeρ1·action별 signed utility target. 추가6회 outcome은 fitting에 사용하지 않음.

| 표현/기준 | BC대비 utility gain | 95% CI | regret | Q MSE | 추가6회12상태 gain |
|---|---:|---|---:|---:|---:|
| action_mean | 0.0131 | [-0.06998820097824414, 0.10226234347083844] | 0.1578 | 0.1530 | 0.0903 |
| metadata | 0.0066 | [-0.07431507500372803, 0.09563673708320436] | 0.1644 | 0.1508 | 0.0903 |
| last | -0.0362 | [-0.08390950698065884, 0.005466656188543857] | 0.2071 | 0.2123 | -0.0574 |
| question_end | -0.0169 | [-0.06611905177744806, 0.03262841068349374] | 0.1878 | 0.2433 | 0.0396 |
| observation_end | -0.0293 | [-0.09686182156130406, 0.03825416283252543] | 0.2002 | 0.2395 | 0.0386 |
| last16_mean | -0.0473 | [-0.10657363209683206, 0.008329013866341994] | 0.2182 | 0.2159 | -0.0389 |
| observation_mean | 0.0030 | [-0.07433549574870078, 0.085955005518729] | 0.1679 | 0.2345 | 0.0173 |
| context_bin_mean | 0.0161 | [-0.05426141649620886, 0.09968430690080476] | 0.1548 | 0.2304 | 0.0907 |

주조건 last−metadata: -0.0428, CI [-0.12226062324472403, 0.03256043464518699]. Gate passed: False.

CI는고정OOF예측의질문bootstrap이며재학습/독립수집seed변동을포함하지않는다. 사후최고aux표현을확정채택하지않는다. Qtarget은고정reference suffix결과이며,새head가이후모든행동을결정하는closed-loop성능이아니다. 정책선택확률/성공확률calibration도검증하지않았다.

[사전 계획](JEV_DECISION_VALUE_PROBE_20261023.md)
