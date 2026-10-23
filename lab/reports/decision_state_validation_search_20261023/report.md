# 입력·구조·비용 선호 validation 탐색

동일32문항. 후보선택에사용한자료이므로최종목표달성/일반화증거로사용하지않음. 최신목표F1+5%p/생성token−30%.

| 후보 | F1 | 생성token | F1 차이(pp) | 생성 절감(%) | 공동 point 목표 |
|---|---:|---:|---:|---:|---|
| grpo_baseline | 33.02 | 54.81 | +0.00 | +0.00 | False |
| same_structure_improved | 34.23 | 50.38 | +1.21 | +8.10 | False |
| last_linear_warmstart | 33.02 | 52.66 | +0.00 | +3.93 | False |
| last_linear | 38.05 | 45.84 | +5.03 | +16.36 | False |
| last_mlp_warmstart | 33.02 | 51.97 | +0.00 | +5.19 | False |
| last_mlp | 44.69 | 62.53 | +11.67 | -14.08 | False |
| multi_linear_warmstart | 33.72 | 53.44 | +0.69 | +2.51 | False |
| multi_linear | 39.13 | 53.50 | +6.11 | +2.39 | False |
| multi_mlp_warmstart | 37.88 | 52.75 | +4.86 | +3.76 | False |
| multi_mlp | 43.12 | 47.66 | +10.10 | +13.06 | False |
| last_linear_lambda0.01 | 40.00 | 61.16 | +6.98 | -11.57 | False |
| last_linear_lambda0.06 | 38.44 | 41.03 | +5.42 | +25.14 | False |
| multi_mlp_cost006 | 40.00 | 42.66 | +6.98 | +22.18 | False |
| multi_mlp_generation015 | 41.00 | 49.16 | +7.98 | +10.32 | False |
| last_linear_cost009 | 38.44 | 41.03 | +5.42 | +25.14 | False |
| last_linear_lambda0.06_seed2 | 53.10 | 59.66 | +20.08 | -8.84 | False |
| ordinary_grpo_continuation | 41.00 | 46.69 | +7.98 | +14.82 | False |

warmstart는같은구조를train에서GRPO분포로distill한대조. 이후counterfactual outcome정책학습효과와구조변경효과를혼동하지않는다. 모든구조에새LLM수집을반복하지않고동일191state1428episode를공유했다. seed2는별도수집/학습이다.

예측확률전체를사용하는기대효용목적은명시적가치모델이나미래상태예측모델이아니다. 생성token직접비용후보와기존호출/생성/입력proxy비용후보를구분한다.
