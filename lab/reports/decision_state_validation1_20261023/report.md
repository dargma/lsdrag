# Decision head 개선 결과

Split: validation, n=32, policies=10. 동일질문paired비교. 단일학습seed.

| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| grpo_baseline | 33.02 | 54.81 | 3783.5 | 2.188 | 0 |
| same_structure_improved | 34.23 | 50.38 | 3005.2 | 1.562 | 0 |
| last_linear_warmstart | 33.02 | 52.66 | 3713.5 | 2.156 | 0 |
| last_linear | 38.05 | 45.84 | 2679.4 | 1.375 | 0 |
| last_mlp_warmstart | 33.02 | 51.97 | 3621.6 | 2.094 | 0 |
| last_mlp | 44.69 | 62.53 | 4475.8 | 2.312 | 0 |
| multi_linear_warmstart | 33.72 | 53.44 | 3700.4 | 2.125 | 0 |
| multi_linear | 39.13 | 53.50 | 3623.2 | 1.688 | 0 |
| multi_mlp_warmstart | 37.88 | 52.75 | 3695.1 | 2.094 | 0 |
| multi_mlp | 43.12 | 47.66 | 2946.7 | 1.531 | 0 |

## 고정 GRPO 대비

- same_structure_improved: F1 delta +1.21pp, CI [-0.11770833333333333, 0.13958333333333334]; 생성ratio {'ratio': 0.9190421892816419, 'ci95': [0.820931315245759, 1.0227551434326319]}; point목표 False, 방향CI False, 목표CI False.
- last_linear_warmstart: F1 delta +0.00pp, CI [0.0, 0.0]; 생성ratio {'ratio': 0.9606613454960091, 'ci95': [0.9232052495166001, 0.9912898810791078]}; point목표 False, 방향CI False, 목표CI False.
- last_linear: F1 delta +5.03pp, CI [-0.0438034188034188, 0.1583406784188033]; 생성ratio {'ratio': 0.8363740022805017, 'ci95': [0.7635971699009717, 0.9094106404344022]}; point목표 False, 방향CI False, 목표CI False.
- last_mlp_warmstart: F1 delta +0.00pp, CI [0.0, 0.0]; 생성ratio {'ratio': 0.9481185860889396, 'ci95': [0.9089929247524291, 0.9825358718758734]}; point목표 False, 방향CI False, 목표CI False.
- last_mlp: F1 delta +11.67pp, CI [-0.01875, 0.2604166666666667]; 생성ratio {'ratio': 1.1408209806157354, 'ci95': [0.9768344870843978, 1.346427645376293]}; point목표 False, 방향CI False, 목표CI False.
- multi_linear_warmstart: F1 delta +0.69pp, CI [-0.07291666666666667, 0.09375]; 생성ratio {'ratio': 0.9749144811858609, 'ci95': [0.9286926836063599, 1.0163301954867487]}; point목표 False, 방향CI False, 목표CI False.
- multi_linear: F1 delta +6.11pp, CI [-0.03645833333333333, 0.171875]; 생성ratio {'ratio': 0.976054732041049, 'ci95': [0.8007451643867769, 1.2446166474367548]}; point목표 False, 방향CI False, 목표CI False.
- multi_mlp_warmstart: F1 delta +4.86pp, CI [0.0, 0.125]; 생성ratio {'ratio': 0.9623717217787914, 'ci95': [0.922148086603967, 0.9994309943002218]}; point목표 False, 방향CI False, 목표CI False.
- multi_mlp: F1 delta +10.10pp, CI [0.02291666666666667, 0.203125]; 생성ratio {'ratio': 0.8694412770809579, 'ci95': [0.7967337719231123, 0.9491134289439374]}; point목표 False, 방향CI True, 목표CI False.

F1+3%p 및생성token−20% 동시목표. 확률은action확률이며정답확률아님. 단일seed/조건부CI/공식dev-only아닌미사용train+dev원천holdout. 최종test결과로후보재선택금지. 전체학습/수집비용별도보고. 통제latency비교없음.
