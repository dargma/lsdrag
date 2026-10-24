# Decision head 개선 결과

Split: validation, n=64, policies=9. 동일질문paired비교. 단일학습seed.

| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| selected_reference | 25.18 | 76.75 | 3870.8 | 1.781 | 0 |
| control | 16.55 | 50.36 | 3005.8 | 1.219 | 0 |
| depth | 16.55 | 58.61 | 3052.8 | 1.266 | 1 |
| input_gate | 16.55 | 58.94 | 3059.1 | 1.281 | 1 |
| depth_input_gate | 16.55 | 59.64 | 3112.4 | 1.297 | 1 |
| observation | 16.55 | 58.98 | 2974.9 | 1.250 | 1 |
| question_attention | 18.37 | 51.48 | 2756.3 | 1.109 | 0 |
| action_attention | 17.72 | 62.91 | 3292.7 | 1.266 | 1 |
| action_set | 16.55 | 48.56 | 2877.3 | 1.172 | 0 |

## 지정 baseline 대비

- control: F1 delta -8.63pp, CI [-0.16703869047619047, -0.015625]; 생성ratio {'ratio': 0.6561482084690554, 'ci95': [0.5130013762189821, 0.8078066157835149]}; point목표 False, 방향CI False, 목표CI False.
- depth: F1 delta -8.63pp, CI [-0.16703869047619047, -0.015625]; 생성ratio {'ratio': 0.7636400651465798, 'ci95': [0.5508760066860932, 1.0567594891663443]}; point목표 False, 방향CI False, 목표CI False.
- input_gate: F1 delta -8.63pp, CI [-0.16703869047619047, -0.015625]; 생성ratio {'ratio': 0.7679153094462541, 'ci95': [0.5596331565027538, 1.0579121305692725]}; point목표 False, 방향CI False, 목표CI False.
- depth_input_gate: F1 delta -8.63pp, CI [-0.16703869047619047, -0.015625]; 생성ratio {'ratio': 0.7770765472312704, 'ci95': [0.56668493345633, 1.071213819550823]}; point목표 False, 방향CI False, 목표CI False.
- observation: F1 delta -8.63pp, CI [-0.16703869047619047, -0.015625]; 생성ratio {'ratio': 0.7685260586319218, 'ci95': [0.5614105094681918, 1.05638752707494]}; point목표 False, 방향CI False, 목표CI False.
- question_attention: F1 delta -6.81pp, CI [-0.14732142857142858, 0.002232142857142858]; 생성ratio {'ratio': 0.6708061889250815, 'ci95': [0.5235288616718188, 0.8129471544031028]}; point목표 False, 방향CI False, 목표CI False.
- action_attention: F1 delta -7.46pp, CI [-0.15792410714285715, 0.0]; 생성ratio {'ratio': 0.8196254071661238, 'ci95': [0.5881724148016475, 1.128855217629863]}; point목표 False, 방향CI False, 목표CI False.
- action_set: F1 delta -8.63pp, CI [-0.16703869047619047, -0.015625]; 생성ratio {'ratio': 0.6327361563517915, 'ci95': [0.49428638933864066, 0.7758238574738073]}; point목표 False, 방향CI False, 목표CI False.

F1+5%p 및생성token−30% 동시목표. 확률은action확률이며정답확률아님. 단일seed/조건부CI/공식train원천의학습/검증가족분리holdout이며공통101962문단검색환경. validation은후보선택자료이므로최종확증아님. 최종test결과로후보재선택금지. 전체학습/수집비용별도보고. 통제latency비교없음.

CI는개별지표의marginal95%구간이며다중비교보정없음. 두CI조건통과를동시95%보장으로해석하지않는다.

## Validation exposure subgroups

```json
{
  "previously_exposed": {
    "n": 32,
    "policies": {
      "selected_reference": {
        "f1": 0.2299107142857143,
        "generated_tokens": 70.84375,
        "prompt_tokens": 4073.84375,
        "tool_calls": 1.875
      },
      "control": {
        "f1": 0.171875,
        "generated_tokens": 49.875,
        "prompt_tokens": 3312.90625,
        "tool_calls": 1.40625
      },
      "depth": {
        "f1": 0.171875,
        "generated_tokens": 66.625,
        "prompt_tokens": 3397.40625,
        "tool_calls": 1.5
      },
      "input_gate": {
        "f1": 0.171875,
        "generated_tokens": 68.8125,
        "prompt_tokens": 3466.3125,
        "tool_calls": 1.5625
      },
      "depth_input_gate": {
        "f1": 0.171875,
        "generated_tokens": 69.65625,
        "prompt_tokens": 3596.59375,
        "tool_calls": 1.59375
      },
      "observation": {
        "f1": 0.171875,
        "generated_tokens": 65.71875,
        "prompt_tokens": 3253.96875,
        "tool_calls": 1.46875
      },
      "question_attention": {
        "f1": 0.171875,
        "generated_tokens": 47.59375,
        "prompt_tokens": 2923.71875,
        "tool_calls": 1.25
      },
      "action_attention": {
        "f1": 0.171875,
        "generated_tokens": 64.375,
        "prompt_tokens": 3020.78125,
        "tool_calls": 1.1875
      },
      "action_set": {
        "f1": 0.171875,
        "generated_tokens": 49.03125,
        "prompt_tokens": 3182.625,
        "tool_calls": 1.375
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  },
  "new_validation": {
    "n": 32,
    "policies": {
      "selected_reference": {
        "f1": 0.27367424242424243,
        "generated_tokens": 82.65625,
        "prompt_tokens": 3667.8125,
        "tool_calls": 1.6875
      },
      "control": {
        "f1": 0.1590909090909091,
        "generated_tokens": 50.84375,
        "prompt_tokens": 2698.75,
        "tool_calls": 1.03125
      },
      "depth": {
        "f1": 0.1590909090909091,
        "generated_tokens": 50.59375,
        "prompt_tokens": 2708.21875,
        "tool_calls": 1.03125
      },
      "input_gate": {
        "f1": 0.1590909090909091,
        "generated_tokens": 49.0625,
        "prompt_tokens": 2651.9375,
        "tool_calls": 1.0
      },
      "depth_input_gate": {
        "f1": 0.1590909090909091,
        "generated_tokens": 49.625,
        "prompt_tokens": 2628.25,
        "tool_calls": 1.0
      },
      "observation": {
        "f1": 0.1590909090909091,
        "generated_tokens": 52.25,
        "prompt_tokens": 2695.90625,
        "tool_calls": 1.03125
      },
      "question_attention": {
        "f1": 0.19554924242424243,
        "generated_tokens": 55.375,
        "prompt_tokens": 2588.9375,
        "tool_calls": 0.96875
      },
      "action_attention": {
        "f1": 0.1825284090909091,
        "generated_tokens": 61.4375,
        "prompt_tokens": 3564.53125,
        "tool_calls": 1.34375
      },
      "action_set": {
        "f1": 0.1590909090909091,
        "generated_tokens": 48.09375,
        "prompt_tokens": 2571.96875,
        "tool_calls": 0.96875
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  }
}
```
