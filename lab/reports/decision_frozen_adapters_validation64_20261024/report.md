# Decision head 개선 결과

Split: validation, n=64, policies=4. 동일질문paired비교. 단일학습seed.

| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| selected_reference | 25.18 | 76.75 | 3870.8 | 1.781 | 0 |
| depth_frozen | 19.93 | 55.44 | 3368.0 | 1.406 | 0 |
| question_attention_frozen | 22.28 | 55.50 | 3050.1 | 1.250 | 0 |
| action_attention_frozen | 20.06 | 76.08 | 4047.5 | 1.672 | 0 |

## 지정 baseline 대비

- depth_frozen: F1 delta -5.25pp, CI [-0.13578869047619047, 0.023437500000000003]; 생성ratio {'ratio': 0.7223127035830619, 'ci95': [0.5723903209484551, 0.8638121912468966]}; point목표 False, 방향CI False, 목표CI False.
- question_attention_frozen: F1 delta -2.90pp, CI [-0.0896670386904762, 0.028273809523809527]; 생성ratio {'ratio': 0.7231270358306189, 'ci95': [0.58301065530573, 0.8556349983350038]}; point목표 False, 방향CI False, 목표CI False.
- action_attention_frozen: F1 delta -5.12pp, CI [-0.12704613095238096, 0.017857142857142856]; 생성ratio {'ratio': 0.9912459283387622, 'ci95': [0.8499906649667079, 1.1387466153312362]}; point목표 False, 방향CI False, 목표CI False.

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
      "depth_frozen": {
        "f1": 0.203125,
        "generated_tokens": 48.90625,
        "prompt_tokens": 3422.75,
        "tool_calls": 1.4375
      },
      "question_attention_frozen": {
        "f1": 0.234375,
        "generated_tokens": 46.9375,
        "prompt_tokens": 2808.375,
        "tool_calls": 1.125
      },
      "action_attention_frozen": {
        "f1": 0.171875,
        "generated_tokens": 66.5625,
        "prompt_tokens": 4157.34375,
        "tool_calls": 1.65625
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
      "depth_frozen": {
        "f1": 0.19554924242424243,
        "generated_tokens": 61.96875,
        "prompt_tokens": 3313.1875,
        "tool_calls": 1.375
      },
      "question_attention_frozen": {
        "f1": 0.21117424242424243,
        "generated_tokens": 64.0625,
        "prompt_tokens": 3291.8125,
        "tool_calls": 1.375
      },
      "action_attention_frozen": {
        "f1": 0.2294034090909091,
        "generated_tokens": 85.59375,
        "prompt_tokens": 3937.75,
        "tool_calls": 1.6875
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  }
}
```
