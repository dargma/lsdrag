# Decision head 개선 결과

Split: validation, n=64, policies=6. 동일질문paired비교. 단일학습seed.

| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| selected_reference | 25.18 | 76.75 | 3870.8 | 1.781 | 0 |
| old_mlp | 15.96 | 61.41 | 3862.9 | 1.719 | 1 |
| old_question_attention | 21.29 | 61.03 | 3154.7 | 1.281 | 0 |
| fresh_mlp | 23.36 | 77.66 | 4603.5 | 2.016 | 0 |
| fresh_question_attention | 21.44 | 64.86 | 3726.9 | 1.781 | 0 |
| fresh_direct_utility | 12.02 | 75.64 | 4541.5 | 1.828 | 2 |

## 지정 baseline 대비

- old_mlp: F1 delta -9.22pp, CI [-0.18058331642316017, -0.011952533143939435]; 생성ratio {'ratio': 0.8000814332247557, 'ci95': [0.5781693808655413, 1.098360581583367]}; point목표 False, 방향CI False, 목표CI False.
- old_question_attention: F1 delta -3.89pp, CI [-0.11256256764069264, 0.03125]; 생성ratio {'ratio': 0.7951954397394136, 'ci95': [0.6888343327023146, 0.8922042381051691]}; point목표 False, 방향CI False, 목표CI False.
- fresh_mlp: F1 delta -1.82pp, CI [-0.0625, 0.015625]; 생성ratio {'ratio': 1.0118078175895766, 'ci95': [0.8637622130241722, 1.1759085053482137]}; point목표 False, 방향CI False, 목표CI False.
- fresh_question_attention: F1 delta -3.74pp, CI [-0.09895833333333333, 0.014732142857142859]; 생성ratio {'ratio': 0.8450732899022801, 'ci95': [0.6876392969341296, 0.9992181836435925]}; point목표 False, 방향CI False, 목표CI False.
- fresh_direct_utility: F1 delta -13.16pp, CI [-0.22277462121212122, -0.047105823863636445]; 생성ratio {'ratio': 0.9855456026058632, 'ci95': [0.642797914577528, 1.4234169869934736]}; point목표 False, 방향CI False, 목표CI False.

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
      "old_mlp": {
        "f1": 0.18125,
        "generated_tokens": 74.78125,
        "prompt_tokens": 4178.25,
        "tool_calls": 1.96875
      },
      "old_question_attention": {
        "f1": 0.234375,
        "generated_tokens": 54.25,
        "prompt_tokens": 3178.90625,
        "tool_calls": 1.28125
      },
      "fresh_mlp": {
        "f1": 0.1986607142857143,
        "generated_tokens": 71.0625,
        "prompt_tokens": 4897.28125,
        "tool_calls": 2.125
      },
      "fresh_question_attention": {
        "f1": 0.196875,
        "generated_tokens": 61.25,
        "prompt_tokens": 3519.1875,
        "tool_calls": 1.8125
      },
      "fresh_direct_utility": {
        "f1": 0.11011904761904762,
        "generated_tokens": 68.3125,
        "prompt_tokens": 5221.59375,
        "tool_calls": 2.0625
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
      "old_mlp": {
        "f1": 0.13802083333333331,
        "generated_tokens": 48.03125,
        "prompt_tokens": 3547.5625,
        "tool_calls": 1.46875
      },
      "old_question_attention": {
        "f1": 0.19140625,
        "generated_tokens": 67.8125,
        "prompt_tokens": 3130.5,
        "tool_calls": 1.28125
      },
      "fresh_mlp": {
        "f1": 0.26846590909090906,
        "generated_tokens": 84.25,
        "prompt_tokens": 4309.6875,
        "tool_calls": 1.90625
      },
      "fresh_question_attention": {
        "f1": 0.23200757575757575,
        "generated_tokens": 68.46875,
        "prompt_tokens": 3934.5625,
        "tool_calls": 1.75
      },
      "fresh_direct_utility": {
        "f1": 0.13020833333333331,
        "generated_tokens": 82.96875,
        "prompt_tokens": 3861.40625,
        "tool_calls": 1.59375
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  }
}
```
