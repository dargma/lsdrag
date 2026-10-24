# Decision head 개선 결과

Split: validation, n=64, policies=5. 동일질문paired비교. 단일학습seed.

| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| mlp_control | 29.14 | 114.69 | 6255.5 | 2.859 | 1 |
| mlp_stop | 29.09 | 114.80 | 6377.4 | 2.922 | 1 |
| mlp_stop_value | 32.45 | 139.03 | 7671.8 | 3.406 | 2 |
| attention_stop | 30.21 | 136.92 | 10263.4 | 3.781 | 1 |
| attention_stop_value | 30.21 | 143.78 | 10964.3 | 3.938 | 1 |

## 지정 baseline 대비

- mlp_stop: F1 delta -0.06pp, CI [-0.046875, 0.04631696428571429]; 생성ratio {'ratio': 1.0009536784741144, 'ci95': [0.9057018377215125, 1.125529193931847]}; point목표 False, 방향CI False, 목표CI False.
- mlp_stop_value: F1 delta +3.31pp, CI [-0.024939903846153848, 0.09710679945054933]; 생성ratio {'ratio': 1.2122615803814714, 'ci95': [1.0263925309608013, 1.4725487959237853]}; point목표 False, 방향CI False, 목표CI False.
- attention_stop: F1 delta +1.07pp, CI [-0.03174115597943723, 0.0636532738095238]; 생성ratio {'ratio': 1.1938692098092643, 'ci95': [1.0382773188928593, 1.3729152562668554]}; point목표 False, 방향CI False, 목표CI False.
- attention_stop_value: F1 delta +1.07pp, CI [-0.03174115597943723, 0.0636532738095238]; 생성ratio {'ratio': 1.2536784741144413, 'ci95': [1.0929701175691087, 1.4476396439742374]}; point목표 False, 방향CI False, 목표CI False.

F1+5%p 및생성token−30% 동시목표. 확률은action확률이며정답확률아님. 단일seed/조건부CI/공식train원천의학습/검증가족분리holdout이며공통101962문단검색환경. validation은후보선택자료이므로최종확증아님. 최종test결과로후보재선택금지. 전체학습/수집비용별도보고. 통제latency비교없음.

CI는개별지표의marginal95%구간이며다중비교보정없음. 두CI조건통과를동시95%보장으로해석하지않는다.

## Validation exposure subgroups

```json
{
  "previously_exposed": {
    "n": 32,
    "policies": {
      "mlp_control": {
        "f1": 0.2924107142857143,
        "generated_tokens": 104.53125,
        "prompt_tokens": 5589.0,
        "tool_calls": 2.78125
      },
      "mlp_stop": {
        "f1": 0.2611607142857143,
        "generated_tokens": 105.28125,
        "prompt_tokens": 5841.1875,
        "tool_calls": 2.8125
      },
      "mlp_stop_value": {
        "f1": 0.3284684065934066,
        "generated_tokens": 129.90625,
        "prompt_tokens": 7487.90625,
        "tool_calls": 3.46875
      },
      "attention_stop": {
        "f1": 0.321875,
        "generated_tokens": 136.75,
        "prompt_tokens": 9110.375,
        "tool_calls": 3.6875
      },
      "attention_stop_value": {
        "f1": 0.321875,
        "generated_tokens": 140.34375,
        "prompt_tokens": 9858.65625,
        "tool_calls": 3.84375
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  },
  "new_validation": {
    "n": 32,
    "policies": {
      "mlp_control": {
        "f1": 0.29041531385281383,
        "generated_tokens": 124.84375,
        "prompt_tokens": 6921.9375,
        "tool_calls": 2.9375
      },
      "mlp_stop": {
        "f1": 0.32054924242424243,
        "generated_tokens": 124.3125,
        "prompt_tokens": 6913.59375,
        "tool_calls": 3.03125
      },
      "mlp_stop_value": {
        "f1": 0.32054924242424243,
        "generated_tokens": 148.15625,
        "prompt_tokens": 7855.78125,
        "tool_calls": 3.34375
      },
      "attention_stop": {
        "f1": 0.28229166666666666,
        "generated_tokens": 137.09375,
        "prompt_tokens": 11416.4375,
        "tool_calls": 3.875
      },
      "attention_stop_value": {
        "f1": 0.28229166666666666,
        "generated_tokens": 147.21875,
        "prompt_tokens": 12070.0,
        "tool_calls": 4.03125
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  }
}
```
