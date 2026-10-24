# Decision head 개선 결과

Split: validation, n=64, policies=2. 동일질문paired비교. 단일학습seed.

| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| selected_reference | 27.32 | 66.89 | 3417.9 | 1.422 | 0 |
| policy_iteration1 | 21.23 | 61.44 | 3535.8 | 1.438 | 0 |

## 지정 baseline 대비

- policy_iteration1: F1 delta -6.09pp, CI [-0.12127232142857143, -0.014027267156862762]; 생성ratio {'ratio': 0.9184769913571595, 'ci95': [0.7607001058044831, 1.0808685022226332]}; point목표 False, 방향CI False, 목표CI False.

F1+5%p 및생성token−30% 동시목표. 확률은action확률이며정답확률아님. 단일seed/조건부CI/공식train원천의학습/검증가족분리holdout이며공통101962문단검색환경. validation은후보선택자료이므로최종확증아님. 최종test결과로후보재선택금지. 전체학습/수집비용별도보고. 통제latency비교없음.

CI는개별지표의marginal95%구간이며다중비교보정없음. 두CI조건통과를동시95%보장으로해석하지않는다.

## Validation exposure subgroups

```json
{
  "previously_exposed": {
    "n": 32,
    "policies": {
      "selected_reference": {
        "f1": 0.3236607142857143,
        "generated_tokens": 54.6875,
        "prompt_tokens": 3038.8125,
        "tool_calls": 1.28125
      },
      "policy_iteration1": {
        "f1": 0.259375,
        "generated_tokens": 57.78125,
        "prompt_tokens": 3428.96875,
        "tool_calls": 1.375
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  },
  "new_validation": {
    "n": 32,
    "policies": {
      "selected_reference": {
        "f1": 0.22265625,
        "generated_tokens": 79.09375,
        "prompt_tokens": 3797.03125,
        "tool_calls": 1.5625
      },
      "policy_iteration1": {
        "f1": 0.16513480392156862,
        "generated_tokens": 65.09375,
        "prompt_tokens": 3642.53125,
        "tool_calls": 1.5
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  }
}
```
