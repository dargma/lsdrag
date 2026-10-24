# Decision head 개선 결과

Split: validation, n=64, policies=3. 동일질문paired비교. 단일학습seed.

| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| quality_start | 32.45 | 139.03 | 7671.8 | 3.406 | 2 |
| budget70 | 24.46 | 79.31 | 5825.4 | 2.203 | 0 |
| budget55 | 3.39 | 9.47 | 805.0 | 0.000 | 0 |

## 지정 baseline 대비

- budget70: F1 delta -7.99pp, CI [-0.16039481351981352, -0.0034819347319347315]; 생성ratio {'ratio': 0.5704652730950776, 'ci95': [0.4108772539785054, 0.7513997527757348]}; point목표 False, 방향CI False, 목표CI False.
- budget55: F1 delta -29.07pp, CI [-0.3953946444180819, -0.19027586996336995]; 생성ratio {'ratio': 0.06810519217801753, 'ci95': [0.05487584216996553, 0.08423325820392223]}; point목표 False, 방향CI False, 목표CI False.

F1+5%p 및생성token−30% 동시목표. 확률은action확률이며정답확률아님. 단일seed/조건부CI/공식train원천의학습/검증가족분리holdout이며공통101962문단검색환경. validation은후보선택자료이므로최종확증아님. 최종test결과로후보재선택금지. 전체학습/수집비용별도보고. 통제latency비교없음.

CI는개별지표의marginal95%구간이며다중비교보정없음. 두CI조건통과를동시95%보장으로해석하지않는다.

## Validation exposure subgroups

```json
{
  "previously_exposed": {
    "n": 32,
    "policies": {
      "quality_start": {
        "f1": 0.3284684065934066,
        "generated_tokens": 129.90625,
        "prompt_tokens": 7487.90625,
        "tool_calls": 3.46875
      },
      "budget70": {
        "f1": 0.2299107142857143,
        "generated_tokens": 60.28125,
        "prompt_tokens": 4032.03125,
        "tool_calls": 1.625
      },
      "budget55": {
        "f1": 0.03125,
        "generated_tokens": 9.28125,
        "prompt_tokens": 804.84375,
        "tool_calls": 0.0
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  },
  "new_validation": {
    "n": 32,
    "policies": {
      "quality_start": {
        "f1": 0.32054924242424243,
        "generated_tokens": 148.15625,
        "prompt_tokens": 7855.78125,
        "tool_calls": 3.34375
      },
      "budget70": {
        "f1": 0.259375,
        "generated_tokens": 98.34375,
        "prompt_tokens": 7618.75,
        "tool_calls": 2.78125
      },
      "budget55": {
        "f1": 0.03645833333333333,
        "generated_tokens": 9.65625,
        "prompt_tokens": 805.0625,
        "tool_calls": 0.0
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  }
}
```
