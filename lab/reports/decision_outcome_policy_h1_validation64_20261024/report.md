# Decision head 개선 결과

Split: validation, n=64, policies=2. 동일질문paired비교. 단일학습seed.

| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| selected_reference | 26.74 | 75.45 | 3676.2 | 1.688 | 0 |
| outcome_policy | 28.30 | 75.34 | 3670.0 | 1.641 | 0 |

## 지정 baseline 대비

- outcome_policy: F1 delta +1.56pp, CI [0.0, 0.046875]; 생성ratio {'ratio': 0.9985504245185338, 'ci95': [0.9249839800922323, 1.10212031024531]}; point목표 False, 방향CI False, 목표CI False.

F1+5%p 및생성token−30% 동시목표. 확률은action확률이며정답확률아님. 단일seed/조건부CI/공식train원천의학습/검증가족분리holdout이며공통101962문단검색환경. validation은후보선택자료이므로최종확증아님. 최종test결과로후보재선택금지. 전체학습/수집비용별도보고. 통제latency비교없음.

CI는개별지표의marginal95%구간이며다중비교보정없음. 두CI조건통과를동시95%보장으로해석하지않는다.

## Validation exposure subgroups

```json
{
  "previously_exposed": {
    "n": 32,
    "policies": {
      "selected_reference": {
        "f1": 0.2611607142857143,
        "generated_tokens": 69.9375,
        "prompt_tokens": 3488.71875,
        "tool_calls": 1.65625
      },
      "outcome_policy": {
        "f1": 0.2924107142857143,
        "generated_tokens": 64.1875,
        "prompt_tokens": 3243.46875,
        "tool_calls": 1.46875
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  },
  "new_validation": {
    "n": 32,
    "policies": {
      "selected_reference": {
        "f1": 0.27367424242424243,
        "generated_tokens": 80.96875,
        "prompt_tokens": 3863.625,
        "tool_calls": 1.71875
      },
      "outcome_policy": {
        "f1": 0.27367424242424243,
        "generated_tokens": 86.5,
        "prompt_tokens": 4096.59375,
        "tool_calls": 1.8125
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  }
}
```
