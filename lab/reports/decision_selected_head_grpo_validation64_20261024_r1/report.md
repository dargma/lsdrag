# Decision head 개선 결과

Split: validation, n=64, policies=2. 동일질문paired비교. 단일학습seed.

| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| multi_mlp_outcome | 25.18 | 76.75 | 3870.8 | 1.781 | 0 |
| multi_mlp_outcome_grpo | 22.95 | 53.19 | 3973.9 | 1.422 | 0 |

## 지정 baseline 대비

- multi_mlp_outcome_grpo: F1 delta -2.23pp, CI [-0.08333333333333333, 0.03243371212121212]; 생성ratio {'ratio': 0.6929967426710097, 'ci95': [0.5172108132707117, 0.8976775892147342]}; point목표 False, 방향CI False, 목표CI False.

F1+5%p 및생성token−30% 동시목표. 확률은action확률이며정답확률아님. 단일seed/조건부CI/공식train원천의학습/검증가족분리holdout이며공통101962문단검색환경. validation은후보선택자료이므로최종확증아님. 최종test결과로후보재선택금지. 전체학습/수집비용별도보고. 통제latency비교없음.

CI는개별지표의marginal95%구간이며다중비교보정없음. 두CI조건통과를동시95%보장으로해석하지않는다.

## Validation exposure subgroups

```json
{
  "previously_exposed": {
    "n": 32,
    "policies": {
      "multi_mlp_outcome": {
        "f1": 0.2299107142857143,
        "generated_tokens": 70.84375,
        "prompt_tokens": 4073.84375,
        "tool_calls": 1.875
      },
      "multi_mlp_outcome_grpo": {
        "f1": 0.2299107142857143,
        "generated_tokens": 50.71875,
        "prompt_tokens": 3752.6875,
        "tool_calls": 1.28125
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  },
  "new_validation": {
    "n": 32,
    "policies": {
      "multi_mlp_outcome": {
        "f1": 0.27367424242424243,
        "generated_tokens": 82.65625,
        "prompt_tokens": 3667.8125,
        "tool_calls": 1.6875
      },
      "multi_mlp_outcome_grpo": {
        "f1": 0.22916666666666663,
        "generated_tokens": 55.65625,
        "prompt_tokens": 4195.125,
        "tool_calls": 1.5625
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  }
}
```
