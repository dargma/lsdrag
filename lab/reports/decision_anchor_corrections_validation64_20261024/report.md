# Decision head 개선 결과

Split: validation, n=64, policies=3. 동일질문paired비교. 단일학습seed.

| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| selected_reference | 25.18 | 76.75 | 3870.8 | 1.781 | 0 |
| mlp | 26.41 | 81.55 | 4038.6 | 1.938 | 0 |
| question_attention | 25.18 | 78.02 | 3839.2 | 1.750 | 0 |

## 지정 baseline 대비

- mlp: F1 delta +1.23pp, CI [-0.015625, 0.05133928571428571]; 생성ratio {'ratio': 1.0625, 'ci95': [0.9753626759050171, 1.185321644753552]}; point목표 False, 방향CI False, 목표CI False.
- question_attention: F1 delta +0.00pp, CI [0.0, 0.0]; 생성ratio {'ratio': 1.0164902280130292, 'ci95': [0.9671780006881596, 1.0703911160116926]}; point목표 False, 방향CI False, 목표CI False.

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
      "mlp": {
        "f1": 0.2611607142857143,
        "generated_tokens": 71.0625,
        "prompt_tokens": 3959.9375,
        "tool_calls": 2.0
      },
      "question_attention": {
        "f1": 0.2299107142857143,
        "generated_tokens": 75.5625,
        "prompt_tokens": 4084.46875,
        "tool_calls": 1.875
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
      "mlp": {
        "f1": 0.26697781385281383,
        "generated_tokens": 92.03125,
        "prompt_tokens": 4117.28125,
        "tool_calls": 1.875
      },
      "question_attention": {
        "f1": 0.27367424242424243,
        "generated_tokens": 80.46875,
        "prompt_tokens": 3593.84375,
        "tool_calls": 1.625
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  }
}
```
