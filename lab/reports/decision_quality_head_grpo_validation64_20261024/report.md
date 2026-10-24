# Decision head 개선 결과

Split: validation, n=64, policies=2. 동일질문paired비교. 단일학습seed.

| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| quality_start | 32.45 | 139.03 | 7671.8 | 3.406 | 2 |
| quality_grpo | 34.75 | 150.91 | 9298.4 | 3.703 | 2 |

## 지정 baseline 대비

- quality_grpo: F1 delta +2.29pp, CI [-0.030940934065934064, 0.07843406593406593]; 생성ratio {'ratio': 1.0854124522364577, 'ci95': [0.9134095682473379, 1.29526427407266]}; point목표 False, 방향CI False, 목표CI False.

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
      "quality_grpo": {
        "f1": 0.3407051282051282,
        "generated_tokens": 134.71875,
        "prompt_tokens": 9021.6875,
        "tool_calls": 3.65625
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
      "quality_grpo": {
        "f1": 0.3542030885780886,
        "generated_tokens": 167.09375,
        "prompt_tokens": 9575.09375,
        "tool_calls": 3.75
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  }
}
```
