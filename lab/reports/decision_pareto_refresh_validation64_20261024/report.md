# Decision head 개선 결과

Split: validation, n=64, policies=2. 동일질문paired비교. 단일학습seed.

| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| quality_start | 32.45 | 139.03 | 7671.8 | 3.406 | 2 |
| pareto | 28.17 | 104.80 | 5498.3 | 2.531 | 1 |

## 지정 baseline 대비

- pareto: F1 delta -4.28pp, CI [-0.10396634615384615, 0.014288250811688316]; 생성ratio {'ratio': 0.7537648909867386, 'ci95': [0.6301513564373993, 0.8756832144644632]}; point목표 False, 방향CI False, 목표CI False.

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
      "pareto": {
        "f1": 0.30625,
        "generated_tokens": 98.28125,
        "prompt_tokens": 4294.65625,
        "tool_calls": 2.25
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
      "pareto": {
        "f1": 0.2571614583333333,
        "generated_tokens": 111.3125,
        "prompt_tokens": 6701.9375,
        "tool_calls": 2.8125
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  }
}
```
