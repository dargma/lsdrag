# Decision head 개선 결과

Split: validation, n=64, policies=2. 동일질문paired비교. 단일학습seed.

| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| token_total | 35.21 | 142.34 | 9967.7 | 4.109 | 1 |
| token_remaining | 32.92 | 127.12 | 8749.2 | 3.766 | 1 |

## 지정 baseline 대비

- token_remaining: F1 delta -2.29pp, CI [-0.07754788614163614, 0.02731990231990232]; 생성ratio {'ratio': 0.8930845225027443, 'ci95': [0.8164182762388877, 0.9591053721049261]}; point목표 False, 방향CI False, 목표CI False.

F1+5%p 및생성token−30% 동시목표. 확률은action확률이며정답확률아님. 단일seed/조건부CI/공식train원천의학습/검증가족분리holdout이며공통101962문단검색환경. validation은후보선택자료이므로최종확증아님. 최종test결과로후보재선택금지. 전체학습/수집비용별도보고. 통제latency비교없음.

CI는개별지표의marginal95%구간이며다중비교보정없음. 두CI조건통과를동시95%보장으로해석하지않는다.

## Validation exposure subgroups

```json
{
  "previously_exposed": {
    "n": 32,
    "policies": {
      "token_total": {
        "f1": 0.3248626373626374,
        "generated_tokens": 129.4375,
        "prompt_tokens": 8090.9375,
        "tool_calls": 3.625
      },
      "token_remaining": {
        "f1": 0.3236607142857143,
        "generated_tokens": 121.625,
        "prompt_tokens": 7779.09375,
        "tool_calls": 3.5
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  },
  "new_validation": {
    "n": 32,
    "policies": {
      "token_total": {
        "f1": 0.3793402777777778,
        "generated_tokens": 155.25,
        "prompt_tokens": 11844.40625,
        "tool_calls": 4.59375
      },
      "token_remaining": {
        "f1": 0.33482142857142855,
        "generated_tokens": 132.625,
        "prompt_tokens": 9719.3125,
        "tool_calls": 4.03125
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  }
}
```
