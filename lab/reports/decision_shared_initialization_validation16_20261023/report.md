# Decision head 개선 결과

Split: validation, n=16, policies=1. 동일질문paired비교. 단일학습seed.

| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| shared_bc | 25.00 | 100.38 | 4772.8 | 2.500 | 0 |

## 지정 baseline 대비


F1+5%p 및생성token−30% 동시목표. 확률은action확률이며정답확률아님. 단일seed/조건부CI/공식train원천의학습/검증가족분리holdout이며공통101962문단검색환경. validation은후보선택자료이므로최종확증아님. 최종test결과로후보재선택금지. 전체학습/수집비용별도보고. 통제latency비교없음.

## Validation exposure subgroups

```json
{
  "previously_exposed": {
    "n": 16,
    "policies": {
      "shared_bc": {
        "f1": 0.25,
        "generated_tokens": 100.375,
        "prompt_tokens": 4772.8125,
        "tool_calls": 2.5
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  }
}
```

shared_bc는공통코퍼스teacher192에서새로학습한초기head이며구코퍼스전이head가아니다. 원summary의전이실험설명문구는복제된evaluator의잔여문구로이조건에적용되지않음. raw수치보존,이후evaluator문구수정. 이검증은모방초기화성능이며RL효과가아님.
