# Decision head 개선 결과

Split: validation, n=16, policies=3. 동일질문paired비교. 단일학습seed.

| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| semantic_once | 26.32 | 56.81 | 2246.2 | 1.000 | 0 |
| grpo_transfer | 25.00 | 101.81 | 4258.1 | 2.125 | 0 |
| native_fast | 31.25 | 104.31 | 4860.6 | 2.562 | 0 |

## 지정 baseline 대비

- semantic_once: F1 delta +1.32pp, CI [-0.14802631578947367, 0.1875]; 생성ratio {'ratio': 0.5580110497237569, 'ci95': [0.43348491322193616, 0.7114144668872127]}; point목표 False, 방향CI False, 목표CI False.
- native_fast: F1 delta +6.25pp, CI [0.0, 0.1875]; 생성ratio {'ratio': 1.0245549416820134, 'ci95': [0.7939034661345601, 1.3056502676403186]}; point목표 False, 방향CI False, 목표CI False.

F1+5%p 및생성token−30% 동시목표. 확률은action확률이며정답확률아님. 단일seed/조건부CI/공식train원천의학습/검증가족분리holdout이며공통101962문단검색환경. validation은후보선택자료이므로최종확증아님. 최종test결과로후보재선택금지. 전체학습/수집비용별도보고. 통제latency비교없음.

## Validation exposure subgroups

{
  "previously_exposed": {
    "n": 16,
    "policies": {
      "semantic_once": {
        "f1": 0.2631578947368421,
        "generated_tokens": 56.8125,
        "prompt_tokens": 2246.1875,
        "tool_calls": 1.0
      },
      "grpo_transfer": {
        "f1": 0.25,
        "generated_tokens": 101.8125,
        "prompt_tokens": 4258.125,
        "tool_calls": 2.125
      },
      "native_fast": {
        "f1": 0.3125,
        "generated_tokens": 104.3125,
        "prompt_tokens": 4860.5625,
        "tool_calls": 2.5625
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  }
}
