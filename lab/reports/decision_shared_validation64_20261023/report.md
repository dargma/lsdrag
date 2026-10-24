# Decision head 개선 결과

Split: validation, n=64, policies=6. 동일질문paired비교. 단일학습seed.

| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| semantic_once | 24.87 | 55.91 | 2220.4 | 1.000 | 0 |
| native_fast | 24.70 | 97.44 | 4691.0 | 2.375 | 0 |
| shared_bc | 22.50 | 101.59 | 4748.8 | 2.391 | 0 |
| policy_iteration_cost003 | 29.46 | 123.25 | 7250.5 | 3.297 | 0 |
| policy_iteration_cost006 | 26.08 | 112.67 | 6112.1 | 2.844 | 0 |
| shared_grpo | 22.50 | 98.41 | 4625.3 | 2.328 | 0 |

## 지정 baseline 대비

- semantic_once: F1 delta +2.37pp, CI [-0.05959779663584931, 0.11123716070234911]; 생성ratio {'ratio': 0.5681168624960304, 'ci95': [0.49907650099772916, 0.6433519132447691]}; point목표 False, 방향CI False, 목표CI False.
- native_fast: F1 delta +2.20pp, CI [-0.002840909090909091, 0.061079545454545456]; 생성ratio {'ratio': 0.9901556049539536, 'ci95': [0.8835472107754893, 1.0891712024437656]}; point목표 False, 방향CI False, 목표CI False.
- shared_bc: F1 delta +0.00pp, CI [0.0, 0.0]; 생성ratio {'ratio': 1.0323912353127978, 'ci95': [1.0015889360045824, 1.0711843636584821]}; point목표 False, 방향CI False, 목표CI False.
- policy_iteration_cost003: F1 delta +6.96pp, CI [0.0, 0.14591196563852796]; 생성ratio {'ratio': 1.2524610987615117, 'ci95': [1.1173054927318196, 1.416133742064805]}; point목표 False, 방향CI False, 목표CI False.
- policy_iteration_cost006: F1 delta +3.58pp, CI [-0.023369859307359304, 0.10349025974025974]; 생성ratio {'ratio': 1.1449666560812957, 'ci95': [1.0069679542800192, 1.2918501680770158]}; point목표 False, 방향CI False, 목표CI False.

F1+5%p 및생성token−30% 동시목표. 확률은action확률이며정답확률아님. 단일seed/조건부CI/공식train원천의학습/검증가족분리holdout이며공통101962문단검색환경. validation은후보선택자료이므로최종확증아님. 최종test결과로후보재선택금지. 전체학습/수집비용별도보고. 통제latency비교없음.

## Native and fixed-workflow comparisons

|Reference|Policy|F1 delta (pp)|Marginal95% CI (pp)|Generated ratio|Marginal95% CI|
|---|---|---:|---|---:|---|
|native_fast|semantic_once|+0.17|[-8.97, 9.57]|0.5738|[0.49135783427521634, 0.6704755249420954]|
|native_fast|shared_bc|-2.20|[-6.11, 0.28]|1.0427|[0.9524100972601379, 1.162899247368429]|
|native_fast|policy_iteration_cost003|+4.76|[-3.39, 13.28]|1.2649|[1.1003807209289673, 1.470394947087954]|
|native_fast|policy_iteration_cost006|+1.38|[-5.39, 8.59]|1.1564|[0.9988429290863828, 1.3418093516586094]|
|native_fast|shared_grpo|-2.20|[-6.11, 0.28]|1.0099|[0.918129305873122, 1.1318014340492382]|
|semantic_once|native_fast|-0.17|[-9.57, 8.97]|1.7429|[1.4914787534351575, 2.0351766693260185]|
|semantic_once|shared_bc|-2.37|[-11.12, 5.96]|1.8172|[1.6056665710305185, 2.0631379106601226]|
|semantic_once|policy_iteration_cost003|+4.59|[-4.94, 14.05]|2.2046|[1.8941106126626204, 2.568803738417881]|
|semantic_once|policy_iteration_cost006|+1.20|[-6.69, 8.94]|2.0154|[1.7906512208858842, 2.274785524767841]|
|semantic_once|shared_grpo|-2.37|[-11.12, 5.96]|1.7602|[1.5543592543586964, 2.0037008314390072]|

CI는개별지표의marginal95%구간이며다중비교보정없음. 두CI조건통과를동시95%보장으로해석하지않는다.

## Validation exposure subgroups

```json
{
  "previously_exposed": {
    "n": 32,
    "policies": {
      "semantic_once": {
        "f1": 0.25032894736842104,
        "generated_tokens": 52.71875,
        "prompt_tokens": 2235.25,
        "tool_calls": 1.0
      },
      "native_fast": {
        "f1": 0.2611607142857143,
        "generated_tokens": 96.09375,
        "prompt_tokens": 4634.71875,
        "tool_calls": 2.40625
      },
      "shared_bc": {
        "f1": 0.2299107142857143,
        "generated_tokens": 96.59375,
        "prompt_tokens": 4664.5,
        "tool_calls": 2.40625
      },
      "policy_iteration_cost003": {
        "f1": 0.2924107142857143,
        "generated_tokens": 113.09375,
        "prompt_tokens": 6828.71875,
        "tool_calls": 3.03125
      },
      "policy_iteration_cost006": {
        "f1": 0.2924107142857143,
        "generated_tokens": 98.1875,
        "prompt_tokens": 5352.75,
        "tool_calls": 2.5625
      },
      "shared_grpo": {
        "f1": 0.2299107142857143,
        "generated_tokens": 93.375,
        "prompt_tokens": 4489.375,
        "tool_calls": 2.3125
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  },
  "new_validation": {
    "n": 32,
    "policies": {
      "semantic_once": {
        "f1": 0.24716605392156862,
        "generated_tokens": 59.09375,
        "prompt_tokens": 2205.59375,
        "tool_calls": 1.0
      },
      "native_fast": {
        "f1": 0.23288690476190477,
        "generated_tokens": 98.78125,
        "prompt_tokens": 4747.34375,
        "tool_calls": 2.34375
      },
      "shared_bc": {
        "f1": 0.22010281385281386,
        "generated_tokens": 106.59375,
        "prompt_tokens": 4833.1875,
        "tool_calls": 2.375
      },
      "policy_iteration_cost003": {
        "f1": 0.296875,
        "generated_tokens": 133.40625,
        "prompt_tokens": 7672.1875,
        "tool_calls": 3.5625
      },
      "policy_iteration_cost006": {
        "f1": 0.22916666666666666,
        "generated_tokens": 127.15625,
        "prompt_tokens": 6871.40625,
        "tool_calls": 3.125
      },
      "shared_grpo": {
        "f1": 0.22010281385281386,
        "generated_tokens": 103.4375,
        "prompt_tokens": 4761.25,
        "tool_calls": 2.34375
      }
    },
    "note": "Descriptive subgroup means, no candidate selection on final test."
  }
}
```
