# jev_v1.1 — 1·2단계 실행 기록

공식 MMQA의 공개 표를 공통 배경으로 사용한 텍스트·이미지 continuation pilot이다. 최종 benchmark test나 독창성·비열등성 검증이 아니다.

- GPU: RTX PRO 6000 Blackwell Server Edition, 96GB.
- Reader: pinned Gemma 4 E4B-it; backbone 동결.
- 질문: train 300 / development 80. 최초 진단 40은 train의 사전 고정 부분집합.
- 공통 축소 corpus: 텍스트 503개·이미지 427개. 실제 이미지 벡터와 텍스트 벡터를 분리.
- 최초 실행에서 누락된 공개 표를 발견해 원결과를 보존하고 동일 40문항·gate로 한 번 재진단.
- 수정 전 EM 6/40·F1 23.3%는 공개 문맥이 누락된 진단이며 완전한 evidence oracle로 해석하지 않는다.

## 1A. 수정된 Reader 진단

| 조건 | EM | F1 | 평균 답변 호출 시간 |
|---|---:|---:|---:|
| 질문+공개 표 | 2.5% | 2.5% | 0.197s |
| 공개 표+정답 근거 | 25.0% | 34.9% | 0.666s |

기존 gate 판정: **True**. 기준은 40문항 중 완전 정답 10개 이상이며 F1 순개선 0.05 이상이다.

## 1B. 모듈 접근 개입

자연 검색의 첫 증거를 query generator와 final reader에 보이거나 숨기는 2×2를 수집한다. 공개 표·사용한 budget은 모든 조건에서 동일하다. 검색어·중간 답은 final reader에 전달하지 않는다.

현재 수집/효과 요약:

```json
{
  "questions": 244,
  "contrast_epsilon": 0.05,
  "contrast_questions": 40,
  "contrast_source_groups": 34,
  "contrast_qids": [
    "5ad7872fd6da869db8a9fb78c9916734",
    "b9f31517e1f7ce179b456674140fb3ed",
    "1cf74625e16b1bcac3324e7ae026180e",
    "ee58a7b4b335b0844e732617e344e639",
    "a3a03a31a4ccb8f4a00f8b41bab91c89",
    "1fca66d900d9929ad11efe8bb5d5daa8",
    "d092021375b8cf272c34104be9048668",
    "9e4bc710029629bada5f8fcedf77fe50",
    "5855d3de8df5a78678c6071b74f3c553",
    "df8b06a9d6cbc40df4e170e4bdbdc2fc",
    "c0148a0e0c409fe3b8f2787766704241",
    "ad1ae09552dcdae900fde952c37ccda5",
    "0dc30f37b5637d410a7ac688ff2918e1",
    "362ce7b438dfa84b2d47d553c91bd26a",
    "fc37e88a761d36479c8a3da34b9047df",
    "1d232c1121d8683692f7b445173a7be2",
    "c55f0e0d59c000f9515e530934f04c75",
    "ee5ebd0b0055e020865afdfdabaab115",
    "daf7c989ad86f8bfc40513d7d8f519f6",
    "e64da4a72a3dde841d84f3443e70fd7e",
    "02622f8c148b7bb164d6826e9451047c",
    "ecab01544d2dae415293d4d7974aca6e",
    "498f774e7540baf5dd3e492effe7a67a",
    "2342981906f3ded5c1c768a80e894e5f",
    "4b568561eebb86aadf0a43e2485b138e",
    "c19eff307190be119808fabac9ec7ace",
    "aa74b0299ab56284a75df76e1a172c45",
    "36d688448bb826c8b676e56664888ebb",
    "2a2066baa69a43a1deedccdcb839b52d",
    "9cce25ba51b340a500186877811b118d",
    "58dce17d28be1e507daa13ebfd58ff9b",
    "254e778f82565c6f335f5bc30d131604",
    "4008a3e0745543f02cb88264ab548caa",
    "f58e32d7f577a6fbeadacc35d2c92828",
    "affde65d031c89884b5e34fb418b73a1",
    "ecebe54ace2b6a88ce6460b3d10c605b",
    "1e8864a1422a0e14b1bd6549ca44e38e",
    "583171ac6bd3dc5e06923e38cf5e29b7",
    "9e8661a52337c3061a211f127ca0a4cc",
    "c4247336404a5a7a5752ea91873e012d"
  ],
  "strict_preference_change_questions": 3
}
```

## 해석의 한계

이미지 관측은 pixels+title package다. 시각 정보만의 필요성은 아직 분리하지 않았다. 공유 corpus와 공개 표로 인해 엄격한 unseen-source 평가가 아니며, 실제 train 노출을 별도로 감사한다. 동일 입력의 셀 재사용은 비용 측정값을 유지한 counterfactual 기록이며 실제 새 정책 평가는 답변을 다시 생성한다. 학습 완료와 가설 지지는 별개다.

[실행 재검토](../../../docs/JEV_V11_EXECUTION_REVIEW.md) · [고정 계획](../../registry/jev_v1.1/execution_plan.json) · [전체 집계](stage12_summary.json)

## GPU availability interruption

Two foreign `pm_native_env` GPU jobs triggered the exclusive-timing guard. Both last saved boundary records were archived and excluded pending recomputation; 244 valid training records remain. Original gates and labels are unchanged. Head acceleration is implemented and CPU-tested, but GPU equivalence and development evaluation have not run. See [acceleration plan](../../../docs/JEV_HEAD_ACCELERATION_PLAN.md) and [Astra review](../../../docs/JEV_HEAD_ACCELERATION_ASTRA_REVIEW.md).
