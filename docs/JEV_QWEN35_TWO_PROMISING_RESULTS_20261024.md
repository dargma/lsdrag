# 9B 두 후보 결과

동일 개발64문항. base/H3/constant는 동일 프로토콜의 검증된 기존 실행을 재사용했다.

| 정책 | F1(%) | EM/64 | 평균 생성토큰 | 정답 평균 | 오답 평균 | 오류 |
|---|---:|---:|---:|---:|---:|---:|
| base | 47.90 | 24 | 156.77 | 145.21 | 163.70 | 0 |
| h3 | 48.42 | 25 | 151.53 | 148.64 | 153.38 | 0 |
| constant | 48.68 | 25 | 154.75 | 143.36 | 162.05 | 0 |
| h1_conservative | 47.67 | 25 | 160.53 | 152.48 | 165.69 | 0 |
| value_aux | 49.46 | 25 | 158.02 | 144.92 | 166.41 | 0 |

| 후보 vs 기준 | 정오 분류 | 문항 | 기준 토큰 | 후보 토큰 | 절약 |
|---|---|---:|---:|---:|---:|
| h1_conservative_vs_base | both_correct | 23 | 3305 | 3495 | -190 |
| h1_conservative_vs_base | gained_correct | 2 | 321 | 317 | 4 |
| h1_conservative_vs_base | lost_correct | 1 | 180 | 281 | -101 |
| h1_conservative_vs_base | both_incorrect | 38 | 6227 | 6181 | 46 |
| h1_conservative_vs_h3 | both_correct | 22 | 3257 | 3366 | -109 |
| h1_conservative_vs_h3 | gained_correct | 3 | 388 | 446 | -58 |
| h1_conservative_vs_h3 | lost_correct | 3 | 459 | 610 | -151 |
| h1_conservative_vs_h3 | both_incorrect | 36 | 5594 | 5852 | -258 |
| h1_conservative_vs_constant | both_correct | 23 | 3305 | 3495 | -190 |
| h1_conservative_vs_constant | gained_correct | 2 | 321 | 317 | 4 |
| h1_conservative_vs_constant | lost_correct | 2 | 279 | 464 | -185 |
| h1_conservative_vs_constant | both_incorrect | 37 | 5999 | 5998 | 1 |
| value_aux_vs_base | both_correct | 24 | 3485 | 3493 | -8 |
| value_aux_vs_base | gained_correct | 1 | 56 | 130 | -74 |
| value_aux_vs_base | lost_correct | 0 | 0 | 0 | 0 |
| value_aux_vs_base | both_incorrect | 39 | 6492 | 6490 | 2 |
| value_aux_vs_h3 | both_correct | 23 | 3437 | 3364 | 73 |
| value_aux_vs_h3 | gained_correct | 2 | 123 | 259 | -136 |
| value_aux_vs_h3 | lost_correct | 2 | 279 | 330 | -51 |
| value_aux_vs_h3 | both_incorrect | 37 | 5859 | 6160 | -301 |
| value_aux_vs_constant | both_correct | 24 | 3485 | 3493 | -8 |
| value_aux_vs_constant | gained_correct | 1 | 56 | 130 | -74 |
| value_aux_vs_constant | lost_correct | 1 | 99 | 183 | -84 |
| value_aux_vs_constant | both_incorrect | 38 | 6264 | 6307 | -43 |

고정한 탐색 통과 기준(base보다 F1·EM 유지/개선 + 공통 정답 토큰 절감 + 오류0): {"h1_conservative": false, "value_aux": false}

두 후보는 기존 방법의9B 적용이다. 고정 이전 정책의 후속 결과로 학습했으며, value_aux는 표현 공동학습도 포함하므로 순수 auxiliary 효과를 분리하지 않는다. 반복 개발셋 결과이며 독립 검증 또는 지연시간 주장을 하지 않는다.

## 종합 판단

**보수적H1은 채택하지 않고, 가치 보조학습은 정확도 방향의 약한 후보로만 남긴다. 둘 다 고정한 정확도·공통정답비용 동시 기준은 미통과다.**

H1conservative는 base 대비F1 −0.2232pp, EM순증1이지만 공통정답23문항에서190토큰(+5.75%)을 더 썼다. 전체생성도241토큰(+2.40%) 증가했다. 학습317상태에서argmax변화0이었으나 개발궤적은 달라졌다. 학습상태변화가 없다는 사실이 정책전체동일을 보장하지 않는다.

Value auxiliary는 base 대비F1 +1.5625pp, EM24→25다. 문항별F1 개선1/하락0이며 기존정답24개를 모두 유지했다. 하지만 공통정답3485→3493토큰(+8,+0.23%), 전체10033→10113(+80,+0.80%)여서 효율개선은 입증하지 못했다. 기존H3 대비 공통정답23개에서는73토큰을 절약했지만 정답을2개 얻고2개 잃었으며 전체비용은 증가했다. 기준을H3로 바꿔 원래base기준의 실패를 성공으로 처리하지 않는다.

유일한F1개선은 `2hop__92768_191233`: 기존 keyword→semantic→FINAL에서 답 island(F1=0), 후보 keyword→semantic→read→keyword→read→FINAL에서 archipelago(F1=1). 생성56→130으로74토큰을 더 써 정답을 얻었다. 이는 유용한 추가정보 수집의 한 사례이며, 더 짧은 정답 경로나 일반적인장기정책능력을 증명하지 않는다. 전체80추가토큰은 이 신규정답74 + 공통정답8 − 둘다오답절약2로 분해된다.

표현공동학습과가치보조를 함께 바꾼 단일seed·반복개발셋 실험이다. 순수critic효과/통계적일반화/최적정책 주장은 보류한다. 요청한 두 실험128회와원본320회(기존192포함) 점수·토큰·해시 검증, 두단계Drive백업검증을 완료했다. 추가실험은 시작하지 않는다.
