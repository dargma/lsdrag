# 통합 행동/read head 실험 v3 — 완료

수정된 native형식 생성경로는 train24상태 모두 native값과일치했다. 고정2상태의FP32 cache/full오차는5.48e-5이하로 precision진단을통과했다. 기존BF16 절대기준을바꾼것이아니며24train/2precision검사는독립QA성공증거가아니다.

고정train24의82natural states로action+read-slot head를함께학습했다. 검증8문항의29고정상태에서행동macroF1=.81399,prematureFINAL0. readslot4의train양성은0으로부분coverage한계가있다. 복구추가/seed선택없음.

| validation8 조건 | F1 | EM | 평균 인자 생성 토큰 | 평균 read 재사용 |
|---|---:|---:|---:|---:|
| always-generate | .21875 | .125 | 41.75 | 0 |
| read-selection | .09375 | 0 | 38.625 | .375 |

read선택은한문항(Ralph Vaughan Williams)의정답을상실하여미채택. 동일통합checkpoint의actionhead+always-generate를fresh평가전에고정했다.

| fresh20 조건 | F1 | EM | 평균 생성 토큰 | 평균 강제 prefix 토큰 | 평균 입력 토큰 | 평균 tool calls | 형식오류 episodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| native | .2900 | .25 | 89.75 | 0 | 3928.5 | 2.20 | 1 |
| primary | .304286 | .25 | 60.55 | 30.55 | 4266.0 | 2.35 | 0 |

1승0패19동률이며F1차이+.014286은한문항의부분점수차이. 탐색paired bootstrap95%구간[0,.042857]. EM정답수는각5/20. 통계적품질향상/비열등성주장없음. native형식실패1건은두조건F1동률인별도문항이며F1증가를설명하는문항은아니다.

생성토큰은32.5%줄었으나생성+강제prefix는89.75→91.10,입력토큰은약8.6%증가. decoder forward도줄었지만전체계산량/latency향상입증은아님. 자율rollout tool calls가다르며동일step실험으로부르지않는다. Native argument_tokens=0은별도계측누락이며생성인자가없었다는뜻이아니다.

검증16+fresh평가40episode완료,원본보존. 현재worker종료. 추가자동확장없음. 상세JSON:improvement_v3_final.json.
