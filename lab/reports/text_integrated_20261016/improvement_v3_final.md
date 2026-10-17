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

## Interaction steps·latency 추가

| 구간 | 조건 | n | 평균 tool steps | 평균 interaction turns | 평균 latency(s) | 중앙값(s) | p90(s) | 합계(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| validation | always_generate | 8 | 2.38 | 3.38 | 0.9679 | 0.8155 | 1.5026 | 7.7431 |
| validation | read_selection | 8 | 2.38 | 3.38 | 0.8584 | 0.7721 | 1.2716 | 6.8674 |
| evaluation | always_generate | 20 | 2.35 | 3.35 | 0.9788 | 0.9598 | 1.2545 | 19.5759 |
| evaluation | native | 20 | 2.20 | 3.20 | 1.2735 | 1.1889 | 1.6417 | 25.4697 |

fresh20에서primary tool steps 증가2 /동일18 /감소0. 관측시간합계비 native/primary=1.301. 통제된속도향상으로해석하지않는다.

latency는episode시작부터종료까지의wall time:컨트롤러·인자·답변·검색(query embedding포함)·Python처리포함. 모델load/사전corpus embedding/학습/채점/파일저장·백업제외. 따라서cold-start전체서비스latency는알수없다. 다른GPU작업간섭,조건별실제trajectory차이,순차실행·warmup/cache차이가있으며인과적속도비교아님. native형식실패1건도기본통계에포함.

동일tool step수·양쪽오류없는사후subset n=18의시간비=1.394. 같은step수라도같은state/도구결과는아니므로보조기술통계만이다.

원문항별수치CSV와JSON을함께저장했다. 원본episode/동결source는변경하지않았다.

## 인자 선택 추가 평가

같은 checkpoint에서 read 인자 선택 ON/OFF를 기존20문항에 추가 평가했다. ON은8/15 read에서 생성 생략, F1 30.43%→25.43%, EM25%→20%. 기존 primary와 원 결과는 유지한다. [별도 탐색 평가](improvement_v3_argument_selection.md).
