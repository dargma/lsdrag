# Frozen 도구 head 새 2-hop100 평가

상태 `complete`, paired 100문항. 모델·head 고정, 재학습·인자 선택 없음.

| 지표 | Native | Head + 인자 생성 |
|---|---:|---:|
| 답변 F1 (%) | 31.1133 | 31.5705 |
| EM (%) | 19.0000 | 20.0000 |
| 근거 F1 (%) | 39.2333 | 40.0667 |
| 평균 tool calls | 2.2600 | 2.3900 |
| 평균 interaction turns | 3.2600 | 3.3900 |
| 평균 생성 tokens | 94.3200 | 60.8100 |
| 평균 forced-prefix tokens | 0.0000 | 31.0700 |
| 평균 입력 tokens | 3914.0100 | 4106.2400 |
| 평균 prefill 수 | 3.2600 | 3.3900 |
| 평균 decoder forward (산출) | 91.0600 | 59.8100 |
| 평균 episode latency(s) | 1.3366 | 0.9783 |
| latency 총합(s) | 133.6616 | 97.8335 |
| 오류 episode | 4.0000 | 0.0000 |

F1 기준 head 5승/90동률/5패. Native exact 정답 상실3 / 새 exact 획득4.
paired F1차이=0.004572, descriptive bootstrap95=[-0.049375, 0.05957236842105264].
관측 시간 총합 Native/head=1.366. 통제된speedup주장없음.

## 시간 성분 관측

Native prefill=0.1448s, decode=1.1438s; head prefill=0.1477s, 일괄prefix=0.0418s, decode=0.7438s. 각경로의usage timer집계이며같은도구/인자/답변을강제한인과분리실험은아니다.

## 해석과 한계

head F1·EM 점추정이 Native 이상인가: True. 이값은확인평가의점추정비교이며비열등성입증이아니다. 실패문항을보고이100문항에맞춰튜닝하지않는다. 후속학습은train자료에서만하고새문항으로재평가해야한다.

기존노출질문·분해하위질문family와겹치지않는100문항을outcome-blind hash순서로사전동결했다. 새100내부의atomic family도불중복이다. 2-hop dev1252중1099eligible에서선별돼원dev전체의무작위표본과같지않다. 모델pretraining노출까지배제한주장은아니다.

Native오류도주분석에포함했다. head조건은도구선택뿐아니라aligned/cache실행경로도바뀌므로 전체차이를learnedhead선택효과하나로돌리지않는다. Native의인자토큰별도값0은미계측이다. 생성+forcedprefix와입력토큰도함께보며출력감소를전체계산감소와동일시하지않는다.

사후보조분석: 양쪽오류없는96문항의F1차이는 -0.565%p, EM차이 0.000%p. Native형식오류4문항중head가정답을낸것은1문항이다. 주분석은여전히100전체다. 형식안정성의기여와정책의질을구분하기위한기술통계이며오류조건으로선택된subset의한계가있다.

latency는실제episode제어·인자/답변생성·검색/queryembedding·계측용tokenization을포함한다. 모델load/문서사전embedding/채점/파일저장제외. corpus준비는별도기록. 순서교대와단일rollout의공유GPU관측값이며통제속도또는동일trajectory비교가아니다.

개별답변·최초분기·qid별품질차이·source/inputhash는JSON,전episode수치는CSV에있다. 최초분기는후보원인분류이며자동인과판정이나최적tool정답label이아니다. 새head학습/3-hop확대는이번실행에포함하지않았다.
