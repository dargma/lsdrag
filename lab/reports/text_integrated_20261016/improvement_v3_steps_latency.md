# Interaction steps와episode latency 추가 집계

기존56episode를CPU로집계했으며모델재실행없음. step은실제primitive tool call,interaction turn은FINAL/실패시도를포함한controller회수다. 두값모두model forward수와다르다.

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
