# PPO 행동 변화 사후 분석

기존두seed의동일40문항raw trajectory를분석. 새학습/추론없음. action0=keyword_search,1=semantic_search,2=read_chunk,3=FINAL.

| 지표 | Seed1 | Seed2 |
|---|---:|---:|
| wins | 7 | 6 |
| losses | 2 | 0 |
| ties | 31 | 34 |
| initial_changed | 2 | 1 |
| sequence_changed | 26 | 16 |
| wins_with_fewer_calls | 0 | 0 |

40문항합계: 초기정책 keyword40/semantic21/read31. PPOseed1 keyword40/semantic37/read5;seed2 keyword41/semantic31/read18. FINAL은모두40.

처음선택보다관측이쌓인후행동경로의차이가더많다. 그러나서로다른경로의후속상태는동일하지않으므로이집계로state-conditioned결정의인과효과를입증할수없다. 전체적으로read감소/semantic증가성향만으로설명되는지추가대조가필요하다.

F1이개선되면서호출수도줄어든문항은두seed모두0개. 평균F1개선과평균호출감소가같은문항에서발생했다는해석은틀리다. 집계상품질/효율동시개선과개별문항내동시개선을구분한다. 다음검증후보는전역actionbias와STOP-only대조이며이test40으로조건튜닝하지않는다.
