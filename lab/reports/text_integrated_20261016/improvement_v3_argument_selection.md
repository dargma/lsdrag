# v3 인자 선택 ON/OFF 추가 평가

사용자 요청으로 같은 frozen 통합 checkpoint와 기존 평가20문항을 두 조건에서 재실행했다. 재학습·튜닝 없음. 기존20문항 결과를 본 뒤 추가한 post-hoc 탐색 평가이며 새 heldout이 아니다. 원래 primary와 결과는 보존한다. OFF 답변·도구·인자·결과 trajectory가 원 실행과20/20 일치했다.

ON: 같은 prefill의 head가 read 인자의 GENERATE 또는 관측 후보1..4를 선택. 후보 선택이면 즉시 실행하고 GENERATE면 cached 생성. keyword/semantic 인자는 두 조건 모두 생성한다.

| 지표 | 선택 OFF: 항상 생성 | 선택 ON: 후보/생성 선택 |
|---|---:|---:|
| 답변 F1 (%) | 30.43 | 25.43 |
| EM (%) | 25.00 | 20.00 |
| 근거 F1 (%) | 46.33 | 43.00 |
| 평균 tool steps | 2.35 | 2.35 |
| 평균 interaction turns (FINAL 포함) | 3.35 | 3.35 |
| 평균 인자 생성 호출 | 2.35 | 1.95 |
| 평균 인자 생성 tokens | 43.60 | 40.20 |
| 평균 전체 생성 tokens | 60.55 | 57.00 |
| 평균 강제 prefix tokens | 30.55 | 25.35 |
| 평균 입력 tokens | 4266.00 | 4260.45 |
| 평균 episode latency (s) | 1.013 | 0.951 |
| 20문항 latency 합계 (s) | 20.255 | 19.017 |
| read 전체 / 인자 선택 실행 수 | 15 / 0 | 15 / 8 |
| 오류 episode | 0 | 0 |

read15회 중8회(53.33%)에서 생성을 생략했다. 전체47tool calls 대비17.02%다. 인자 tokens는7.80% 줄었지만, 답변 F1/EM이 각각5.00 percentage points 떨어졌다. ON 기준0승·19동률·1패이며 step/turn 평균은 동일하다. 생성 최소화 기능은 실행됐고 품질 유지에는 실패했다.

## 인자 선택 자체의 진단

고정 validation teacher read 상태 6개의 slot 정확도는 33.33% (항상 GENERATE의 slot 정확도 16.67%). 예측 선택2 / teacher 인자와 동일 후보 존재5 / 선택 slot 정답1. 전체5×5 confusion과 각 상태는 JSON에 보존했다. teacher exact-match 모방 정확도이며 답변 효용·최적 선택 지표는 아니다.

ON에서 실제 선택된8회 모두의 선택 인자와 OFF 생성 인자, 분기 전 관측 일치 여부를 JSON에 기록했다. 오답으로 바뀐 문항은 `2hop__151455_120171`: OFF 답변1986 → ON1985. OFF는 Acornsoft의 p15(운영 종료1986)를 읽었고, ON은 Castle of Riddles의 p16(게임 판매1985 언급)을 골랐다. 선택 직전 관측은 동일했다. 실제 선택8회 중3회는 OFF 인자와 같고5회는 달랐다. 선택이 문서 내용을 바꿀 수 있으므로 인자 생략 횟수만으로 성공으로 판정하지 않는다.

## 측정 범위와 재현

latency는 episode 내 제어·인자·답변 생성·검색(query embedding 포함)·Python 처리 시간이다. 모델 로딩/corpus 사전 embedding/채점/파일 저장은 제외한다. 조건별 순서를 교대했으나 동일 상태 timing 통제나 반복측정을 하지 않았다. 공유 GPU 실행이며 관측 시간 차이를 speedup 입증으로 해석하지 않는다.

- run: `/content/local_fast/lsdrag-text-v1/results/improvement_v3_argument_selection_20261017`
- head SHA256: `534da7b64150878869850cf7ccb735ec0fa8fbd643ddd0a74db19afeea4ebaef`
- 실행: `PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /content/local_fast/lsdrag-text-v1/experiment-env/bin/python scripts/evaluate_v3_argument_selection.py --output <새 local-fast 경로>`
- 집계: `PYTHONPATH=src /content/local_fast/lsdrag-text-v1/experiment-env/bin/python scripts/report_v3_argument_selection.py`
- 검색 인자 선택과 미래 남은 step의 가치 학습은 아직 구현/평가한 범위에 포함되지 않는다.
