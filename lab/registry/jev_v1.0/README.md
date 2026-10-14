# jev_v1.0 분할과 감사

2026-10-14, 새 모델 학습 전에 고정한 문항 분할이다.

- 입력: 기존 `search_linked100/m3docvqa/questions.jsonl`의 100개 qid.
- 방법: qid 정렬 → Python `random.Random(20261014).shuffle` → 처음 60 train / 다음 20 development / 나머지 20 test.
- 같은 qid의 B0·C1·G1 궤적은 같은 분할이다. 성공 여부로 문항을 배정하지 않았다.
- 분할을 만든 뒤 기존 score의 EM=1인 train 궤적만 성공 teacher label로 집계했다. dev/test label은 학습에 사용하지 않는다.
- 원래 결과는 이미 관찰한 개발 자료이므로 새 비공개 평가 자료라고 주장하지 않는다.
- 문항 holdout이며 같은 검색 corpus를 공유한다. 새로운 문서 일반화와 구분하고 근거 문서 공유를 후속 감사한다.

[split.json](split.json)에 입력 해시와 전체 ID를 저장했다. [teacher_audit.json](../../reports/jev_v1.0/teacher_audit.json)은 원래 300개 궤적의 class·성공 수를 집계한 자료이며, 새로운 decision 모델의 성능 결과가 아니다.

현재 train 성공 라벨은 4개 문항·5개 궤적의 15턴이고, keyword 5 / semantic 5 / FINAL 5 / read 0이다. 이 자료만으로 일반 4-class decision을 학습·검증했다고 주장하지 않는다.

## Astra 재검토 후 실제 실행 분할

위 split은 이력을 보존한다. 실제 최소 PoC는 [label_collection_plan.json](label_collection_plan.json)의 새 M3 train82/test20을 사용하며, 기존 train60의 성공 궤적은 추가 풀이 없이 활용할 수 있다. 새 test20은 image10/nonimage10이고 새train82·기존100과 qid 중복이 없다. 기존test20은재실행·학습전용하지않는다. MultiHop목록은부족할때만사용할최대1,000문항의고정fallback이며모델실행완료기록이아니다.
