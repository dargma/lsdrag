# lsdrag

M3DocVQA 이미지 관련 50문항과 Mem-Gallery 50문항(VS 25·VR 25)으로 멀티모달 A-RAG와 EmbeddingGemma 2의 효과를 비교합니다. 개발 질문은 최종 평가와 분리합니다.

- [실험 실행 지시서](docs/EXPERIMENT_PLAN.md)
- [결정 기록](docs/DECISIONS.md)
- [진행 기록](lab/RESEARCH_JOURNAL.md)
- [현재 상태](lab/CURRENT_STATE.json)
- [Mem-Gallery 사전 확인](lab/reports/MEM_GALLERY_PREFLIGHT.md)

실험: B0 텍스트 검색 기반 멀티모달 RAG, V1 이미지 검색 융합, V2 도구·프롬프트 개선, C1 텍스트 임베더 교체 통제.

Mem-Gallery는 B0·V1·V2와 정답 근거 직접 제공 Reader 진단을 수행합니다. 최종 평가와 비교 보고서는 아직 진행 중이며 개발 결과를 최종 성능으로 해석하지 않습니다.

주요 진입점은 `scripts/build_ir.py`, `scripts/build_index.py`, `scripts/run_experiment.py`, `scripts/score_m3doc.py`, `scripts/prepare_memgallery.py`, `scripts/score_memgallery.py`입니다. `--help`로 입력 경로와 실행 옵션을 확인할 수 있습니다. 데이터·모델·인덱스는 저장소 밖 로컬 디스크에 둡니다.

원본 텍스트 전용 동작은 `scripts/original_arag.py`로 고정된 upstream 스크립트에 직접 위임합니다. 예를 들어 `--upstream /path/to/arag --operation batch_runner -- --config configs/original_arag.yaml --questions /path/to/questions.json --output /path/to/results`를 사용합니다. 이 경로는 lsdrag의 멀티모달 도구와 프롬프트를 사용하지 않습니다. 원본 설정의 데이터·인덱스 경로와 OpenAI 호환 API 환경 변수를 실행 환경에 맞게 설정해야 합니다.
