# lsdrag

M3DocVQA 이미지 관련 50문항과 Mem-Gallery 50문항(VS 25·VR 25)으로 멀티모달 A-RAG와 EmbeddingGemma 2의 효과를 비교합니다. 개발 질문은 최종 평가와 분리합니다.

- [실험 실행 지시서](docs/EXPERIMENT_PLAN.md)
- [결정 기록](docs/DECISIONS.md)
- [진행 기록](lab/RESEARCH_JOURNAL.md)
- [현재 상태](lab/CURRENT_STATE.json)
- [Mem-Gallery 사전 확인](lab/reports/MEM_GALLERY_PREFLIGHT.md)

현재 비교는 **B0 Qwen 텍스트, C1 Google 텍스트, G1 Google 텍스트+이미지**의 세 조건입니다. 동일 Reader VLM과 A-RAG 최대 8회 반복을 적용하며, 모든 조건에서 검색된 캡션의 연결 이미지를 읽을 수 있습니다. Mem-Gallery 질문은 텍스트만 입력합니다.

[최신 결과 보고서](lab/reports/INTERIM_REPORT.md): 개발 M3DocVQA 50문항과 Mem-Gallery 10문항을 조건별 문항당 한 번씩 평가했습니다. 이미지 검색 recall은 개선됐지만 정답률 개선을 확정하기에는 표본이 작습니다. 별도 최종 평가는 아직 시작하지 않았습니다.

주요 진입점은 `scripts/build_ir.py`, `scripts/build_index.py`, `scripts/run_experiment.py`, `scripts/score_m3doc.py`, `scripts/prepare_memgallery.py`, `scripts/score_memgallery.py`입니다. `--help`로 입력 경로와 실행 옵션을 확인할 수 있습니다. 데이터·모델·인덱스는 저장소 밖 로컬 디스크에 둡니다.

원본 텍스트 전용 동작은 `scripts/original_arag.py`로 고정된 upstream 스크립트에 직접 위임합니다. 예를 들어 `--upstream /path/to/arag --operation batch_runner -- --config configs/original_arag.yaml --questions /path/to/questions.json --output /path/to/results`를 사용합니다. 이 경로는 lsdrag의 멀티모달 도구와 프롬프트를 사용하지 않습니다. 원본 설정의 데이터·인덱스 경로와 OpenAI 호환 API 환경 변수를 실행 환경에 맞게 설정해야 합니다.
