# lsdrag

M3DocVQA 100문항과 Mem-Gallery 100문항(VS 50·VR 50)으로 멀티모달 A-RAG와 EmbeddingGemma 2의 효과를 비교합니다.

- [실험 실행 지시서](docs/EXPERIMENT_PLAN.md)
- [Google 멀티모달 임베더 구현 지시서](docs/IMPLEMENTING_GOOGLE_MULTIMODAL_EMBEDDING.md)
- [결정 기록](docs/DECISIONS.md)
- [진행 기록](lab/RESEARCH_JOURNAL.md)
- [현재 상태](lab/CURRENT_STATE.json)
- [Mem-Gallery 사전 확인](lab/reports/MEM_GALLERY_PREFLIGHT.md)

현재 비교는 **B0 Qwen 텍스트, C1 Google 텍스트, G1 Google 텍스트+이미지**의 세 조건입니다. 같은 Reader VLM(Gemma 4 E4B-it)과 A-RAG max iteration 8을 적용합니다. 원 질문 첨부 이미지·그 캡션은 제외하고, 검색 자료의 캡션은 사용합니다. 세 조건 모두 검색 연결 실제 이미지를 자동 읽어 Reader에 전달합니다.

[최신 결과 보고서](lab/reports/INTERIM_REPORT.md): 자동 전달 방식으로 각 벤치마크 100문항 × 세 조건, 총 600회 풀이·채점 완료. Qwen 대비 Google multimodal의 EM은 M3DocVQA **2→7%**, Mem-Gallery **24→22%**입니다. 이미지 전달 recall은 각각 **11.4→54.3%**, **53.7→55.1%**입니다. 문서 이미지 검색은 개선됐지만 Mem-Gallery의 정답률 개선은 없었고, 전체 교체를 추천할 근거는 부족합니다. 반환된 이미지 연결 결과의 전달 누락은 0건입니다.

[정성 사례 원본 데이터](lab/reports/qualitative_cases/README.md): 벤치마크별 두 사례씩, 원본 질문·정답·세 조건의 전체 실행·관련 원문·이미지·캡션을 포함합니다. 다른 머신에서도 데이터 준비 없이 확인할 수 있습니다.

주요 진입점은 `scripts/build_ir.py`, `scripts/build_index.py`, `scripts/run_experiment.py`, `scripts/score_m3doc.py`, `scripts/prepare_memgallery.py`, `scripts/score_memgallery.py`입니다. `--help`로 입력 경로와 실행 옵션을 확인할 수 있습니다. 데이터·모델·인덱스는 저장소 밖 로컬 디스크에 둡니다.

원본 텍스트 전용 동작은 `scripts/original_arag.py`로 고정된 upstream 스크립트에 직접 위임합니다. 예를 들어 `--upstream /path/to/arag --operation batch_runner -- --config configs/original_arag.yaml --questions /path/to/questions.json --output /path/to/results`를 사용합니다. 이 경로는 lsdrag의 멀티모달 도구와 프롬프트를 사용하지 않습니다. 원본 설정의 데이터·인덱스 경로와 OpenAI 호환 API 환경 변수를 실행 환경에 맞게 설정해야 합니다.
