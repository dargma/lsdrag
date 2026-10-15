# lsdrag

M3DocVQA 100문항과 Mem-Gallery 100문항(VS 50·VR 50)으로 멀티모달 A-RAG와 EmbeddingGemma 2의 효과를 비교합니다.

- [실험 실행 지시서](docs/EXPERIMENT_PLAN.md)
- [Google 멀티모달 임베더 구현 지시서](docs/IMPLEMENTING_GOOGLE_MULTIMODAL_EMBEDDING.md)
- [Gemini Embedding 2 논문 리뷰](docs/GEMINI_EMBEDDING_2_PAPER_REVIEW.md)
- [TrustMem 논문 상세 정리](docs/TRUSTMEM_PAPER_REVIEW.md)
- [결정 기록](docs/DECISIONS.md)
- [진행 기록](lab/RESEARCH_JOURNAL.md)
- [현재 상태](lab/CURRENT_STATE.json)
- [Mem-Gallery 사전 확인](lab/reports/MEM_GALLERY_PREFLIGHT.md)

현재 비교는 **B0 Qwen 텍스트, C1 Google 텍스트, G1 Google 텍스트+이미지**의 세 조건입니다. 같은 Reader VLM(Gemma 4 E4B-it)과 A-RAG max iteration 8을 적용합니다. 원 질문 첨부 이미지·그 캡션은 제외하고, 검색 자료의 캡션은 사용합니다. 세 조건 모두 검색 연결 실제 이미지를 자동 읽어 Reader에 전달합니다.

[최신 결과 보고서](lab/reports/INTERIM_REPORT.md)는 M3DocVQA 100문항의 세 조건을 비교합니다. Qwen3-Embedding-4B 대비 EmbeddingGemma 2 텍스트+이미지 검색의 EM은 **2→7%**, 이미지 전달 recall은 **11.4→54.3%**입니다. 텍스트 질문으로 이미지 벡터를 검색하는 원리, 임베더와 Reader VLM의 구분, A-RAG 수정 및 모델 구조 부록을 포함합니다. 전체 측정 기록은 보존합니다.

[정성 사례 원본 데이터](lab/reports/qualitative_cases/README.md): 벤치마크별 두 사례씩, 원본 질문·정답·세 조건의 전체 실행·관련 원문·이미지·캡션을 포함합니다. 다른 머신에서도 데이터 준비 없이 확인할 수 있습니다.

주요 진입점은 `scripts/build_ir.py`, `scripts/build_index.py`, `scripts/run_experiment.py`, `scripts/score_m3doc.py`, `scripts/prepare_memgallery.py`, `scripts/score_memgallery.py`입니다. `--help`로 입력 경로와 실행 옵션을 확인할 수 있습니다. 데이터·모델·인덱스는 저장소 밖 로컬 디스크에 둡니다.

원본 텍스트 전용 동작은 `scripts/original_arag.py`로 고정된 upstream 스크립트에 직접 위임합니다. 예를 들어 `--upstream /path/to/arag --operation batch_runner -- --config configs/original_arag.yaml --questions /path/to/questions.json --output /path/to/results`를 사용합니다. 이 경로는 lsdrag의 멀티모달 도구와 프롬프트를 사용하지 않습니다. 원본 설정의 데이터·인덱스 경로와 OpenAI 호환 API 환경 변수를 실행 환경에 맞게 설정해야 합니다.

## jev_v1.0 연구 브랜치

기존 로컬 Gemma 4 E4B-it에 decision head를 붙이는 연구다. 외부 decision API는 사용하지 않는다. 20문항 PoC의 학습·평가를 완료했다. 평균 시간은30.6% 감소했지만, 유일한 EM 개선 사례는 기존 답도 의미상 맞았던 출력 형식 차이였다. 추론 정확도 개선이나 head 일반화는 확정하지 않는다.

- [완료된 PoC 결과](lab/reports/jev_v1.0/PROGRESS.md)
- [한 문항 상세 비교: 모델·행동·근거·시간·원본 이미지](lab/reports/jev_v1.0/qualitative/kerry-condon/README.md)
- [head 학습·가중치·과적합 가능성 점검](lab/reports/jev_v1.0/HEAD_TRAINING_AUDIT.md)
- [실험 계획과 Astra 재검토: 두 조건만 비교](docs/JEV_EXPERIMENT_PLAN.md)
- [라벨 자료 검토: M3 우선·MultiHop-RAG 단일 보완](docs/JEV_LABEL_DATASET_REVIEW.md)
- [기여 가설: 속도·근거 탐색·기억 제어](docs/JEV_CONTRIBUTIONS.md)
- [Jev·Laya·Kev·SemIf의 최신 학습법 확인](docs/JEV_DECISION_MODEL_TRAINING_REVIEW.md)
- [Astra 최신 RAG·장기 기억 연구 메모](docs/JEV_MEMORY_RESEARCH_ASTRA.md)

## jev_v1.1 연구 주제 검토

ICML 2027 제출 준비를 위해 decision/control·멀티모달 RAG·agent memory 관련 문헌을 병렬 조사하고 Astra가 재검토했다. 단순 두 단계 value head는 새 방법으로 채택하지 않았다. 현재 조건부 후보는 **증거가 다음 검색과 최종 답변에 주는 영향을 교차 개입으로 구분해 비생성 제어를 학습하는 것**이다. 독창성·성능은 아직 실험으로 검증하지 않았다.

- [연구 방향·선행연구·baseline·2주 반증 계획](docs/JEV_ICML2027_RESEARCH_PROPOSAL.md)
- [Astra 검토·방법 정교화](docs/JEV_ICML2027_ASTRA_REVIEW.md)
