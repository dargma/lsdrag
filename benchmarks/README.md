# benchmarks/ — 내부 테스트·벤치마크 전용 (배포 skill과 무관)

> ⚠️ 이 디렉터리는 **개발 중 엔진을 검증·평가하기 위한 내부 도구**다.
> `/rag` skill이나 엔진(`src/`) 런타임은 여기를 **전혀 import/사용하지 않는다**(대원칙 5, 09_skill G7).
> 사용자 설치·사용 경로와 완전히 분리되어 있다 — skill 패키징 시 포함되지 않는다.

## 구성
| 파일 | 용도 |
|------|------|
| `eval_pageindex_agent.py` | page_index + agentic 동작 20-케이스 의도 평가(라이브) |
| `eval_benchmark.py` | 독립 골드셋으로 Recall/F1/EM + 평균 LLM호출수·시간 측정. `RAG_EVAL_SET`·`RAG_READER_PROVIDER` 지원 |
| `build_eval_set.py` | 파싱 IR에서 gold를 추출/검증해 평가셋 생성(무오류) |
| `eval_set_100.json` | 독립 벤치마크 99문항(표/그림/텍스트 + 도구5종 균형). 모델-비의존 |
| `eval_set.json` / `eval_subset.json` | 초기 12문항 / 비교용 21문항 서브셋 |
| `results/` | 검증·벤치마크 결과 보고서(REPORT/VERIFICATION/EVAL/BENCHMARK) |

## 실행 (개발자만)
```bash
# repo 루트에서, 빌드된 인덱스 + 키 준비 후
PYTHONPATH=. RAG_EVAL_SET=benchmarks/eval_set_100.json RAG_READER_PROVIDER=openai \
  python benchmarks/eval_benchmark.py
```

이 셋은 **Reader LLM(또는 임베더)을 바꿔 같은 기준으로 비교**하기 위한 것이다. 임베더는 비교 시 고정.
