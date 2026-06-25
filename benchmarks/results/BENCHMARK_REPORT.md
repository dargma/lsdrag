# BENCHMARK REPORT — 임베더 × Reader 조합 비교 (내부 평가)

> 내부 테스트 전용(배포 skill과 무관). 같은 인덱스·같은 평가셋에서 **부품만 바꿔** 비교했다.

---

## 1. 데이터셋 — 구성과 의도

### 1.1 코퍼스 (중요: 매뉴얼 일부만)
ARM v8-A Reference Manual(전체 6,354p)에서 **Upstage 크레딧 소진까지 ~1,800p(20조각)만 파싱**했다.
검증 구간(E2-2797) 중심으로 앞뒤 확장(≈ D1·D2·D5·E2 시스템 레벨 영역). **전체 매뉴얼이 아니다** —
모든 질문·"문서에 없음(부정)" 판정은 **인덱싱된 이 부분 기준**이다.
규모: 20 docs · 36,467 chunks · 79,515 sentences · 2,385 tables · 376 figures.

### 1.2 평가셋 `eval_set_100.json` (99문항) — 구성
gold를 **원문에서 추출·verbatim 검증**(무오류), **모델 비의존**(Reader를 바꿔도 공통 기준).

| 컨텍스트 | category | 수 | 의도 도구 |
|----------|----------|----|----------|
| 텍스트 | concept(25) · heading-loc(16) | 41 | semantic_search · page_index_search |
| 표 | table(17) · table-loc(8) | 25 | read_chunk · keyword_search |
| 그림 | figure(8) · figure-loc(15) | 23 | image_read · page_index_search |
| 부정 | negative(10) | 10 | (없음 — 환각 저항) |

도구 균형: page_index 31 · semantic 25 · keyword 17 · read_chunk 8 · image_read 8 · (부정)10.
→ **표·그림·텍스트 + 에이전트 도구 5종**이 골고루 작동하는지를 본다.

### 1.3 한계 (정직)
이 셋은 **검색 + 단답 grounding + 도구 선택 + 환각 저항**을 잰다.
**긴-컨텍스트/멀티홉 이해는 측정하지 않는다**(gold가 단일 청크의 짧은 span). → 별도 `eval_longctx.json`로 분리 측정.

---

## 2. 비교 조합 — 무엇을, 왜

- **임베더(검색 품질)**: `all-MiniLM-L6-v2`(22M, 기본) · `Qwen3-Embedding-0.6B`(소형 SOTA급) · `Qwen3-Embedding-8B`(대형).
  → "임베더를 키우면 검색이 좋아지나? 자원 대비 가치는?"
- **Reader(답변 LLM/VLM)**: `openai`(GPT-4.1 mini, API·유료) · `claude_code`(Claude Code 내장, 로컬 CLI·키 불필요).
  → "Reader를 바꾸면 답·자원이 어떻게 달라지나?"
- 임베더는 빌드만 GPU(L4), 서빙은 어디서나. Reader는 둘 다 멀티모달.

---

## 3. 결과

### 3.1 전체 99문항 (Reader = openai 고정, 임베더 비교)
| 임베더 | 검색 Recall | 답변 F1 | 답변 EM |
|--------|:-----------:|:-------:|:-------:|
| all-MiniLM-L6-v2 (22M) | 93.9% | 0.474 | 40.4% |
| **Qwen3-Embedding-0.6B** | **96.0%** | 0.463 | 38.4% |

→ 임베더를 키우면 **검색 Recall +2pp**. 답변 F1/EM은 비슷(Reader가 좌우). **임베더 효과 = 검색을 더 잘 찾아줌**.

### 3.2 3×2 매트릭스 (균형 서브셋 21문항 — 자원 지표 포함)
| 임베더 | Reader | Recall | F1 | EM | 평균 LLM 호출 | 평균 시간 |
|--------|--------|:------:|:---:|:---:|:---:|:---:|
| MiniLM | openai | 100% | 0.494 | 38% | 5.05 | 5.5s |
| MiniLM | **claude_code** | 95% | 0.441 | **57%** | **2.52** | 13.7s |
| Qwen-0.6B | openai | 100% | 0.514 | 38% | 5.62 | 6.0s |
| **Qwen-0.6B** | **claude_code** | 100% | **0.526** | **62%** | 2.81 | 17.7s |
| Qwen-8B | openai | 100% | **0.537** | 33% | 5.38 | 7.3s |
| Qwen-8B | claude_code | 100% | 0.482 | 57% | 2.71 | 16.5s |

---

## 4. 지표/결과별 의미

- **검색 Recall** = 정답이 든 출처(페이지/핵심어)를 검색 trajectory가 실제로 가져온 비율. *검색 단계*의 성능. 임베더가 주로 좌우.
- **F1** = 답변 토큰과 gold의 겹침(부분점수). 생성형이 장황하면 정밀도가 깎여 낮게 나옴 → 절대값보다 *상대 비교*로 본다.
- **EM** = 정답(짧은 span)이 답변에 정확히 들어갔나. *실용적 정답률*. **Claude가 일관되게 높음(57~62% vs 33~38%)** = 정확한 단답을 잘 맞춘다.
- **평균 LLM 호출** = 한 질문에 도구 루프를 몇 번 도나. **claude_code ≈ 2.5~2.8 vs openai ≈ 5.0~5.6** → Claude가 **절반의 호출**로 더 결정적으로 끝낸다.
- **평균 시간** = 답까지 wall-clock. **openai ≈ 5.5~7.3s vs claude_code ≈ 13.7~17.7s** → claude CLI는 스텝마다 프로세스 기동이라 **느림**(호출은 적지만 호출당 비쌈).

### 핵심 발견
1. **Reader가 정답 품질을 좌우**: Claude(claude_code)가 EM에서 확실히 우세 + 호출 수 절반. 단, wall-time은 2~3배 느림.
2. **임베더 ↑ = 검색 Recall ↑(완만)**, 답변엔 거의 무영향. **8B는 0.6B 대비 뚜렷한 이득 없음** + 자원(15GB·빌드↑) ↑ → **0.6B가 sweet spot**.
3. 서브셋(21)은 Recall이 거의 100%로 포화 → 임베더 변별은 **전체 99(§3.1)**에서 보는 게 맞다.

---

## 5. "적은 자원으로 가능한가?" — 자원 관점 결론

| 우선순위 | 조합 | 이유 |
|----------|------|------|
| **키 없이·정확도 우선** | MiniLM/Qwen-0.6B + **claude_code** | API 키 0(Claude Code 내장), EM 최고, 호출 절반. *느린 것만 감수* |
| **속도·처리량 우선** | MiniLM/Qwen-0.6B + openai | wall-time 2~3배 빠름. API 키·비용 필요 |
| **검색 정밀도 더** | Qwen-0.6B (빌드만 GPU) | Recall +2pp. 서빙은 CPU도 OK |
| 과함 | 8B | 자원 대비 이득 미미 |

- **임베더 빌드는 GPU 1회, 서빙은 CPU 가능**(쿼리 인코딩 ~0.1~0.3s/건).
- **가장 적은 자원**: MiniLM(22M, CPU) + claude_code(키 0) → 외부 비용 없이 동작, 정확도도 양호.

> 측정 환경: NVIDIA L4, eval_set_100(파싱 부분). 서브셋 21은 표본이 작아 절대수치보다 경향으로 해석.
> 원시 로그: `BENCHMARK.md`(MiniLM-openai 99), `BENCHMARK_qwen.md`(Qwen-openai 99).
