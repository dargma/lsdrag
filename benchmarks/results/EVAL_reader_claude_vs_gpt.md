# Reader 비교 — Claude(Claude Code 내장) vs GPT-4.1 mini

> 같은 ARM v8-A 인덱스에서 Reader/VLM만 바꿔 비교. Claude는 **API 키 없이** 로컬 `claude` CLI
> (지금 Claude Code에서 도는 모델)를 사용(`reader.provider: claude_code`).

## A. VLM 비교 — 같은 figure·같은 프롬프트로 그림 읽기

| Figure | Claude (claude_code) | GPT-4.1 mini (OpenAI) |
|--------|----------------------|------------------------|
| 상태기계 ① (E2-2797) | ✅ 전 전이를 표로 완전, `*`=IMPLEMENTATION DEFINED 의미까지 | △ 일부만, `StoreExc**1**`(l→1 오독) |
| 상태기계 ② (E2-2801) | ✅ `n/!n`·`Marked_address`·`*`·`‡`(succeed/fail) 정확 해석 | △ "기호 !·*는 **불명확**…" (표기 이해 못 함) |
| "Note" 배너 (E2-2804) | ✅ **"상태도 아님, Note 헤더"** 정확 식별(+파서 오분류 지적) | ❌ **환각** — 없는 `q0/q1, a/b` 자동기계를 지어냄 |

- **Claude 3/3 정확**(비-다이어그램까지 올바르게 거부). **GPT-4.1 mini: 부분1 + 모호1 + 환각1.**
- 가장 큰 차이는 **환각**: 다이어그램이 아닌 그림에 GPT는 가짜 상태기계를 지어냈고, Claude는 정확히 잡았다.
  기술 문서 RAG에서 이는 곧 **답변 신뢰도**.
- 보너스: `E2-2804`의 figure는 실제론 "Note" 배너 → **Upstage 오분류**(데이터 품질 노트). Claude가 짚음.

## B. agent 루프 비교 — Claude Code 내장이 도구를 호출하며 답하는가

`provider: claude_code`는 `claude` CLI가 A-RAG 루프를 매 스텝 구동한다(외부 API·키 0).

> 질의: "Store-Exclusive와 글로벌 모니터는 몇 페이지?"

```
loop1: page_index_search(heading='Store-Exclusive')   ← 올바른 도구·인자명
loop2: (종료) → "printed page E2-2800"                ← 실제 페이지 라벨 정확 인용
```
**2 loop만에 정답** — API 없이 Claude Code 내장 모델로 end-to-end 동작 확인.
(초기엔 `query` 같은 잘못된 인자로 헤맸으나, 컨트롤러 프롬프트에 도구별 정확한 인자·종료 유도를 넣어 수렴.)

## 결론 / 권장
- **기본 권장 = `claude_code`**: 키 불필요, 다이어그램 해석 우수, **환각 적음**.
- **OpenAI/Anthropic API**: 키가 있고 더 빠른 처리량이 필요할 때(각 스텝이 단일 API 호출). `claude_code`는
  스텝마다 `claude` CLI를 띄워 **느린 대신** 키가 없다.
- 셋 다 OpenAI 호환/동일 인터페이스라 `reader.provider` 한 줄로 전환.

## 한계 (정직하게)
- A의 VLM 비교는 3개 figure 표본. B의 루프 비교는 대표 질의 1건(수렴·정확 인용 확인).
- `claude_code`는 스텝당 CLI 기동이라 지연이 크다(다건 평가엔 API provider가 빠름).
- 전체 20케이스를 Claude로 재평가하려면 시간이 더 들 뿐 동일 방식으로 가능.
