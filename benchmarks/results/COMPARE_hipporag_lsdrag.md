# COMPARE — HippoRAG 2 vs lsdrag (공정 비교, 내부)

> 동등 조건: **같은 corpus(1798 페이지 passage, page/figure/table 구조 주입)** · **같은 두 평가셋** ·
> **같은 Reader(gpt-4.1-mini)** · **같은 지표**. HippoRAG 2는 vllm/gritlm 스텁 + OpenAI 경로로 실행.

## 결과

| 셋 | 시스템 | Recall | F1 | EM | 비고 |
|----|--------|:---:|:---:|:---:|------|
| **set100** (99, 단답 grounding) | HippoRAG 2 | **100%** | **0.486** | **38.4%** | 질의=단일 LLM콜 |
| | lsdrag (page index) | 92.9% | 0.386 | 34.3% | agentic 평균 4.66 루프, 5.2s/Q |
| **longctx** (20, 멀티홉/집계) | HippoRAG 2 | **92.5%** | 0.256 | 25% | multi-src recall |
| | lsdrag (page index) | 72.5% | 0.236(synth) | — | 평균 6.25 루프, 8.5s/Q |

(longctx Recall = multi-source recall: 필요 출처 중 검색된 비율. lsdrag synthesis=집계 커버리지/token F1.)

## 핵심 발견
1. **HippoRAG가 검색 Recall 우세** — 특히 **멀티홉 92.5% vs 72.5%**. 그래프(PPR) 기반 associativity 강점이 실측됨.
2. **답변 F1/EM도 HippoRAG가 소폭 높음**(단답 0.486 vs 0.386).
3. **자원 트레이드오프 정반대**:
   - HippoRAG 오프라인 인덱싱 = **OpenIE LLM ~57분**(1798 passage). 질의는 단일 LLM콜(빠름).
   - lsdrag 인덱싱 = **수 초**(임베딩만). 질의는 agentic 다단계(4.7~6.3콜).

## 정직한 공정성 한계 (해석 시 유의)
- **임베더 미통제**: HippoRAG=`text-embedding-3-small`(OpenAI) vs lsdrag=`all-MiniLM-L6-v2`(22M).
  lsdrag Recall 불리의 큰 원인 — **동일 임베더(Qwen/OpenAI)로 맞추면 격차 축소 예상**.
- **멀티모달 무력화**: 페이지 passage가 figure를 텍스트로 평탄화 → lsdrag의 `image_read`(그림 VLM) 강점은 이 텍스트-only 비교에 미반영.
- **Reader 통제됨**(gpt-4.1-mini 동일). Claude reader 비교는 HippoRAG용 OpenAI-호환 claude-CLI shim 필요(미실행).
- **corpus = 매뉴얼 일부**(~1800p, 크레딧 소진까지). 단일 실행(노이즈 존재).

## 종합
- **검색 품질·멀티홉**: HippoRAG 2 우위(설계 목적대로). 단 **무거운 LLM 인덱싱** 비용.
- **인덱싱 경량·즉시성·멀티모달(그림)·키 없는 Reader(claude_code)**: lsdrag 강점(이 비교엔 일부 미반영).
- 다음 단계(더 공정): lsdrag 임베더를 동일하게 통제 → 순수 "retrieval 메커니즘" 대 비교.

> 원시: `HIPPORAG_eval_set_100.md`·`HIPPORAG_eval_longctx.md`·`LSDRAG_pages_*.md`
