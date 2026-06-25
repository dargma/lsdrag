<!-- README는 GitHub 대표 얼굴. 08_e2e 이후 실제 질의-응답 예시로 채워 완성한다. 지금은 골격. -->

<h1 align="center">lsdrag</h1>
<p align="center"><b>그래프 없이 GraphRAG급. 페이지·Figure·이미지까지 읽는 agentic RAG —<br/>Claude Code <code>/rag</code> 한 줄로.</b></p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10+-blue">
</p>

<!-- TODO(08 이후): docs/demo.gif (실제 /rag 질의-응답 녹화) -->

## What is this?
사내 문서를 위한 **agentic RAG**. 그래프를 빌드하지 않고도 GraphRAG급 성능을 낸다.
페이지·Figure 같은 **구조 질의**와 다이어그램 같은 **이미지 질의(VLM)**까지 답한다.
Claude Code의 `/rag` skill로 즉시 사용.

## Why
- 🚫 **그래프 빌드 불필요** — vector + Page Index + 조건부 VLM으로 그래프 효과만 가져온다. 빠르다.
- 🧩 **파싱 → DB 구축 → 검색 3단계 완전 분리** — 부품(파서/임베더/Reader) 통째 교체 쉬움.
- 🖼️ **멀티모달** — 비트필드 다이어그램 같은 이미지를 읽고 설명한다.
- 🩺 **설치 self-check 내장** — `doctor.py`가 9단계로 어디서 막혔는지 사람 말로 알려준다.
- ➕ **문서 add/remove** — 전체 재빌드 없이 문서만 증분 반영. **삭제도 한 명령.**

## Quick Start
```bash
export UP_TOKEN=...  OPENAI_API_KEY=...
cp -r rag ~/.claude/skills/rag
python -m src.indexing.build --config config.yaml
python rag/scripts/doctor.py --json     # 전부 ✅면 끝
```
자세한 건 → [INSTALL.md](INSTALL.md)

## Usage
```
> /rag SCTLR_EL1 레지스터가 무엇인가?
< (텍스트 답변 + 출처 페이지)

> /rag TCR_EL1 비트필드가 몇 페이지 어느 figure에 있나?
< (page_index_search로 위치 특정 → 해당 figure를 VLM이 읽어 비트필드 설명)
```
<!-- TODO(08 이후): 실제 ARM 매뉴얼 질의-응답으로 교체 -->

## How it works
```
┌─ 1. 파싱 ──────┐   ┌─ 2. DB 구축 ────────┐   ┌─ 3. 검색 ─────────────────┐
│ 문서 → Upstage │ → │ IR → 임베딩 → 인덱스  │ → │ 쿼리 → A-RAG agent         │
│      → IR      │IR │     → Page Index     │   │ → tool 5종 → [VLM] →       │
│                │   │                      │   │ Reader(GPT-4.1 mini) → 답  │
└────────────────┘   └──────────────────────┘   └────────────────────────────┘
   교체: 파서          교체: 임베더/저장소         교체: 검색전략/Reader
```

## Swap components
| 바꿀 것 | 어디만 손대면 되나 |
|---------|-------------------|
| 파서(사내 파서로) | `src/parser/adapter.py` |
| Reader/VLM | `config.yaml`의 `reader` 블록 |
| 임베더 | `config.yaml`의 `embedding.model` |

## Project layout
```
lsdrag/
├── docs/        모듈 명세 (00~09)
├── src/         schema · parser · page_index · indexing · retrieval · agent/vendor
├── rag/         skill (SKILL.md, run/docs/doctor/uninstall)
├── examples/    ARM v8-A 예시 PDF
└── tests/       회귀 + E2E
```

## Install
요약은 위 Quick Start, 전체는 [INSTALL.md](INSTALL.md). 삭제: `python rag/scripts/uninstall.py`.

## Credits & License
MIT. 엔진은 [A-RAG](https://github.com/Ayanami0730/arag)(@`a44de6b`, MIT)를 vendoring.
가져온/지운 목록은 [`src/agent/vendor/SOURCE.md`](src/agent/vendor/SOURCE.md).
