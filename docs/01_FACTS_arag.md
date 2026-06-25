# 01 — FACTS (불변 사실, 변경 금지)

> 모든 내용은 **실제 A-RAG 레포에서 직접 확인한 코드 사실**이다.
> 출처: `github.com/Ayanami0730/arag` @ commit `a44de6b` · 라이선스 **MIT**(vendoring 자유).
> 추측이 아니다. 어긋나는 구현은 틀린 것이다. 의심되면 레포를 직접 확인하고 근거(파일·줄)와 함께 갱신하라.

## vendoring 대상 (런타임 경로가 import하는 것만)

```
src/arag/
├── core/{context.py, llm.py, config.py}
├── tools/{base.py, registry.py, semantic_search.py, keyword_search.py, read_chunk.py}
└── agent/base.py
```
평가용 곁가지(`scripts/eval.py`, `batch_runner.py`, 데모, 미사용 tool)는 **제외**. → `src/agent/vendor/`로 가져온다.

## 사실 1 — Tool 계약 (`tools/base.py`)

모든 tool은 `BaseTool`(ABC) 상속, **3개 구현**:
```python
class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]: ...                  # OpenAI function schema
    @abstractmethod
    def execute(self, context: 'AgentContext', **kwargs) -> Tuple[str, Dict[str, Any]]: ...
        # 반환: (LLM에게 갈 결과 문자열, 로그 dict)
```

## 사실 2 — Tool 등록 (`tools/registry.py`)
```python
registry.register(tool)                   # tool.name을 키로 저장
registry.get_all_schemas()                # 전체 스키마 → LLM 전달
registry.execute(name, context, **kwargs) # 이름으로 실행, 예외는 내부에서 에러 문자열로
```
신규 tool은 `register()` 한 줄로 추가. **코어 수정 불필요.**

## 사실 3 — 스니펫 vs 본문 (검증됨)
- `semantic_search` / `keyword_search` → 매칭 문장 **스니펫**만. (전체는 read_chunk로 — 스키마에 명시)
- `read_chunk` → 청크 **전체 본문**.
- **함의**: `page_index_search`를 "위치 특정 + 본문 반환"으로 만들면 그 경로에선 read_chunk 생략 가능. 일반 검색 경로는 read_chunk 유지.

## 사실 4 — 중복 방지 (`core/context.py`)
`AgentContext`가 읽은 청크를 set으로 추적:
```python
read_chunk_ids: Set[str]
mark_chunk_as_read(chunk_id); is_chunk_read(chunk_id) -> bool
add_retrieval_log(tool_name, tokens, metadata)   # 모든 tool이 호출
```
- **함의**: `image_read`도 같은 패턴으로 VLM 중복 호출 차단. **코어에 가하는 유일한 변경** =
  `AgentContext`에 `read_image_ids: Set` + `mark/is_image_read` 추가(청크 패턴 복제).
  vendor 직접 수정보다 **서브클래스 확장 우선**.

## 사실 5 — Agent loop (`agent/base.py`)
```python
class BaseAgent:
    def run(self, query: str) -> Dict[str, Any]: ...
```
- `max_loops`(기본 10) 반복, LLM이 tool 호출.
- `max_token_budget`(기본 128000) 초과 시 `_force_final_answer`로 강제 종료. **무한루프 방지 이미 구현됨.**
- `messages`엔 텍스트가 쌓임 → `image_read`는 VLM 결과를 **텍스트 설명으로 변환**해 반환(기존 `(str, dict)` 계약 유지).

## 사실 6 — Config (`core/config.py`)
A-RAG는 이미 `Config.from_yaml(path)` / `.get(key, default)` 제공. 우리 `config.yaml`과 호환.
**새 config 체계를 만들지 말고 이걸 확장해 쓰는 것을 우선 검토.**

## 사실 7 — A-RAG 임베더 (`tools/semantic_search.py`)
`semantic_search`는 **sentence-transformers**로 임베딩(검증됨).
- 코드 기본값 `all-MiniLM-L6-v2`. config 예시값 `Qwen/Qwen3-Embedding-0.6B`.
- config의 `embedding.model`로 지정. 작은 모델 → 로컬 자동 로드, **별도 서빙·GPU 불필요.**
- **함의**: 임베더는 A-RAG 내장 그대로. 추가 임베딩 인프라 만들지 마라. 05_indexing이 이걸 호출해 인덱스 생성.
- **검증된 인덱스 형식**: `semantic_search`는 `index_dir/sentence_index.pkl`을 읽는다(`tools/semantic_search.py:59`).
  05_indexing은 **같은 임베더로 이 파일을 생성**해야 query/index 벡터공간이 일치한다.
  (전체 FACTS는 `github.com/Ayanami0730/arag@a44de6b` 실코드와 대조 검증 완료 — 2026-06-25.)

## 사실 8 — 외부 API 스펙 (공식 문서 검증)

**Upstage Document Parse** (파서)
- `POST https://api.upstage.ai/v1/document-digitization`, form: `document=@file`, `model=document-parse`.
  인증: `Authorization: Bearer $UP_TOKEN`.
- 한도: sync 100p / async 1000p, 50MB. 분할 PDF(50~70p)는 sync로 충분.
- 응답: `elements[]` 각 = `{id, category, content{html,markdown,text}, page, coordinates}`. 전체 `content{...}`도 제공.
- category: paragraph, table, figure, chart, heading1, header, footer, caption, equation, list, index, footnote.
- figure/chart/table 이미지를 base64로 받으려면 `base64_encoding=['figure', ...]`.

**Reader LLM = GPT-4.1 mini (멀티모달)**
- OpenAI Chat Completions(`/v1/chat/completions`), 키 `OPENAI_API_KEY`. 멀티모달 입력(이미지) 지원.
- **VLM 겸용**: image_read가 이미지를 이 Reader로 직접 보낼 수 있어 별도 VLM 백엔드 불필요.
- A-RAG `LLMClient`가 OpenAI 호환이면 base_url/model만 바꿔 연결.

## 우리가 추가하는 것 (요약)

| 항목 | 종류 | 계약 |
|------|------|------|
| `page_index_search` | 신규 tool | BaseTool 상속, 위치 특정 + 본문 + 이미지명 |
| `image_read` | 신규 tool | BaseTool 상속, 이미지→텍스트 설명, context로 중복 방지 |
| `read_image_ids` 등 | 코어 확장 | AgentContext에 이미지 추적 추가 (유일한 코어 변경) |

이 셋 외 A-RAG 코어는 손대지 않는다.
