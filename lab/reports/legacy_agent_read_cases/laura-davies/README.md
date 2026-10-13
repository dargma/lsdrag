# M3DocVQA · 매우 짧은 금발의 선수 찾기

문항 `be0baecf90a40b1ba8230d9c771f75ca` · additional50. 모델·데이터 다운로드 없이 이 폴더의 JSON과 이미지를 열면 실행 현황을 확인할 수 있습니다.

**원문 질문:** Which player with very short blond hair is included in the order of merit rankings of the 2009 Ladies European tour?

**정답:** `["Laura Davies"]`

**원 질문 첨부 이미지:** 없음. **이 질문 첨부 이미지의 실제 입력:** 없음. 검색 근거 이미지는 아래 별도로 표시한 실제 입력 기록이다.

**원 질문 이미지의 캡션:** 원 데이터에 없음. **이 질문 캡션의 실제 입력:** 없음. 검색 대상 이미지의 캡션은 검색용으로 사용하며 [캡션 목록](caption_inventory.json)에 기록했습니다.

- [원본 질문](raw/question.json) · [정답·주석](raw/gold.json) · [원 데이터 질문](raw/source_question.json)
- 전체 실행: [Qwen](predictions/B0.json) · [Google text-only](predictions/C1.json) · [Google multimodal](predictions/G1.json)
- [원문 청크](raw/chunks.json) · [원본 이미지 메타데이터](raw/figures.json) · [로컬 이미지 경로](figures_local.json) · [캡션 목록](caption_inventory.json)
- [파일 해시·출처](manifest.json). 기존 원본의 절대 경로는 출처 기록이며 `figures_local.json`의 `image_path`는 이 폴더 기준 상대 경로입니다.

## 실제 답변·입력 이미지

| 조건 | 답변 | 실제 입력 이미지 수 |
|---|---|---:|
| Qwen | `[]` | 0 |
| Google text-only | `[]` | 0 |
| Google multimodal | `["Laura Davies"]` | 1 |

### Qwen

전달 이미지 없음.

### Google text-only

전달 이미지 없음.

### Google multimodal

<img src="images/432091348c329779a97b4f45.jpg" width="150" alt="실제 Reader 입력 이미지">

이 사례는 수정 전 `agent-read` 정책으로 측정했습니다. 현재 `search-linked` 자동 전달 코드의 성능 사례로 해석하지 않습니다.
