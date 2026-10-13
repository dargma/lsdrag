# Mem-Gallery · 대화에 나온 작품 이름

문항 `Architecture_Art_Culture_Exhibition_Technology:qa0068` · development. 모델·데이터 다운로드 없이 이 폴더의 JSON과 이미지를 열면 실행 현황을 확인할 수 있습니다.

**원문 질문:** What specific artwork mentioned in the conversation would be interesting to compare with the attached modern art image? Please return the artwork name.

**정답:** `"The Birth of Venus."`

**원 질문 첨부 이미지:** 있음. **이 질문 첨부 이미지의 실제 입력:** 없음. 검색 근거 이미지는 아래 별도로 표시한 실제 입력 기록이다.

**원 질문 이미지의 캡션:** 원 데이터에 있음. **이 질문 캡션의 실제 입력:** 없음. 검색 대상 이미지의 캡션은 검색용으로 사용하며 [캡션 목록](caption_inventory.json)에 기록했습니다.

- [원본 질문](raw/question.json) · [정답·주석](raw/gold.json) · [원 데이터 질문](raw/source_question.json)
- 전체 실행: [Qwen](predictions/B0.json) · [Google text-only](predictions/C1.json) · [Google multimodal](predictions/G1.json)
- [원문 청크](raw/chunks.json) · [원본 이미지 메타데이터](raw/figures.json) · [로컬 이미지 경로](figures_local.json) · [캡션 목록](caption_inventory.json)
- [파일 해시·출처](manifest.json). 기존 원본의 절대 경로는 출처 기록이며 `figures_local.json`의 `image_path`는 이 폴더 기준 상대 경로입니다.

## 실제 답변·입력 이미지

| 조건 | 답변 | 실제 입력 이미지 수 |
|---|---|---:|
| Qwen | `["Ancient paintings"]` | 0 |
| Google text-only | `["Impressionism (specifically, paintings of the same vase)"]` | 0 |
| Google multimodal | `["The Birth of Venus"]` | 5 |

### Qwen

전달 이미지 없음.

### Google text-only

전달 이미지 없음.

### Google multimodal

<img src="images/2bae1cc22715c7872e95d4d3.jpg" width="150" alt="실제 Reader 입력 이미지"> <img src="images/0c596eb8423ec7ab79692958.jpg" width="150" alt="실제 Reader 입력 이미지"> <img src="images/bdadb0a7d643a769cbe04658.jpg" width="150" alt="실제 Reader 입력 이미지"> <img src="images/1eb770d42d7a830222a83096.jpg" width="150" alt="실제 Reader 입력 이미지"> <img src="images/205b1a4478a172bb1186b02f.jpg" width="150" alt="실제 Reader 입력 이미지">

이 사례는 수정 전 `agent-read` 정책으로 측정했습니다. 현재 `search-linked` 자동 전달 코드의 성능 사례로 해석하지 않습니다.
