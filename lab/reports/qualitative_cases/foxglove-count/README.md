# Mem-Gallery · 사진 누락과 개수 오류

문항 `Baking_Dessert_Daily_Life_Skill:qa0042` · search-linked100. 모델·데이터 다운로드 없이 이 폴더의 JSON과 이미지를 열면 실행 현황을 확인할 수 있습니다.

**원문 질문:** How many foxglove flower images did Maya share in total during the autumn outing? Please answer using an Arabic numeral (e.g., “2” instead of “Two”).

**정답:** `"5"`

**원 질문 첨부 이미지:** 없음. **이 질문 첨부 이미지의 실제 입력:** 없음. 검색 근거 이미지는 아래 실제 입력 기록에 표시한다.

**원 질문 이미지의 캡션:** 원래 없음 · 실제 입력하지 않음. 검색 자료의 캡션은 사용.

- [원본 질문](raw/question.json) · [정답·주석](raw/gold.json) · [원 데이터 질문](raw/source_question.json)
- 전체 실행: [Qwen](predictions/B0.json) · [Google text-only](predictions/C1.json) · [Google multimodal](predictions/G1.json)
- [원문 청크](raw/chunks.json) · [원본 이미지 메타데이터](raw/figures.json) · [로컬 이미지 경로](figures_local.json) · [캡션 목록](caption_inventory.json)
- [파일 해시·출처](manifest.json). 기존 원본의 절대 경로는 출처 기록이며 `figures_local.json`의 `image_path`는 이 폴더 기준 상대 경로입니다.

## 실제 답변·입력 이미지

| 조건 | 답변 | 실제 입력 이미지 수 |
|---|---|---:|
| Qwen | `["3"]` | 4 |
| Google text-only | `["4"]` | 5 |
| Google multimodal | `["5"]` | 5 |

### Qwen

<img src="images/0c4131f4a377154fcb72db3e.jpg" width="150" alt="실제 Reader 입력 이미지"> <img src="images/b7836a8e59aa2985e801e636.jpg" width="150" alt="실제 Reader 입력 이미지"> <img src="images/ad38fbbc80201b3a7cc50464.jpg" width="150" alt="실제 Reader 입력 이미지"> <img src="images/013efb5e6dd26448231d5df7.jpg" width="150" alt="실제 Reader 입력 이미지">

### Google text-only

<img src="images/0c4131f4a377154fcb72db3e.jpg" width="150" alt="실제 Reader 입력 이미지"> <img src="images/b7836a8e59aa2985e801e636.jpg" width="150" alt="실제 Reader 입력 이미지"> <img src="images/52e6dff530845ebe0d3134f3.jpg" width="150" alt="실제 Reader 입력 이미지"> <img src="images/ad38fbbc80201b3a7cc50464.jpg" width="150" alt="실제 Reader 입력 이미지"> <img src="images/013efb5e6dd26448231d5df7.jpg" width="150" alt="실제 Reader 입력 이미지">

### Google multimodal

<img src="images/0c4131f4a377154fcb72db3e.jpg" width="150" alt="실제 Reader 입력 이미지"> <img src="images/b7836a8e59aa2985e801e636.jpg" width="150" alt="실제 Reader 입력 이미지"> <img src="images/52e6dff530845ebe0d3134f3.jpg" width="150" alt="실제 Reader 입력 이미지"> <img src="images/ad38fbbc80201b3a7cc50464.jpg" width="150" alt="실제 Reader 입력 이미지"> <img src="images/013efb5e6dd26448231d5df7.jpg" width="150" alt="실제 Reader 입력 이미지">
