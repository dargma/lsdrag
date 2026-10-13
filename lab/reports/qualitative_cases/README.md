# 정성 평가 원본 데이터

각 사례 폴더에 원본 질문·정답, 세 조건의 전체 실행 JSON, 관련 원문·이미지·캡션과 파일 해시를 포함했습니다. 별도 데이터 준비 없이 GitHub에서 열거나 clone 후 확인할 수 있습니다. 전체 벤치마크 재실행을 위한 인덱스 묶음은 아닙니다.

- [M3DocVQA: 캡션 없는 표지의 물체 식별](aria-windmill/README.md) · `8ddcc98208a9a3476003255054870ba1`
- [M3DocVQA: 고릴라가 등장하는 앨범 표지 찾기](gorilla-album/README.md) · `299e684293fffb0f9dca76a1092c83f9`
- [Mem-Gallery: 사진 누락과 개수 오류](foxglove-count/README.md) · `Baking_Dessert_Daily_Life_Skill:qa0042`
- [Mem-Gallery: 조명이 켜진 사진의 이미지 ID 선택](colosseum-lighting/README.md) · `Global_Travel_Sustainable_Fashion_Design:qa0042`

이 묶음은 100문항 자동 전달(`search-linked`) 실행에서 새로 선정한 사례입니다. Mem-Gallery의 성공 사례만 보여주지만 전체 EM은 Qwen 24%, Google multimodal 22%이므로 전체 개선의 근거로 해석하지 않습니다.

```sh
python lab/reports/qualitative_cases/verify.py
```
