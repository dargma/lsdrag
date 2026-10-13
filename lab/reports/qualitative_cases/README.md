# 정성 평가 원본 데이터

각 사례 폴더에 원본 질문·정답, 세 조건의 전체 실행 JSON, 관련 원문·이미지·캡션과 파일 해시를 포함했습니다. 별도 데이터 준비 없이 GitHub에서 열거나 clone 후 확인할 수 있습니다. 전체 벤치마크 재실행을 위한 인덱스 묶음은 아닙니다.

- [M3DocVQA: 펭귄이 등장하는 영화 포스터](happy-feet/README.md) · `8d3bf560c35d4b2798db545f1b0a7b94`
- [M3DocVQA: 매우 짧은 금발의 선수 찾기](laura-davies/README.md) · `be0baecf90a40b1ba8230d9c771f75ca`
- [Mem-Gallery: 대화에 나온 작품 이름](birth-of-venus/README.md) · `Architecture_Art_Culture_Exhibition_Technology:qa0068`
- [Mem-Gallery: 관련 사진을 빠짐없이 찾아 개수 세기](foxglove-count/README.md) · `Baking_Dessert_Daily_Life_Skill:qa0042`

파일 해시와 사례 점수는 Python 표준 라이브러리만으로 확인할 수 있습니다:

```sh
python lab/reports/qualitative_cases/verify.py
```
