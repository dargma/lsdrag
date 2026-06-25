# INSTALL.md — 설치 가이드 (사람·기계 양립)

> 사람은 산문을 읽고, Claude Code는 bash 블록을 실행 단위로 인식한다.
> 각 단계 = **실행 블록 + 검증 블록**. 위→아래로, 다음 단계 전에 검증을 통과시켜라.
> 경로는 **하드코딩하지 말고 `config.yaml`에서** 읽는다(데이터 위치를 바꾸려면 `paths.*`만 고친다). 키는 env.

## 시작 전 설정
**경로·모델은 `config.yaml` 한 곳**(데이터 위치를 바꾸려면 `paths.*`만 고친다. 기본은 repo 내 `./data/`).
**키는 env만** — 평문 커밋 금지(`.env`는 gitignore).
```bash
cp .env.example .env        # 그 후 .env에 값 채우기
export UP_TOKEN="..."       # 파서(Upstage)        ← 또는 .env에 기입
export OPENAI_API_KEY="..." # Reader(GPT-4.1 mini)  ← 또는 .env에 기입
```
키 노출 시 즉시 재발급.

## 사전 요구사항
```bash
python rag/scripts/doctor.py --check deps --check keys     # C2, C3
```
✅ OK면 다음. ⚠️ 실패하면 출력된 조치를 따른다.

## 1단계: skill 패밀리 배치 (중첩 금지)
검색·DB빌드·파싱 세 슬래시 커맨드를 각각 설치한다.
```bash
mkdir -p ~/.claude/skills
cp -r rag       ~/.claude/skills/rag         # /rag        검색(질의)
cp -r rag-build ~/.claude/skills/rag-build   # /rag-build  DB 빌드 + 문서 add/remove/list
cp -r rag-parse ~/.claude/skills/rag-parse   # /rag-parse  파싱(IR 점검)
python rag/scripts/doctor.py --check skill                 # C1
```

## 2단계: 외부 API 연결 확인
```bash
python rag/scripts/doctor.py --check parser --check reader --check embedder   # C4, C5, C6
```
파서(Upstage)·Reader(GPT-4.1 mini)·임베더(A-RAG 내장) 각각 확인. 자체 GPU 서빙 없음.

## 3단계: 예시 문서 준비 (ARM 매뉴얼 — 이미지+테이블 있는 중간 1조각만)
전부 빌드하지 않는다. **figure와 table이 함께 들어있는 중간 본문 조각 1개**만 쓴다
(표지·목차는 표/그림이 없어 구조·이미지 질의 검증이 안 됨).
```bash
# 1) 후보 범위를 examples/parts/ 로 분할 (조각당 ≤100p)
python scripts/split_pdf.py --input examples/ARMv8-Reference-Manual.pdf \
    --ranges "1-60,1500-1560,3000-3060" --out examples/parts/
# 2) 표·그림이 있는 조각 1개를 골라 data/docs_in/ 에만 복사 (나머지는 두지 않음)
#    (1조각 파싱 후 IR의 block_type 분포로 table/figure 포함 확인 가능)
cp examples/parts/ARMv8-Reference-Manual_part2.pdf ./data/docs_in/
```

## 4단계: 인덱스 빌드
```bash
python -m src.indexing.build --config config.yaml          # 분할 PDF → 파싱 → IR → Page Index → 인덱스
python rag/scripts/doctor.py --check index                 # C7
```

## 5단계: 전체 확인 (스모크)
```bash
python rag/scripts/doctor.py --json    # C1~C9 전체. Claude Code가 --json 출력을 진행률로 중계
```
✅ 전부 통과하면 설치 완료. 새 세션에서 `/rag` 또는 사내 문서 질의로 사용.

## 문서 추가 / 제거 (설치 후 언제든)
```bash
python rag/scripts/docs.py add  <새_PDF>     # 분할(필요시)→파싱→인덱스에 추가
python rag/scripts/docs.py remove <doc_id>   # 인덱스·이미지에서 회수
python rag/scripts/docs.py list              # 현재 색인 문서
```
자연어로도 됩니다: "이 PDF 추가해줘", "X 문서 빼줘".

## 삭제 (uninstall)
```bash
python rag/scripts/uninstall.py              # skill 폴더만 제거
python rag/scripts/uninstall.py --purge-data # 인덱스/IR/이미지 데이터까지 (확인 후)
```

## 문제 해결 (사람용)
| 증상 | 원인 | 조치 |
|------|------|------|
| skill 안 잡힘 | 경로 중첩 | `~/.claude/skills/rag/SKILL.md`인지 확인 |
| 파서 실패(C4) | 키/네트워크 | `UP_TOKEN`·엔드포인트 확인 |
| Reader 실패(C5) | 키/모델명 | `OPENAI_API_KEY`·모델명 확인 |
| 임베더 실패(C6) | 모델 다운로드 | config `embedding.model`·캐시 확인 |
| 진행률 지저분 | 덮어쓰기 bar | `--json` 또는 plain 모드 |
