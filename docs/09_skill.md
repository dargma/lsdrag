# 09 — skill (/rag 노출 + 문서 관리 + 설치/삭제)

## 목적
완성된 엔진을 **스킬 패밀리**로 노출. 검색뿐 아니라 파싱·DB 빌드도 Claude Code 슬래시 커맨드로 쓴다.
"rag skill 설치해줘" 시 Claude Code가 **단계별 self-check하며** 설치 완료. 문서 추가/제거·손쉬운 삭제 포함.

## 스킬 패밀리 (슬래시 커맨드)
각 단계(파싱→DB빌드→검색)를 독립 슬래시 커맨드로. 모두 **얇은 껍데기**(엔진 `src/`를 config로 호출).

| 커맨드 | skill 폴더 | 하는 일 | 호출 엔진 |
|--------|-----------|---------|-----------|
| `/rag <질문>` | `rag/` | **검색**(질의→답변) | `src.agent.runner` |
| `/rag-build` | `rag-build/` | **DB 빌드**(파싱→IR→인덱스) + 증분 add/remove/list | `src.indexing.build` |
| `/rag-parse <pdf>` | `rag-parse/` | **파싱만**(문서→IR 확인, 빌드 전 점검) | `src.parser` |

- 세 skill은 각각 `.../skills/<name>/SKILL.md`로 설치(중첩 금지). 모두 같은 `config.yaml`·`.env`를 본다.
- 각 skill은 자기 `scripts/_bootstrap.py`로 engine.root를 resolve(자기완결, 개발 지시서 비의존 — 대원칙 5).
- 문서 add/remove/list는 `/rag-build`의 서브명령(`add/remove/list`)으로 통합(별도 docs 커맨드 불필요).

## 선행 / 산출물
- 선행: 08(엔진 동작 확인).
- 산출물: `rag/SKILL.md`, `rag/scripts/run.py`(질의), `rag/scripts/docs.py`(문서 관리),
  `rag/scripts/doctor.py`(설치 self-check), `rag/scripts/uninstall.py`(삭제), `INSTALL.md`.

## 패키징 경계 (개발 지시서는 배포 금지 — 대원칙 5)
배포되는 skill에는 **개발 전용 파일이 한 줄도 들어가면 안 된다.**
- **skill에 들어가는 것**: `rag/SKILL.md`, `rag/scripts/*`(run·docs·doctor·uninstall), 그리고 엔진(`src/`)·`config.yaml`을 가리키는 참조.
- **절대 들어가지 않는 것**: `CLAUDE.md`, `PROGRESS.md`, `docs/`, `_MASTER.md`, `tests/`, `examples/`, `benchmarks/`(내부 평가).
- `rag/` 스크립트는 **`docs/`·`CLAUDE.md`를 런타임에 읽지 않는다.** `src/`(엔진) + `config.yaml`에만 의존 → 지시서 없이 단독 동작.
- 엔진 위치는 `config.yaml`의 `engine.root`로 해석(절대경로 하드코딩 금지). skill 위치는 `${CLAUDE_SKILL_DIR}`.
- **게이트 G7(분리)**: `~/.claude/skills/rag/`에 개발 전용 파일이 0개. skill만으로 스모크 질의가 답을 낸다(docs/ 삭제해도 동작).

## 설계 결정
1. **SKILL.md는 얇은 껍데기.** 엔진 로직 금지. 발동 조건(description)과 "스크립트 실행" 지시만. frontmatter `name`/`description` 필수.
2. **description은 약간 적극적으로** — 사내 문서 검색·구조 질의·Figure/이미지 질의에서 발동하도록 트리거 맥락 명시.
3. **설치 경로 규칙(검증된 사실)**: `~/.claude/skills/rag/SKILL.md`(개인) 또는 `.claude/skills/rag/SKILL.md`(프로젝트).
   **중첩 금지** — 한 단계 더 깊으면 인식 안 됨. SKILL.md 없는 폴더는 무시.
4. **스크립트 경로는 `${CLAUDE_SKILL_DIR}`**. 데이터·엔진 경로는 `config.yaml`.
5. **doctor.py는 런타임 검증의 설치판**.

## 문서 관리 (추가/제거/목록 — 필수)
사용자가 문서를 새로 넣거나 빼면 skill이 엔진의 증분 API(05)를 호출해 인덱스를 최신으로 유지.
- `python ${CLAUDE_SKILL_DIR}/scripts/docs.py add <pdf>` — 분할(필요시)→파싱→인덱스 추가. 끝나면 추가된 doc_id·청크 수 출력.
- `python ${CLAUDE_SKILL_DIR}/scripts/docs.py remove <doc_id>` — 인덱스·Page Index·이미지에서 회수. 남은 문서 목록 출력.
- `python ${CLAUDE_SKILL_DIR}/scripts/docs.py list` — 현재 색인 문서.
- 자연어로도 발동: "이 PDF 추가해줘" / "X 문서 빼줘" → skill이 위 명령으로 연결.
- **사용 중 검증**: add 시 파싱 실패·중복 doc_id 명확 안내. remove 시 없는 doc_id면 "현재 목록은 …" 안내(크래시 금지).

## 손쉬운 삭제 (uninstall — 필수)
- `python ${CLAUDE_SKILL_DIR}/scripts/uninstall.py` — 한 명령으로 skill 제거.
  - 기본: skill 폴더(`.../skills/rag/`)만 제거. "skill 삭제됨" 명확 출력.
  - `--purge-data`: config가 가리키는 인덱스/IR/이미지 데이터까지 제거(확인 프롬프트 + 무엇을 지우는지 먼저 출력).
  - **안전**: 지우기 전 대상 경로를 보여주고, config 밖/시스템 경로는 거부. 침묵 삭제 금지.
- INSTALL.md에 "삭제하려면" 한 줄로 안내.

## 설치 Self-Check (doctor.py)
각 체크는 **독립적으로 실패를 진단**한다. "전체 실패"가 아니라 "몇 번에서 막혔는지 + 조치"를 출력.

| # | 체크 | 통과 조건 | 실패 시 안내 |
|---|------|-----------|-------------|
| C1 | skill 배치 | `.../skills/rag/SKILL.md` 존재·중첩 아님 | 위치/중첩 수정 |
| C2 | Python 의존성 | 필수 import 성공(sentence-transformers 포함) | `pip install` 안내 |
| C3 | API 키 | `UP_TOKEN`·`OPENAI_API_KEY` env 존재 | `.env` 설정 안내(평문 커밋 금지) |
| C4 | 파서 API | Upstage 소량 호출 OK | 키·네트워크·엔드포인트 확인 |
| C5 | Reader API | GPT-4.1 mini 1콜 응답 | 키·모델명·엔드포인트 확인 |
| C6 | 임베더 | A-RAG 임베더 모델 로드 OK | 모델명·다운로드 확인 |
| C7 | 인덱스 | 인덱스+Page Index 로드 가능 | 빌드 스크립트 안내 |
| C8 | 멀티모달 | Reader에 이미지 1건 전달 성공 | Reader 멀티모달 권한·형식 확인 |
| C9 | 스모크 | 알려진 질의 1개 답변 | 위 재확인 |

## 진행률 출력 (이중 모드)
Claude Code는 VT100 제어 시퀀스를 stripping → `\r` 덮어쓰기 bar는 줄마다 쏟아진다. 그래서:
- **기본(비TTY)**: 덮어쓰기 금지. 단계별 **한 줄 append**. 예: `[4/9] Reader ... OK`.
- **`--json`**: 각 단계 `{"step":4,"total":9,"name":"...","status":"ok"}` 한 줄. Claude Code가 읽어 "4/9 통과(44%)" 중계 → 가장 안정적.
- **`--rich`(TTY일 때만)**: 사람이 터미널에서 직접 볼 때만 화려한 bar.
- `sys.stdout.isatty()` False면 plain. 기본값으로 덮어쓰기형 bar 금지.

## 검증 — 작업 중 (Inline + 게이트)
- Inline: 각 check 함수 작성 즉시 단독 실행해 통과/실패가 정확히 갈리는지.
- **G1 진단력**: 의도적으로 망가뜨린 환경(예: `OPENAI_API_KEY` 제거)에서 doctor가 정확히 C3/C5에서 멈추고 올바른 조치 출력.
- **G2 정상 완주**: 의존 충족 시 C1~C9 통과, 스모크 실제 답변.
- **G3 발동**: 새 세션에서 사내 문서 질의 시 rag skill 발동·답변.
- **G4 안내형 설치**: "rag skill 설치해줘" 시 INSTALL.md 따라 단계별 진행, 실패 시 다음 단계로 안 넘어감.
- **G5 문서 관리**: add→list에 보임, remove→list에서 사라짐, 질의 결과에도 반영.
- **G6 삭제**: uninstall 후 skill 폴더 없음, `--purge-data` 시 데이터도 제거(확인 후), 시스템 경로 거부.

## 검증 — 사용 중 (런타임 가드)
- doctor의 각 체크 = 런타임 가드. 설치·질의·문서관리·삭제 어디서 막혀도 사람 말 + 조치로 드러난다.
