# v2.2 S1 결과 — pointer 품질 gate 미통과로 중단

576문항 teacher 수집과 KD 학습을 완료했지만, 사전 조건인 내부 검증
gold-present pointer support F1 ≥ 같은 상태 teacher F1을 충족하지 못했다.
Pointer **0.697980**, teacher **0.714141**,
차이 **-0.016162**. PLAN 개정6 및 Claude 실행 지침에 따라
S2~S4 추가 실행·자동 재학습·보상 탐색을 중단했다. 전체 GRPO 비교의 성공이나
완료를 뜻하지 않는다. 이번 단계의 산출물 보존과 gate-stop 보고를 완료한다.

## 실행 계약과 완료 범위

- Qwen3-4B-Instruct-2507, non-thinking, BF16, 모델 revision cdbee75f17c01a7cc42f958dc650907174af0554.
- MuSiQue mixed-hop train576(각 hop192), shared corpus101,962문서. KD fit459/internal validation117, qid 중복0.
- tool은 0 keyword_search / 1 semantic_search / 2 read_chunk / 3 FINAL.
- tool 선택 후 검색 인자·FINAL은 LLM이 생성. read는 관측된 미독 후보의 동적 pointer 선택을 사용하도록 구현했다.
- STOP은 첫 선택 전에 금지, 중복 금지, 최대8개/후보 소진 시 강제 종료. 후보 K는 실제 수집에서1~10.
- teacher action hard KD + gold-present pointer 지도이며 pure KD가 아니다. gold는 오프라인 지도/채점 전용.
- LLM frozen, tool의 사용되는 층과 pointer 학습, AdamW3e-4,100epoch, 내부검증 합산CE 최소 checkpoint 선택.
- S0 legacy dev192의 답·trajectory192/192 완전 재현. F1 .1934366 / EM .1458333 / calls2.47917 / 생성107.80729. 새 공통계약 N0와 혼동하지 않는다.
- CPU43 tests 통과. 작은 FP32 replay/gradient와 optimizer 복원 검사 포함. 실제4B full-FT BF16 replay/gradient는 미실행.

## 데이터 수집·감사

576 QA,2,065 비강제 학습 상태,358 유효 read 라벨을 저장했다.
gold-present296 / teacher-fallback62 / invalid teacher1 / non-read1706.
모든 후보의 관측 이력·입력 token span·라벨·유한 feature·qid 분할을 감사해 통과했다.
fit read280≥100; 내부검증 read2hop25/3hop26/4hop27≥각10으로 coverage 통과.

실패 QA1건(`4hop3__69818_38788_25582_21116`)에서 LLM이 중복 chunk ID를 생성했다.
스키마 검증이 거부했으며 QA/유효 tool 상태는 보존하고 해당 pointer 라벨만 제외했다.
실패 사례를 제거하여 성능을 높이지 않았다. 이 실패는 pipeline crash가 아닌 모델 출력 형식 실패다.
전체 생성64,696 tokens/입력2,984,599 tokens. 본 resume worker는1,576.27초이며
첫16 smoke 시간은 별도이므로 이 수치를 전체 비용으로 단정하지 않는다.

## KD 결과

선택 checkpoint: epoch index14(15번째 epoch), 내부검증 합산CE 1.238867,
tool teacher-label accuracy 0.930788(QA 정답률 아님).

| gold-present 내부검증 | 고유 qid | pointer support F1 | teacher support F1 |
|---|---:|---:|---:|
| 2hop | 23 | 0.724638 | 0.831884 |
| 3hop | 24 | 0.833333 | 0.762500 |
| 4hop | 19 | 0.494737 | 0.510526 |
| 전체 | 66 | 0.697980 | 0.714141 |

질문별 균등 가중. Pointer 우세16 / 열세17 / 동률33.
탐색적 qid-paired bootstrap95% 차이 구간 [-0.109596, 0.078283].
통계적으로 열등하다고 확정하지 않으며, gate의 사전 점추정 조건 미달이라는 판정이다.
Teacher-fallback12상태 ordered exact 0.500.
Gold-present 상태의 pointer 평균 선택수 1.030, gold target 1.258.
적게 선택하는 경향은 보이지만 이것만으로 전체 실패 원인을 단정하지 않는다.

100epoch 마지막의 pointer train CE는 0.070224까지 감소했으나
검증 합산CE는 1.677001로 선택 checkpoint보다 높다.
과적합 가능성을 시사하며, 마지막 epoch를 사후 선택하지 않았다. 실패 후 학습률·seed·epoch를 바꾸지 않았다.

## 미실행과 해석 제한

새 공통계약 N0/H0-P 자율 dev192 평가, 조건부 H0-G, S3 reward pilot,
실제 full-FT smoke, H-RL/N-RL의 rollout8×192×2iteration 및 final test320은 모두 **미실행**이다.
따라서 QA 정확도 유지, 호출10% 절감, 토큰 절감, native GRPO 대비 개선을 주장할 근거가 없다.
H0-G 트리거도 측정하지 않았으므로 실행하지 않았다. 빈 비교 칸은0이 아니다.
현재는 같은 teacher 상태에서 pointer를 검사했으며, 자율 trajectory 품질과 다르다.
내부검증59qid는 fit과 atomic family를 공유하므로 독립 일반화 검증으로 부르지 않는다.
공유GPU, seed1, gold 추가지도 등 제한을 유지한다.

기본 reward는 Q=.8EM+.2F1,
R=Q×(1-.10T/(T+128))-.15EM×N/8로 그대로 보존한다.
12계수 offline 진단과 최대3회×2update smoke는 허용된 계획이나 이번에는 실행하지 않았다.
성능이 나왔다는 이유로 보상 가중치를 최적화했다고 주장하지 않는다.

## 저장·재사용

- HF 비공개 Dataset: https://huggingface.co/datasets/sungkwang2/lsdrag-v22-generated
- episodes: train459/validation117; labels: train1646/validation419. `load_dataset` 두 config 재로딩 검증 완료.
- 원본 tensor: artifacts/teacher_states.pt, 정확한 BF16 특징·순서 보존. state_index로 labels와 매핑.
- [HF revision/원격 검증 기록](HF_DATASET.json), [teacher 감사](S1_TEACHER_AUDIT.json), [teacher Drive 백업](S1_BACKUP.json).
- KD 결과 원본: `/content/local_fast/lsdrag-text-v1/results/v22_kd_20261110`; head SHA256 `86b163fa2106a98bf66fb8ed3ab6027fab2c53df6a397923b2b077bce7d06781`.
- [기계 판독 보고서](REPORT_S1.json), [gate 누적](GATES.md), [사전 계획](PLAN.md), [비교 템플릿](COMPARISON_TEMPLATE.md).

추가 수집·학습은 자동으로 시작하지 않는다. 후속 검토 시 이번 실패를 근거로 별도 계획을 먼저 정해야 한다.

최종 KD checkpoint·보고서·소스·HF 메타데이터 Drive archive: [FINAL_BACKUP.json](FINAL_BACKUP.json).
