"""Rebuild the S1 gate-stop report from completed immutable evidence."""
import json
import statistics
from pathlib import Path

import numpy as np

from common import ROOT, REPO, save, sha


def main():
    kd = ROOT/'results/v22_kd_20261110'
    teacher = ROOT/'results/v22_teacher576_20261110'
    status = json.loads((kd/'status.json').read_text())
    assert status == dict(phase='gate_failed', gate='pointer_quality', pid=None)
    r = json.loads((kd/'summary.json').read_text())
    audit = json.loads((ROOT/'specs/v22/teacher_audit.json').read_text())
    assert audit['audit_passed'] and audit['coverage']['passed']
    assert sha(kd/'head.pt') == r['head_sha256']
    assert sha(teacher/'teacher_states.pt') == r['teacher_sha256'] == audit['teacher_sha256']
    checks = r['validation_checks']; gold = [c for c in checks if c['source'] == 'gold_present']
    fallback = [c for c in checks if c['source'] == 'teacher_fallback']
    by_q = {}
    for c in gold:
        by_q.setdefault(c['qid'], []).append(c)
    paired = [{k: statistics.mean(c[k] for c in cs) for k in ('pointer_support_f1', 'teacher_support_f1')} | {'qid': q}
              for q, cs in sorted(by_q.items())]
    deltas = np.array([c['pointer_support_f1']-c['teacher_support_f1'] for c in paired])
    bootstrap = np.random.default_rng(20261110).choice(deltas, (10000, len(deltas)), replace=True).mean(1)
    per_hop = {}
    for h in ('2hop', '3hop', '4hop'):
        rows = [c for c in paired if c['qid'].startswith(h)]
        per_hop[h] = dict(qids=len(rows), pointer_f1=statistics.mean(c['pointer_support_f1'] for c in rows),
                          teacher_f1=statistics.mean(c['teacher_support_f1'] for c in rows))
    selected = r['learning_curve'][r['selected_epoch']]
    result = dict(phase='stopped_at_S1_pointer_quality_gate', coverage=audit['coverage'],
        episodes=audit['episodes'], states=audit['records'], errors=audit['episodes_with_errors'],
        label_sources=audit['label_sources'], candidate_counts=audit['candidate_counts'],
        pointer_support_f1=r['pointer_support_f1'], teacher_support_f1=r['teacher_support_f1'],
        delta=float(deltas.mean()), internal_paired_bootstrap95_exploratory=np.quantile(bootstrap,[.025,.975]).tolist(),
        gold_validation_qids=len(paired), wins=int((deltas>0).sum()), losses=int((deltas<0).sum()), ties=int((deltas==0).sum()),
        per_hop=per_hop, fallback_states=len(fallback), fallback_ordered_exact=statistics.mean(c['exact'] for c in fallback),
        pointer_selected_mean=statistics.mean(len(c['predicted']) for c in gold),
        gold_target_mean=statistics.mean(len(c['target']) for c in gold),
        selected_checkpoint=selected, last_epoch=r['learning_curve'][-1],
        not_run=['S2 common-contract N0/H0-P dev192','conditional H0-G','S3 group8 reward pilot',
                 'full-FT update smoke','H-RL and N-RL GRPO','i1/i2 dev192 and final test320'],
        head_sha256=r['head_sha256'], teacher_sha256=r['teacher_sha256'],
        evidence={str(p):sha(p) for p in [kd/'summary.json',kd/'status.json',kd/'spec.json',teacher/'summary.json',ROOT/'specs/v22/teacher_audit.json']})
    save(REPO/'codex/REPORT_S1.json',result)
    hop_lines='\n'.join(f"| {h} | {v['qids']} | {v['pointer_f1']:.6f} | {v['teacher_f1']:.6f} |" for h,v in per_hop.items())
    ci=result['internal_paired_bootstrap95_exploratory']
    report=f'''# v2.2 S1 결과 — pointer 품질 gate 미통과로 중단

576문항 teacher 수집과 KD 학습을 완료했지만, 사전 조건인 내부 검증
gold-present pointer support F1 ≥ 같은 상태 teacher F1을 충족하지 못했다.
Pointer **{r['pointer_support_f1']:.6f}**, teacher **{r['teacher_support_f1']:.6f}**,
차이 **{result['delta']:.6f}**. PLAN 개정6 및 Claude 실행 지침에 따라
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

선택 checkpoint: epoch index{r['selected_epoch']}(15번째 epoch), 내부검증 합산CE {selected['validation_ce']:.6f},
tool teacher-label accuracy {selected['tool_validation_accuracy']:.6f}(QA 정답률 아님).

| gold-present 내부검증 | 고유 qid | pointer support F1 | teacher support F1 |
|---|---:|---:|---:|
{hop_lines}
| 전체 | {len(paired)} | {r['pointer_support_f1']:.6f} | {r['teacher_support_f1']:.6f} |

질문별 균등 가중. Pointer 우세{result['wins']} / 열세{result['losses']} / 동률{result['ties']}.
탐색적 qid-paired bootstrap95% 차이 구간 [{ci[0]:.6f}, {ci[1]:.6f}].
통계적으로 열등하다고 확정하지 않으며, gate의 사전 점추정 조건 미달이라는 판정이다.
Teacher-fallback12상태 ordered exact {result['fallback_ordered_exact']:.3f}.
Gold-present 상태의 pointer 평균 선택수 {result['pointer_selected_mean']:.3f}, gold target {result['gold_target_mean']:.3f}.
적게 선택하는 경향은 보이지만 이것만으로 전체 실패 원인을 단정하지 않는다.

100epoch 마지막의 pointer train CE는 {r['learning_curve'][-1]['pointer_train_ce']:.6f}까지 감소했으나
검증 합산CE는 {r['learning_curve'][-1]['validation_ce']:.6f}로 선택 checkpoint보다 높다.
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
- KD 결과 원본: `{kd}`; head SHA256 `{r['head_sha256']}`.
- [기계 판독 보고서](REPORT_S1.json), [gate 누적](GATES.md), [사전 계획](PLAN.md), [비교 템플릿](COMPARISON_TEMPLATE.md).

추가 수집·학습은 자동으로 시작하지 않는다. 후속 검토 시 이번 실패를 근거로 별도 계획을 먼저 정해야 한다.
'''
    (REPO/'codex/REPORT_S1.md').write_text(report)
    print(json.dumps({k:result[k] for k in ['phase','delta','per_hop','not_run']},indent=2))


if __name__ == '__main__':
    main()
