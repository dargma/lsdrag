"""CPU-only audit of frozen argument selection follow-up; no policy tuning."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lsdrag.text_integrated import slots_for

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/content/local_fast/lsdrag-text-v1')
RUN = ROOT / 'results/improvement_v3_argument_selection_20261017'
ORIGIN = ROOT / 'results/improvement_v3_merged_20261017'
TEACHER = ROOT / 'results/text_integrated_20261016_v5_fast_only'
OUT = REPO / 'lab/reports/text_integrated_20261016'
summary = json.loads((RUN / 'summary.json').read_text())
manifest = json.loads((RUN / 'manifest.json').read_text())
ck = torch.load(ORIGIN / 'head.pt', map_location='cpu', weights_only=True)
assert summary['policies']['always_generate']['n'] == summary['policies']['read_selection']['n'] == 20

# Evaluate slot imitation at the same frozen teacher validation states, even
# where the independently predicted action would not choose read_chunk.
split = json.loads((REPO / 'lab/registry/text_integrated_20261016/improvement_v2_split.json').read_text())
cm = np.zeros((5, 5), dtype=int)
fixed = []
for qid in split['validation']:
    records = json.loads((TEACHER / 'teacher' / f'{qid}.json').read_text())['records']
    features = np.load(TEACHER / 'teacher' / f'{qid}.npz')['features']
    for index, (r, f) in enumerate(zip(records, features)):
        if r['action'] != 'read_chunk':
            continue
        scores = ck['slot_weight'][2] @ torch.tensor(f, dtype=torch.float32) + ck['slot_bias'][2]
        options = slots_for(r['options'], 'read_chunk')
        scores[len(options):] = -torch.inf
        predicted, target = int(scores.argmax()), r['slot']
        cm[target, predicted] += 1
        fixed.append({'qid': qid, 'state_index': index, 'target': target, 'predicted': predicted,
                      'selected_arguments': options[predicted]['arguments'],
                      'teacher_arguments': r['teacher_arguments']})
assert fixed
fixed_stats = {'n_read_states': len(fixed), 'confusion_target_rows_prediction_columns': cm.tolist(),
               'slot_order': ['GENERATE', 'candidate1', 'candidate2', 'candidate3', 'candidate4'],
               'accuracy': float(np.trace(cm)/cm.sum()),
               'always_generate_slot_accuracy': sum(r['target'] == 0 for r in fixed)/len(fixed),
               'predicted_select': sum(r['predicted'] > 0 for r in fixed),
               'teacher_representable_select': sum(r['target'] > 0 for r in fixed),
               'correct_selected_slot': sum(r['predicted'] > 0 and r['predicted'] == r['target'] for r in fixed),
               'note': 'Teacher argument exact-match imitation, not optimality. Conditional on teacher read action at frozen states; not autonomous rollout accuracy.',
               'states': fixed}

rows, audits, hashes = [], [], {}
for qid in manifest['qids']:
    es = {}
    for policy in manifest['policies']:
        path = RUN / 'evaluation' / policy / f'{qid}.json'
        ep = json.loads(path.read_text())
        hashes[str(path.relative_to(RUN))] = hashlib.sha256(path.read_bytes()).hexdigest()
        es[policy] = ep
        assert ep['tool_calls'] == len(ep['trajectory'])
        rows.append({k: ep[k] for k in ['qid', 'policy', 'f1', 'em', 'support_f1', 'tool_calls',
                                      'interaction_turns', 'read_calls', 'selected_read_calls',
                                      'generated_read_calls', 'argument_generation_calls', 'argument_tokens',
                                      'completion_tokens', 'prompt_tokens', 'forced_prefix_tokens', 'seconds']})
    off, on = es['always_generate'], es['read_selection']
    for index, t in enumerate(on['trajectory']):
        if not t['argument_reused']:
            continue
        prior_equal = index <= len(off['trajectory']) and all(
            (a['tool_name'], a['arguments'], a['result']) == (b['tool_name'], b['arguments'], b['result'])
            for a, b in zip(on['trajectory'][:index], off['trajectory'][:index]))
        compare = off['trajectory'][index] if index < len(off['trajectory']) else None
        audits.append({'qid': qid, 'step': index, 'selected_arguments': t['arguments'],
                       'prior_observations_equal': prior_equal,
                       'off_action_at_same_step': compare['tool_name'] if compare else None,
                       'generated_arguments_off': compare['arguments'] if compare else None,
                       'same_arguments': t['arguments'] == compare['arguments'] if compare and prior_equal else None,
                       'f1_off': off['f1'], 'f1_on': on['f1'],
                       'answer_off': off['answer'], 'answer_on': on['answer']})

assert len(rows) == 40 and len(audits) == summary['policies']['read_selection']['selected_read_total']
summary.update(manifest=manifest, input_sha256=hashes, fixed_validation_read_slot=fixed_stats,
               selected_read_audit=audits,
               definitions={'tool_steps': 'completed primitive calls; FINAL excluded',
                            'interaction_turns': 'head decision attempts plus forced FINAL if any',
                            'latency': manifest['timing']})
(OUT / 'improvement_v3_argument_selection.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
with (OUT / 'improvement_v3_argument_selection.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]),lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)

off, on = (summary['policies'][p] for p in manifest['policies'])
lines = ['# v3 인자 선택 ON/OFF 추가 평가', '',
         '사용자 요청으로 같은 frozen 통합 checkpoint와 기존 평가20문항을 두 조건에서 재실행했다. '
         '재학습·튜닝 없음. 기존20문항 결과를 본 뒤 추가한 post-hoc 탐색 평가이며 새 heldout이 아니다. '
         '원래 primary와 결과는 보존한다. OFF 답변·도구·인자·결과 trajectory가 원 실행과20/20 일치했다.', '',
         'ON: 같은 prefill의 head가 read 인자의 GENERATE 또는 관측 후보1..4를 선택. '
         '후보 선택이면 즉시 실행하고 GENERATE면 cached 생성. keyword/semantic 인자는 두 조건 모두 생성한다.', '',
         '| 지표 | 선택 OFF: 항상 생성 | 선택 ON: 후보/생성 선택 |',
         '|---|---:|---:|']
for title, key, scale, fmt in [
    ('답변 F1 (%)', 'f1', 100, '.2f'), ('EM (%)', 'em', 100, '.2f'),
    ('근거 F1 (%)', 'support_f1', 100, '.2f'), ('평균 tool steps', 'tool_calls', 1, '.2f'),
    ('평균 interaction turns (FINAL 포함)', 'interaction_turns', 1, '.2f'),
    ('평균 인자 생성 호출', 'argument_generation_calls', 1, '.2f'),
    ('평균 인자 생성 tokens', 'argument_tokens', 1, '.2f'),
    ('평균 전체 생성 tokens', 'completion_tokens', 1, '.2f'),
    ('평균 강제 prefix tokens', 'forced_prefix_tokens', 1, '.2f'),
    ('평균 입력 tokens', 'prompt_tokens', 1, '.2f'),
    ('평균 episode latency (s)', 'seconds', 1, '.3f'),
    ('20문항 latency 합계 (s)', 'latency_total_seconds', 1, '.3f')]:
    lines.append(f'| {title} | {off[key]*scale:{fmt}} | {on[key]*scale:{fmt}} |')
lines += [f"| read 전체 / 인자 선택 실행 수 | {off['read_total']} / {off['selected_read_total']} | {on['read_total']} / {on['selected_read_total']} |",
          f"| 오류 episode | {off['errors']} | {on['errors']} |", '',
          f"read15회 중8회(53.33%)에서 생성을 생략했다. 전체47tool calls 대비17.02%다. "
          f"인자 tokens는{(1-on['argument_tokens']/off['argument_tokens'])*100:.2f}% 줄었지만, "
          '답변 F1/EM이 각각5.00 percentage points 떨어졌다. ON 기준0승·19동률·1패이며 '
          'step/turn 평균은 동일하다. 생성 최소화 기능은 실행됐고 품질 유지에는 실패했다.', '',
          '## 인자 선택 자체의 진단', '',
          f"고정 validation teacher read 상태 {len(fixed)}개의 slot 정확도는 "
          f"{fixed_stats['accuracy']*100:.2f}% (항상 GENERATE의 slot 정확도 "
          f"{fixed_stats['always_generate_slot_accuracy']*100:.2f}%). "
          f"예측 선택{fixed_stats['predicted_select']} / teacher 인자와 동일 후보 존재{fixed_stats['teacher_representable_select']} / "
          f"선택 slot 정답{fixed_stats['correct_selected_slot']}. 전체5×5 confusion과 각 상태는 JSON에 보존했다. "
          'teacher exact-match 모방 정확도이며 답변 효용·최적 선택 지표는 아니다.', '',
          'ON에서 실제 선택된8회 모두의 선택 인자와 OFF 생성 인자, 분기 전 관측 일치 여부를 JSON에 기록했다. '
          '오답으로 바뀐 문항은 `2hop__151455_120171`: OFF 답변1986 → ON1985. '
          'OFF는 Acornsoft의 p15(운영 종료1986)를 읽었고, ON은 Castle of Riddles의 p16(게임 판매1985 언급)을 골랐다. 선택 직전 관측은 동일했다. 실제 선택8회 중3회는 OFF 인자와 같고5회는 달랐다. '
          '선택이 문서 내용을 바꿀 수 있으므로 인자 생략 횟수만으로 성공으로 판정하지 않는다.', '',
          '## 측정 범위와 재현', '',
          'latency는 episode 내 제어·인자·답변 생성·검색(query embedding 포함)·Python 처리 시간이다. '
          '모델 로딩/corpus 사전 embedding/채점/파일 저장은 제외한다. 조건별 순서를 교대했으나 '
          '동일 상태 timing 통제나 반복측정을 하지 않았다. 공유 GPU 실행이며 관측 시간 차이를 speedup 입증으로 해석하지 않는다.', '',
          f"- run: `{RUN}`", f"- head SHA256: `{manifest['head_sha256']}`",
          '- 실행: `PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /content/local_fast/lsdrag-text-v1/experiment-env/bin/python scripts/evaluate_v3_argument_selection.py --output <새 local-fast 경로>`',
          '- 집계: `PYTHONPATH=src /content/local_fast/lsdrag-text-v1/experiment-env/bin/python scripts/report_v3_argument_selection.py`',
          '- 검색 인자 선택과 미래 남은 step의 가치 학습은 아직 구현/평가한 범위에 포함되지 않는다.']
assert all(p['off_matches_original_answer'] and p['off_matches_original_trajectory'] for p in summary['pairs'])
(OUT / 'improvement_v3_argument_selection.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'policies': summary['policies'], 'fixed_read_slot': {k:v for k,v in fixed_stats.items() if k != 'states'},
                  'selected_read_audit': audits}, ensure_ascii=False, indent=2))
