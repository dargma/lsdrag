"""Rebuild a bounded-pilot report from actual artifacts; never infer completion."""
import argparse
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
repo=Path(__file__).resolve().parents[2];destination=repo/'lab/reports/decision_policy_20261023';destination.mkdir(parents=True,exist_ok=True)
status=json.loads((run/'status.json').read_text());diag=json.loads((run/'representation_diagnostic.json').read_text())
summary=json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else None
curves=json.loads((run/'ppo_curve.json').read_text()) if (run/'ppo_curve.json').exists() else []
counts={name:len(list((run/name).glob('*.json'))) for name in ['branches','train_baseline','validation','test']}
records=[]
for folder in ['validation','test']:
 for path in (run/folder).glob('*.json'):
  ep=json.loads(path.read_text());traj=ep['trajectory'];keys=[json.dumps([t['tool_name'],t['arguments']],sort_keys=True) for t in traj]
  empty=[i for i,t in enumerate(traj) if t['tool_name'].endswith('search') and not t['result']]
  usage=ep['reader_calls'];usage=[u.get('usage',u) for u in usage]
  records.append({'group':folder,'policy':ep['policy'],'qid':ep['qid'],'em':ep['em'],'f1':ep['f1'],'empty_search':bool(empty),
   'empty_search_then_tool_then_correct':bool(empty and any(i<len(traj)-1 for i in empty) and ep['em']==1),
   'repeated_identical_tool_calls':len(keys)-len(set(keys)),
   'prefill_seconds':sum(u.get('prefill_seconds',0) for u in usage),'decode_seconds':sum(u.get('decode_seconds',0) for u in usage),
   'prefix_seconds':sum(u.get('prefix_seconds',0) for u in usage),'tool_seconds':sum(t['seconds'] for t in traj),
   'episode_seconds':ep['seconds']})
report={'run':str(run),'status':status,'artifact_counts':counts,'representation':diag,'ppo_curve':curves,'summary':summary,'posthoc_diagnostics':records,
 'limitations':['Single-seed small pilot; final20 cannot be reused for tuning.','Branch states are exposed train24; no heldout generalization from branch gains.','Success/cost auxiliary heads are not calibrated current-PPO success probabilities.','Observed wall times include diagnostic readout/CPU-copy overhead; no controlled speedup claim.','Attention reads32 context-bin means, not full-token attention.','Native-fast changes prefix constraints; not an exact native decoder replacement.','Outcome CE uses branch data; PPO additionally collects144 on-policy episodes. Any difference is not an equal-data algorithm-only comparison.']}
(destination/'pilot1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
lines=['# Decision policy v1.2 pilot1 결과', '',f"실제 상태: `{status['phase']}`. raw: `{run}`.", '', '## 입력·구조 진단', '', 'Teacher 행동 모방 지표이며 QA 성능이 아니다. train24/validation8, validation29상태.', '', '| 조건 | macro F1 | 조기 FINAL 오분류 | 파라미터(보조 출력 포함) |','|---|---:|---:|---:|']
for name,row in diag['models'].items():lines.append(f"| {name} | {row['macro_f1']:.4f} | {row['premature_final']} | {row['parameters']:,} |")
lines+=['',f"사전 규칙으로 선택: `{diag['selected']}`.",'','## 실제 학습 및 수집','',f"분기 episode {counts['branches']}, train baseline {counts['train_baseline']}, 완료 PPO 업데이트 {len(curves)}."]
for c in curves:lines.append(f"- PPO {c['iteration']+1}: {c['episodes']} episodes, {c['states']} 학습상태, 수집 중 F1 {c['train_f1']:.4f}. 이는 on-policy train 성능이다.")
if summary:
 lines+=['','## 동결 fresh20 평가','','| 정책 | F1 | EM | tool calls | 생성 tokens | 전체초(관측) | 오류 |','|---|---:|---:|---:|---:|---:|---:|']
 for name,row in summary['policies']['test'].items():lines.append(f"| {name} | {row['f1']:.4f} | {row['em']:.3f} | {row['tool_calls']:.2f} | {row['generated_tokens']:.2f} | {row['seconds']:.3f} | {row['errors']} |")
 lines+=['','질문 단위 paired bootstrap(탐색):']
 for name,row in summary['exploratory_ci'].items():lines.append(f"- {name}: F1차이 {row['mean']:+.4f}, 95% [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}], {row['wins']}승/{row['losses']}패.")
else:lines+=['','최종 평가 미완료. 정책 개선 결과를 아직 주장할 수 없다.']
lines+=['','## 해석 제한','']+['- '+s for s in report['limitations']]
lines+=['','원자료의 실패와 timeout도 포함한다. 구조 개선, RL 개선, 시간 개선과 신규성은 별개로 판단한다.']
(repo/'docs/JEV_DECISION_POLICY_PILOT1_REPORT_20261023.md').write_text('\n'.join(lines)+'\n')
print(status['phase'],counts)
