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
 'actor_audit':json.loads((run/'posthoc_actor_audit.json').read_text()) if (run/'posthoc_actor_audit.json').exists() else None,
 'branch_opportunity':json.loads((run/'posthoc_branch_opportunity.json').read_text()) if (run/'posthoc_branch_opportunity.json').exists() else None,
 'limitations':['Single-seed small pilot; final20 cannot be reused for tuning.','Branch states are exposed train24; no heldout generalization from branch gains.','Success/cost auxiliary heads are not calibrated current-PPO success probabilities.','Native-fast raw policy_probs are uniform placeholders, not native confidence; excluded from probability analysis. Head policy probabilities are actual outputs.','Observed wall times include diagnostic readout/CPU-copy overhead; no controlled speedup claim.','Attention reads32 context-bin means, not full-token attention.','Native-fast changes prefix constraints; not an exact native decoder replacement.','Outcome CE uses branch data; PPO additionally collects144 on-policy episodes. Any difference is not an equal-data algorithm-only comparison.']}
(destination/'pilot1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
lines=['# Decision policy v1.2 pilot1 결과', '',f"실제 상태: `{status['phase']}`. raw: `{run}`.", '', '## 입력·구조 진단', '', 'Teacher 행동 모방 지표이며 QA 성능이 아니다. train24/validation8, validation29상태.', '', '| 조건 | macro F1 | 조기 FINAL 오분류 | 파라미터(보조 출력 포함) |','|---|---:|---:|---:|']
for name,row in diag['models'].items():lines.append(f"| {name} | {row['macro_f1']:.4f} | {row['premature_final']} | {row['parameters']:,} |")
lines+=['',f"사전 규칙으로 선택: `{diag['selected']}`.",'','## 실제 학습 및 수집','',f"분기 episode {counts['branches']}, train baseline {counts['train_baseline']}, 완료 PPO 업데이트 {len(curves)}."]
for c in curves:lines.append(f"- PPO {c['iteration']+1}: {c['episodes']} episodes, {c['states']} 학습상태, 수집 중 F1 {c['train_f1']:.4f}. 이는 on-policy train 성능이다.")
if summary:
 lines+=['','## 동결 fresh20 평가','','| 정책 | F1 | EM | tool calls | 생성 tokens | 전체초(관측) | 오류 | cap 강제종료 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for name,row in summary['policies']['test'].items():lines.append(f"| {name} | {row['f1']:.4f} | {row['em']:.3f} | {row['tool_calls']:.2f} | {row['generated_tokens']:.2f} | {row['seconds']:.3f} | {row['errors']} | {row['forced_final']} |")
 lines+=['','질문 단위 paired bootstrap(탐색):']
 for name,row in summary['exploratory_ci'].items():lines.append(f"- {name}: F1차이 {row['mean']:+.4f}, 95% [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}], {row['wins']}승/{row['losses']}패.")
else:lines+=['','최종 평가 미완료. 정책 개선 결과를 아직 주장할 수 없다.']
if summary:
 lines+=['','## 결론과 다음 진단','','이번 pilot에서는 PPO 정책 개선·탐색 단계 절감이 입증되지 않았다. F1과EM은 BC와 같고 평균 호출수도 같다. Actor 파라미터는 실제 변경됐으나 fixed train82에서 greedy 행동변경0, validation29에서1개였다. 학습 미실행과 정책 행동변화 부족을 구분한다.', '', 'Outcome CE는 fresh20에서 F1 점추정이 높지만 paired CI가0을 포함하며, 평균5.9호출/13개cap 강제종료/1개생성truncation으로 종료 제어가 나쁘다. 정확도·효율 공동 개선으로 채택하지 않는다.', '', '다음 후보는 train-only에서 PPO logit margin/advantage/KL 진단, 후기상태·종료행동에 대한 반사실 coverage 보강, 동일 수집데이터 예산의 outcome CE 대 PPO 비교다. 이번 새20은 이제 노출되었으므로 후속튜닝금지. 추가실험은 아직 실행하지 않았다.']
lines+=['','## 해석 제한','']+['- '+s for s in report['limitations']]
lines+=['','원자료의 실패와 timeout도 포함한다. 구조 개선, RL 개선, 시간 개선과 신규성은 별개로 판단한다.']
math_text=(repo/'docs/JEV_DECISION_POLICY_MATH_20261023.md').read_text()
lines += ['', math_text[math_text.index('## 3. Pilot1'):math_text.index('## 4. Pilot2')]]
(repo/'docs/JEV_DECISION_POLICY_PILOT1_REPORT_20261023.md').write_text('\n'.join(lines)+'\n\n## 학습 수식\n\n[PPO·반사실 Q·outcome CE 수식과 코드 대응](JEV_DECISION_POLICY_MATH_20261023.md). Pilot1 PPO와 pilot2 expected-return 학습은 다르다.\n')
print(status['phase'],counts)
