"""Post-hoc trajectory description, not a causal action ablation."""
from pathlib import Path
import json,collections
REPO=Path(__file__).resolve().parents[2];BASE=Path('/content/local_fast/lsdrag-text-v1/results')
results=[]
for name in ['decision_ppo_tempered_20261023','decision_ppo_tempered_seed2_20261023']:
 r=BASE/name;m=json.loads((r/'manifest.json').read_text());pairs=[];counts={p:collections.Counter() for p in ['tempered_anchor','ppo']}
 for q in m['test']:
  a=json.loads((r/f'episodes/test_tempered_anchor_{q}.json').read_text());b=json.loads((r/f'episodes/test_ppo_{q}.json').read_text());delta=b['f1']-a['f1']
  for p,e in [('tempered_anchor',a),('ppo',b)]:counts[p].update(d['action'] for d in e['decisions'])
  pairs.append({'qid':q,'delta':delta,'initial_changed':a['decisions'][0]['action']!=b['decisions'][0]['action'],'sequence_changed':[d['action'] for d in a['decisions']]!=[d['action'] for d in b['decisions']],'calls_delta':b['tool_calls']-a['tool_calls']})
 results.append({'run':name,'wins':sum(p['delta']>1e-9 for p in pairs),'losses':sum(p['delta']< -1e-9 for p in pairs),'ties':sum(abs(p['delta'])<=1e-9 for p in pairs),'initial_changed':sum(p['initial_changed'] for p in pairs),'sequence_changed':sum(p['sequence_changed'] for p in pairs),'wins_with_fewer_calls':sum(p['delta']>1e-9 and p['calls_delta']<0 for p in pairs),'action_counts':counts,'pairs':pairs})
p=REPO/'lab/reports/decision_ppo_two_seeds_20261023/action_changes.json';p.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
lines=['# PPO 행동 변화 사후 분석','', '기존두seed의동일40문항raw trajectory를분석. 새학습/추론없음. action0=keyword_search,1=semantic_search,2=read_chunk,3=FINAL.','', '| 지표 | Seed1 | Seed2 |','|---|---:|---:|']
for k in ['wins','losses','ties','initial_changed','sequence_changed','wins_with_fewer_calls']:lines.append(f"| {k} | {results[0][k]} | {results[1][k]} |")
lines+=['','40문항합계: 초기정책 keyword40/semantic21/read31. PPOseed1 keyword40/semantic37/read5;seed2 keyword41/semantic31/read18. FINAL은모두40.','', '처음선택보다관측이쌓인후행동경로의차이가더많다. 그러나서로다른경로의후속상태는동일하지않으므로이집계로state-conditioned결정의인과효과를입증할수없다. 전체적으로read감소/semantic증가성향만으로설명되는지추가대조가필요하다.','', 'F1이개선되면서호출수도줄어든문항은두seed모두0개. 평균F1개선과평균호출감소가같은문항에서발생했다는해석은틀리다. 집계상품질/효율동시개선과개별문항내동시개선을구분한다. 다음검증후보는전역actionbias와STOP-only대조이며이test40으로조건튜닝하지않는다.']
(REPO/'docs/JEV_DECISION_PPO_ACTION_CHANGES_20261023.md').write_text('\n'.join(lines)+'\n')
