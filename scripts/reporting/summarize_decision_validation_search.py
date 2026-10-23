"""Consolidate validation-only search without selecting on test outcomes."""
import json
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];ROOT=Path('/content/local_fast/lsdrag-text-v1');HEADS=ROOT/'results/decision_state_heads_20261023'
def read(p):return json.loads(p.read_text())
def main():
 runs=['decision_state_validation1_20261023','decision_state_validation2_20261023','decision_state_validation3_20261023','decision_state_validation_seed2_20261023'];rows={};origins={};budgets={}
 for name in runs:
  run=ROOT/'results'/name
  if not (run/'summary.json').exists():continue
  assert read(run/'status.json')['phase']=='complete';s=read(run/'summary.json');assert s['split']=='validation';budgets[name]=s['counts']
  for k,v in s['policies'].items():
   if k in rows:
    assert rows[k]==v,'Repeated baseline differs: inspect determinism instead of overwriting'
   rows[k]=v;origins[k]=name
 if (ROOT/'results/decision_grpo_continuation_20261023/summary.json').exists():
  c=read(ROOT/'results/decision_grpo_continuation_20261023/summary.json');rows['ordinary_grpo_continuation']=c['policies']['grpo'];origins['ordinary_grpo_continuation']='decision_grpo_continuation_20261023'
 base=rows['grpo_baseline'];derived={k:{'f1_delta':v['f1']-base['f1'],'generated_ratio':v['generated_tokens']/base['generated_tokens'],'joint_point_target':v['f1']-base['f1']>=.05 and v['generated_tokens']/base['generated_tokens']<=.7} for k,v in rows.items()}
 result={'policies':rows,'comparisons':derived,'origin_runs':origins,'validation_evaluation_cost':budgets,'n':32,'selection_used_this_data':True,'not_confirmatory':True,'target':{'f1_delta':.05,'generated_ratio':.7}}
 if (HEADS/'final_candidate_selection_before_seed2.json').exists():result['selected']=read(HEADS/'final_candidate_selection_before_seed2.json')
 out=REPO/'lab/reports/decision_state_validation_search_20261023';out.mkdir(parents=True,exist_ok=True);(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 lines=['# 입력·구조·비용 선호 validation 탐색','', '동일32문항. 후보선택에사용한자료이므로최종목표달성/일반화증거로사용하지않음. 최신목표F1+5%p/생성token−30%.','', '| 후보 | F1 | 생성token | F1 차이(pp) | 생성 절감(%) | 공동 point 목표 |','|---|---:|---:|---:|---:|---|']
 for k,v in rows.items():
  d=derived[k];lines.append(f"| {k} | {v['f1']*100:.2f} | {v['generated_tokens']:.2f} | {d['f1_delta']*100:+.2f} | {(1-d['generated_ratio'])*100:+.2f} | {d['joint_point_target']} |")
 lines+=['','warmstart는같은구조를train에서GRPO분포로distill한대조. 이후counterfactual outcome정책학습효과와구조변경효과를혼동하지않는다. 모든구조에새LLM수집을반복하지않고동일191state1428episode를공유했다. seed2는별도수집/학습이다.','', '예측확률전체를사용하는기대효용목적은명시적가치모델이나미래상태예측모델이아니다. 생성token직접비용후보와기존호출/생성/입력proxy비용후보를구분한다.']
 (out/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(derived,indent=2))
if __name__=='__main__':main()
