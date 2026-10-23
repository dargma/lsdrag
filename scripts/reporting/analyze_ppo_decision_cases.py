"""Trace first common-prefix divergence using saved probabilities; no new inference."""
from pathlib import Path
import json,math,collections
REPO=Path(__file__).resolve().parents[2];BASE=Path('/content/local_fast/lsdrag-text-v1');RUN=BASE/'results/decision_policy_ablation_20261023';NAMES=['keyword_search','semantic_search','read_chunk','FINAL']
def read(p):return json.loads(p.read_text())
def public_step(s):return {k:s[k] for k in ['tool_name','arguments','result']}
def main():
 m=read(RUN/'manifest.json');raw={q['id']:q for q in map(json.loads,(BASE/'source/musique/musique_ans_v1.0_dev.jsonl').read_text().splitlines()) if q['id'] in set(m['test'])};records=[]
 for seed in [1,2]:
  for qid in m['test']:
   a=read(RUN/f'episodes/anchor_{qid}.json');b=read(RUN/f'episodes/full_s{seed}_{qid}.json');first=None
   for i,(ad,bd) in enumerate(zip(a['decisions'],b['decisions'])):
    if ad['action']!=bd['action']:
     assert ad['legal']==bd['legal'];old=ad['action'];new=bd['action'];pa=ad['policy_probs'];pb=bd['policy_probs']
     first={'kind':'action_at_identical_public_prefix','turn':i,'anchor_action':NAMES[old],'ppo_action':NAMES[new],'anchor_probs':pa,'ppo_probs':pb,'new_vs_old_logodds_change':math.log(max(pb[new],1e-30)/max(pb[old],1e-30))-math.log(max(pa[new],1e-30)/max(pa[old],1e-30)),'last_observation':public_step(a['trajectory'][i-1]) if i else None,'anchor_next':public_step(a['trajectory'][i]) if i<len(a['trajectory']) else None,'ppo_next':public_step(b['trajectory'][i]) if i<len(b['trajectory']) else None};break
    if i<len(a['trajectory']) and i<len(b['trajectory']) and public_step(a['trajectory'][i])!=public_step(b['trajectory'][i]):
     first={'kind':'arguments_or_tool_result_before_action_divergence','turn':i,'anchor_step':public_step(a['trajectory'][i]),'ppo_step':public_step(b['trajectory'][i])};break
   q=raw[qid];gold_chunks={f'p{v["idx"]}' for v in q['paragraphs'] if v['is_supporting']}
   def seen(ep):return {v['chunk_id'] for t in ep['trajectory'] if isinstance(t['result'],list) for v in t['result'] if isinstance(v,dict) and 'chunk_id' in v}
   evidence={'gold_chunks':sorted(gold_chunks),'anchor_retrieved_gold':sorted(seen(a)&gold_chunks),'ppo_retrieved_gold':sorted(seen(b)&gold_chunks),'new_gold_chunks':sorted((seen(b)-seen(a))&gold_chunks),'anchor_cites':a['support_ids'],'ppo_cites':b['support_ids']}
   records.append({'qid':qid,'seed':seed,'question':q['question'],'gold':q['answer'],'anchor_answer':a['answer'],'ppo_answer':b['answer'],'anchor_f1':a['f1'],'ppo_f1':b['f1'],'delta':b['f1']-a['f1'],'calls_delta':b['tool_calls']-a['tool_calls'],'evidence':evidence,'first':first})
 summary={}
 for seed in [1,2]:
  rr=[r for r in records if r['seed']==seed];by=collections.defaultdict(lambda:collections.Counter())
  for r in rr:
   f=r['first'];kind='no_public_action_or_tool_difference' if not f else f['kind'];key=f"{f['anchor_action']} -> {f['ppo_action']}" if f and kind=='action_at_identical_public_prefix' else kind;by[key]['total']+=1;by[key]['win' if r['delta']>1e-9 else 'loss' if r['delta']< -1e-9 else 'tie']+=1
  summary[str(seed)]={k:dict(v) for k,v in by.items()}
 out=REPO/'lab/reports/decision_ppo_decision_cases_20261023';out.mkdir(parents=True,exist_ok=True);(out/'result.json').write_text(json.dumps({'summary':summary,'cases':records},ensure_ascii=False,indent=2)+'\n')
 lines=['# 개선·실패 문항의 최초 결정 분기','', '기존새40×2seed실행기록80쌍을모두분석. 추가LLM실행/재학습없음. 질문·도구이름·인자·관측이같은prefix에서처음다른action을찾고저장된실제head확률을비교한다. 인자/도구결과가먼저다르면같은state행동대조로분류하지않는다. 확률비의log차이는두행동상대logit변화이며정답확률이아니다.','', '| seed | 최초 변화 | 개수 | 개선 | 악화 | 동률 |','|---|---|---:|---:|---:|---:|']
 for seed,items in summary.items():
  for k,v in items.items():lines.append(f"| {seed} | {k} | {v['total']} | {v.get('win',0)} | {v.get('loss',0)} | {v.get('tie',0)} |")
 selected=[]
 for sign in [1,-1]:
  seen=set()
  for r in sorted([r for r in records if sign*r['delta']>1e-9],key=lambda r:(-abs(r['delta']),r['qid'],r['seed'])):
   if r['qid'] not in seen:selected.append(r);seen.add(r['qid'])
   if len(seen)==3:break
 lines+=['','## 사례: 개선·악화 각 최대3문항','', '효과절대값순·qid순으로선정한설명용사례이며대표성/인과효과증거가아니다. 모든80쌍은JSON에보존.']
 for r in selected:
  lines+=['',f"### {r['qid']} / seed{r['seed']}",'',f"질문: {r['question']}",f"정답: {r['gold']}",f"Anchor: {r['anchor_answer']} (F1 {r['anchor_f1']:.3f})",f"PPO: {r['ppo_answer']} (F1 {r['ppo_f1']:.3f})",f"정답근거chunk검색: anchor={r['evidence']['anchor_retrieved_gold']}, PPO={r['evidence']['ppo_retrieved_gold']}, 새근거={r['evidence']['new_gold_chunks']}"]
  f=r['first']
  if f and f['kind']=='action_at_identical_public_prefix':
   old=NAMES.index(f['anchor_action']);new=NAMES.index(f['ppo_action']);lines +=[f"최초분기 turn{f['turn']}: {f['anchor_action']}→{f['ppo_action']}; new/old logodds변화 {f['new_vs_old_logodds_change']:+.3f}.",f"두행동확률 anchor(old,new)=({f['anchor_probs'][old]:.3f},{f['anchor_probs'][new]:.3f}), PPO=({f['ppo_probs'][old]:.3f},{f['ppo_probs'][new]:.3f})."]
   for key,label in [('last_observation','직전관측'),('anchor_next','Anchor후속도구'),('ppo_next','PPO후속도구')]:
    if f[key]:lines.append(f"{label}: {json.dumps(f[key],ensure_ascii=False)[:1200]}")
  else:lines.append(f"동일prefix action분기없음/다른유형: {None if f is None else f['kind']}")
 lines+=['','## 해석 제한','', 'Goldsupport chunk가검색됐다는것은snippet에서필요한근거를모두읽고이해했다는뜻이아니다. 문서ID기반coverage와실제entailment를구분한다.','', '분기와최종답변차이는관련경로를보여주지만첫action하나의인과효과는아니다. 이후모든action·인자·관측이함께달라진다. 답변F1은문자열overlap지표여서표현/길이변화도점수변화에영향을준다. 개선사례만으로이해를주장하지않고같은전환이실패한사례도함께본다. 이자료를진행중PPO96의학습질문선정이나target에사용하지않음.']
 (REPO/'docs/JEV_DECISION_PPO_DECISION_CASES_20261023.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
