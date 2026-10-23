"""Freeze conditional expanded training and independent extra validation, not test tuning."""
import json,re,hashlib
from pathlib import Path
from run_decision_policy_pilot import ROOT,REPO,sha,save
REG=REPO/'lab/registry/decision_state_policy_iteration_20261023'
def read(p):return json.loads(p.read_text())
def main():
 old=read(REPO/'lab/registry/decision_state_outcomes_20261023/manifest.json');hm=read(ROOT/'results/decision_state_heads_20261023/manifest.json');sel=read(ROOT/'results/decision_state_heads_20261023/final_candidate_selection_before_seed2.json');name=sel['primary'];spec=hm['candidates'][name];exposed=set()
 for folder in [REPO/'lab',ROOT/'results']:
  for p in folder.rglob('*.json'):exposed.update(re.findall(r'\b[234]hop__\d+_\d+\b',p.read_text()))
 data={n:[json.loads(l) for l in (ROOT/f'source/musique/musique_ans_v1.0_{n}.jsonl').open()] for n in ['train','dev']};families={d['id'] for q in data['train']+data['dev'] if q['id'] in exposed for d in q['question_decomposition']};fresh=[];used=set()
 for q in sorted(data['train']+data['dev'],key=lambda q:hashlib.sha256(('policy-iteration:'+q['id']).encode()).hexdigest()):
  f={d['id'] for d in q['question_decomposition']}
  if q['answerable'] and len(f)==2 and q['id'] not in exposed and not f&(families|used):fresh.append(q['id']);used.update(f)
  if len(fresh)==128:break
 assert len(fresh)==128,('Not enough fresh families',len(fresh))
 m=old.copy();m.update(seed=20261051,train=old['train']+fresh[:96],validation=old['validation']+fresh[96:],extra_train=fresh[:96],extra_validation=fresh[96:],reference=spec['checkpoint'],reference_name=name,reference_spec=spec,budget_episodes=3264,max_seconds=7200,max_generated=1800000,max_inputs=60000000,target={'f1_delta':.05,'generated_ratio':.7},purpose='conditional policy iteration after cost-weight plateau; old96+new96 refreshed under selected policy; original test160 unchanged',trigger='Execute only if primary two-seed validation mean fails either raised target; no final test read')
 sources=['scripts/run_decision_state_policy_iteration.py','scripts/fit_decision_policy_iteration.py','scripts/evaluate_decision_policy_iteration.py','scripts/fit_decision_state_heads.py','src/lsdrag/decision_state_learning.py','src/lsdrag/decision_policy.py','src/lsdrag/decision_policy_recovery.py','src/lsdrag/decision_policy_followup.py','src/lsdrag/text_integrated.py','src/lsdrag/text_integrated_aligned.py','prompts/text_rag_integrated_v5.txt'];m['source_sha256']={p:sha(REPO/p) for p in sources};m['input_sha256']=old['input_sha256']|{spec['checkpoint']:sha(Path(spec['checkpoint']))}
 assert not (REG/'manifest.json').exists();save(REG/'manifest.json',m);print('Frozen conditional train192 / validation64 / same untouched test160')
if __name__=='__main__':main()
