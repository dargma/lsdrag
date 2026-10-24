"""Freeze combined current-policy data and one conservative action update."""
import json
from pathlib import Path
import torch
from run_decision_policy_pilot import ROOT,REPO,save,sha
from lsdrag.decision_current_policy_targets import current_policy_targets

def read(p):return json.loads(p.read_text())
def main():
 names=['decision_h3_policy_iteration_collect24_20261024','decision_h3_policy_coverage24b_20261024'];records=[];parents={};verifications={}
 for name in names:
  r=ROOT/'results'/name;v=REPO/'lab/reports'/name/'independent_verification.json';m=read(v);assert m['verified'];assert sha(r/'states.pt')==m['states_sha256']
  archive=read(REPO/'lab/registry'/name/'archive.json');assert sha(Path(archive['path']))==m['archive_sha256']==archive['sha256']
  records+=torch.load(r/'states.pt',weights_only=True);parents[str(r/'states.pt')]=sha(r/'states.pt');verifications[str(v)]=sha(v)
 assert len({(r['qid'],r['prefix_sha256']) for r in records})==len(records) and len({r['qid'] for r in records})==48
 labels,evidence=current_policy_targets(records);assert evidence['changed_questions']>=3
 name='decision_h3_conservative48_20261024';reg=REPO/'lab/registry'/name;reg.mkdir(exist_ok=False)
 data=ROOT/'results/decision_h3_conservative48_inputs_20261024';data.mkdir(exist_ok=False);torch.save(records,data/'states.pt');save(data/'provenance.json',dict(parents=parents,verifications=verifications))
 verification=reg/'combined_verification.json';save(verification,dict(verified=True,parents=parents,verifications=verifications,states=len(records),questions=48))
 plan=read(REPO/'lab/registry/decision_h3_policy_iteration1_20261024/plan.json');plan.update(seed=20261075,fresh=str(data/'states.pt'),verification=str(verification),expected_labels=labels,kl_limit=.01,unsupported_change_limit=.02,source_policy='Combined disjoint48train H3rollin/H3suffix outcomes; conservative outputlayer step')
 for p in [data/'states.pt',data/'provenance.json',verification]:plan['input_sha256'][str(p)]=sha(p)
 plan['input_sha256'].update(parents);plan['input_sha256'].update(verifications)
 plan['source_sha256']={p:sha(REPO/p) for p in plan['source_sha256']}
 for p in ['scripts/fit_decision_h3_conservative48.py','scripts/prepare_h3_conservative48_fit.py','src/lsdrag/decision_conservative_update.py','docs/JEV_H3_COVERAGE_CONSERVATIVE_UPDATE_20261024.md']:plan['source_sha256'][p]=sha(REPO/p)
 save(reg/'plan.json',plan);save(reg/'target_evidence.json',evidence);print(json.dumps({k:v for k,v in evidence.items() if k!='moves'},indent=2))
if __name__=='__main__':main()
