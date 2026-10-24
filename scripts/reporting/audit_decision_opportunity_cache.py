"""Train-only cache feasibility audit for stopping/value-head learning."""
import json,hashlib
from pathlib import Path
import torch
from lsdrag.decision_state_learning import action_values
ROOT=Path('/content/local_fast/lsdrag-text-v1')
REPO=Path(__file__).resolve().parents[2]

def main():
    torch.set_num_threads(2)
    run=ROOT/'results/decision_shared_outcomes192_20261023'
    m=json.loads((run/'manifest.json').read_text())
    records=torch.load(run/'states.pt',weights_only=True,map_location='cpu')
    assert {r['qid'] for r in records}==set(m['train'])
    assert not {r['qid'] for r in records}&set(m['validation']+m['test'])
    utility,q,c=action_values(records,.03)
    legal=torch.stack([r['legal'] for r in records]);continuing=legal.clone();continuing[:,3]=False
    assert legal[:,3].all() and continuing.any(-1).all()
    delta=utility.masked_fill(~continuing,-torch.inf).max(-1).values-utility[:,3]
    positive=q.masked_fill(~legal,0).max(-1).values>0
    rawdelta=q.masked_fill(~continuing,-torch.inf).max(-1).values-q[:,3]
    summary={'states':len(records),'questions':len(set(r['qid'] for r in records)),
      'states_with_any_positive_quality':int(positive.sum()),
      'quality_continue_better':int((rawdelta>1e-8).sum()),'quality_stop_better':int((rawdelta < -1e-8).sum()),
      'positive_quality_states_utility_continue_better':int(((delta>1e-8)&positive).sum()),
      'positive_quality_states_utility_stop_better':int(((delta < -1e-8)&positive).sum()),
      'all_zero_quality_states':int((~positive).sum()),
      'samples_per_legal_action':sorted(set(sum(s['action']==a for s in r['samples']) for r in records for a in range(4) if r['legal'][a])),
      'states_sha256':hashlib.sha256((run/'states.pt').read_bytes()).hexdigest(),
      'limitations':'Train-only, two suffix samples/action under fixed BC; noisy off-policy outcomes, not current-policy counterfactual ground truth. No validation/test records read.'}
    out=REPO/'lab/reports/decision_opportunity_value_20261024';out.mkdir(parents=True,exist_ok=True)
    (out/'cache_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
