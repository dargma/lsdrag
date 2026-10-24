"""Train-only paired-state diagnostic; no counterfactual generalization claim."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from evaluate_decision_value_heads import load_head
from lsdrag.decision_policy import distribution
from lsdrag.decision_state_learning import action_values
ROOT=Path('/content/local_fast/lsdrag-text-v1/results')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(2)
    source=ROOT/'decision_shared_outcomes192_20261023';m=read(source/'manifest.json')
    records=torch.load(source/'states.pt',weights_only=True,map_location='cpu')
    assert {r['qid'] for r in records}==set(m['train'])
    assert not {r['qid'] for r in records}&set(m['validation']+m['test'])
    x=torch.stack([r['pack'] for r in records]).float();legal=torch.stack([r['legal'] for r in records])
    utility,q,c=action_values(records,.03);cont=legal.clone();cont[:,3]=False
    gap=utility[:,1]-utility[:,3]  # fixed semantic_search versus FINAL contrast
    groups={}
    for i,r in enumerate(records):groups.setdefault(r['qid'],[]).append(i)
    pairs=[];identical_masks=0
    for indices in groups.values():
        if len(indices)<2:continue
        indices.sort(key=lambda i:records[i]['turn']);a,b=indices[0],indices[-1]
        identical_masks+=int(torch.equal(legal[a],legal[b]))
        if legal[a,1] and legal[b,1] and legal[a,3] and legal[b,3] and abs(float(gap[b]-gap[a]))>1e-8:pairs.append((a,b))
    candidates=read(ROOT/'decision_opportunity_value_20261024/manifest.json')['candidates'].copy()
    reg=read(REPO/'lab/registry/decision_selected_head_grpo192_20261024_r1/manifest.json')
    candidates['selected_before']=reg['reference_spec']
    for name,suffix in [('selected_grpo','r1'),('selected_gate','zero_gate')]:
        cp=ROOT/f'decision_selected_head_grpo192_20261024_{suffix}/grpo_iteration1.pt'
        candidates[name]=dict(kind='structure',inputs='multi',body='mlp',checkpoint=str(cp),sha256=sha(cp))
    stats={}
    for name,spec in candidates.items():
        assert sha(Path(spec['checkpoint']))==spec['sha256']
        h=load_head(spec)
        with torch.no_grad():
            logits=h(x,legal)[0];p=distribution(logits,legal).probs
        probability=(logits[:,1]-logits[:,3]).sigmoid().numpy();delta_p=np.array([probability[b]-probability[a] for a,b in pairs]);delta_g=np.array([float(gap[b]-gap[a]) for a,b in pairs])
        stats[name]=dict(mean_final_probability=float(p[:,3].mean()),std_final_probability=float(p[:,3].std(unbiased=False)),same_question_shared_semantic_final_pairs=len(pairs),direction_agreement=float((delta_p*delta_g>0).mean()) if pairs else None,mean_absolute_two_action_probability_change=float(np.abs(delta_p).mean()) if pairs else None,checkpoint_sha256=spec['sha256'])
    out=REPO/'lab/reports/decision_opportunity_value_20261024'
    report=dict(states=len(records),later_always_less_semantic_preference_direction_agreement=float(np.mean([gap[b]<gap[a] for a,b in pairs])),identical_full_legal_mask_pairs=identical_masks,contrast='semantic_search versus FINAL, conditional two-action probability; not full policy continue probability',train_only=True,cache_sha256=sha(source/'states.pt'),policies=stats,limitations='Same-question first/last states, both contrasted actions legal, differing observed semantic-minus-FINAL utility. Initial full-mask-matched probe had zero pairs, so its evidence is unavailable; using fixed-action log-odds removes the changed action-set denominator. Fitting data reused: in-sample response only. No shuffled-input intervention, no causal proof or heldout evidence. Two fixed-BC suffix samples/action are noisy; step/history and evidence change together. State-independent distribution has zero pairwise change, not a trained baseline evaluation.')
    (out/'state_response_probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
