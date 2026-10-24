"""Compare train-only suffix targets; seeds/policies differ, no causal claim."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/content/local_fast/lsdrag-text-v1/results')
OUT = Path('lab/reports/decision_current_policy_refresh48_20261024')

def means(record, field):
    return {a: float(np.mean([s[field] for s in record['samples'] if s['action'] == a]))
            for a in sorted({s['action'] for s in record['samples']})}

def best_set(values):
    maximum = max(values.values())
    return {a for a,v in values.items() if abs(v-maximum) < 1e-8}

def reliability(records):
    gaps = []
    overlaps = []
    positive_overlaps = []
    positive = 0
    stopping_signs = []
    informative_stopping_signs = []
    for r in records:
        by_action = {}
        for s in r['samples']:
            by_action.setdefault(s['action'], {})[s['rep']] = s['f1']
        assert all(set(v) == {0,1} for v in by_action.values())
        gaps.extend(abs(v[0]-v[1]) for v in by_action.values())
        if 3 in by_action and len(by_action)>1:
            differences = [max(v[i] for k,v in by_action.items() if k != 3)-by_action[3][i] for i in (0,1)]
            agreement = bool(np.sign(differences[0]) == np.sign(differences[1]))
            stopping_signs.append(agreement)
            if max(abs(x) for x in differences) > 0:
                informative_stopping_signs.append(agreement)
        a,b = [best_set({k:v[i] for k,v in by_action.items()}) for i in (0,1)]
        overlaps.append(bool(a & b))
        if any(s['f1'] > 0 for s in r['samples']):
            positive += 1
            positive_overlaps.append(bool(a & b))
    return {'states':len(records), 'any_positive_states':positive,
            'nonzero_stopping_contrast_states':len(informative_stopping_signs),
            'nonzero_stopping_contrast_sign_agreement':float(np.mean(informative_stopping_signs)),
            'action_pair_mean_absolute_f1_gap':float(np.mean(gaps)),
            'repeat_best_action_set_overlap_rate':float(np.mean(overlaps)),
            'positive_state_repeat_best_action_set_overlap_rate':float(np.mean(positive_overlaps))}

def main():
    paths = {'old': ROOT/'decision_shared_outcomes192_20261023/states.pt',
             'fresh': ROOT/'decision_current_policy_refresh48_20261024/states.pt'}
    data = {k:torch.load(p,map_location='cpu',weights_only=False) for k,p in paths.items()}
    old = {(r['qid'],r['prefix_sha256']):r for r in data['old']}
    pairs = [(old[(r['qid'],r['prefix_sha256'])],r) for r in data['fresh']
             if (r['qid'],r['prefix_sha256']) in old]
    assert pairs
    changes=[]
    for a,b in pairs:
        qa,qb=means(a,'f1'),means(b,'f1')
        assert set(qa)==set(qb)
        changes.append({'qid':b['qid'],'turn':b['turn'],
            'best_quality_set_overlap':bool(best_set(qa)&best_set(qb)),
            'mean_absolute_action_f1_change':float(np.mean([abs(qa[k]-qb[k]) for k in qa])),
            'old_f1':qa,'fresh_f1':qb,'old_generated':means(a,'generated'),
            'fresh_generated':means(b,'generated')})
    report={'source_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
        'reliability':{k:reliability(v) for k,v in data.items()},
        'matched_prefix_states':len(pairs),
        'matched_initial_states':sum(b['turn']==0 for _,b in pairs),
        'matched_best_quality_set_overlap_rate':float(np.mean([x['best_quality_set_overlap'] for x in changes])),
        'matched_mean_absolute_action_f1_change':float(np.mean([x['mean_absolute_action_f1_change'] for x in changes])),
        'comparisons':changes,
        'limitations':['Training-only, two stochastic samples per action.',
            'Different suffix policy and random seeds: policy effect and sampling noise are confounded.',
            'Best-set overlap counts ties and is optimistic; not policy accuracy.',
            'Old pooled states and fresh states differ in sampling; compare identical prefix hashes only.']}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'target_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='comparisons'},indent=2))
if __name__=='__main__':main()
