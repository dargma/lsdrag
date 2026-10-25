"""Stage 1 (CPU): simulate stop policies with recomposition (R) on train48 states; question-grouped CV probes.

Input cost at stop t = agent prompts up to and including the decision capture at t + R prompt.
Idea 7 (draft features) additionally pays an R prompt at every visited state before stopping.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path('/content/local_fast/lsdrag-text-v1')
OUT = ROOT/'results/poc_stage1_state_answers_20261025'
rows = [json.loads(l) for l in (OUT/'rows.jsonl').open()]
packs = {(s['qid'], s['turn']): s['pack'].float().reshape(-1)[:4096].numpy()
         for s in torch.load(ROOT/'results/qwen35_9b_outcomes48_20261024/states.pt', weights_only=False, map_location='cpu')}
byq = defaultdict(list)
for r in rows:
    byq[r['qid']].append(r)
for q in byq.values():
    q.sort(key=lambda r: r['t'])
# label: R_all correct now; next-step label for VOI
for q in byq.values():
    for i, r in enumerate(q):
        r['y_now'] = int(r['R_all']['em'] == 1)
        r['y_next'] = int(q[min(i+1, len(q)-1)]['R_all']['em'] == 1)
data = [r for r in rows if (r['qid'], r['t']) in packs]
groups = np.array([r['qid'] for r in data])


def features(r, draft):
    f = [packs[(r['qid'], r['t'])], [r['t'], r['observed']]]
    if draft:
        f.append([r['R_all']['first_logp'], r['R_all']['mean_logp']])
    return np.concatenate([np.asarray(x, dtype=np.float32) for x in f])


def cv_scores(label, draft, C=0.001):
    X = np.stack([features(r, draft) for r in data])
    y = np.array([r[label] for r in data])
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=8).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        p[te] = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
    return {(r['qid'], r['t']): float(v) for r, v in zip(data, p)}, roc_auc_score(y, p)


def simulate(stop_fn, draft_cost=False, cond='R_all'):
    em = inp = 0
    for qid, q in byq.items():
        stop = next((r for r in q if stop_fn(r)), q[-1])
        em += stop[cond]['em']
        inp += stop['cumulative_agent_prompt']+stop[cond]['prompt_tokens']
        if draft_cost:
            inp += sum(r['R_all']['prompt_tokens'] for r in q if r['t'] < stop['t'])
    return dict(EM=int(em), inputs=round(inp/len(byq)))


n = len(byq)
result = {'questions': n, 'states': len(rows), 'states_with_features': len(data)}
result['agent'] = dict(EM=sum(q[0]['agent_em'] for q in byq.values()), inputs=round(sum(q[0]['agent_prompt_tokens'] for q in byq.values())/n))
result['R_at_agent_end'] = simulate(lambda r: False)
result['oracle_stop_R'] = simulate(lambda r: r['y_now'] == 1)
result['R_gold_at_agent_end (idea4 oracle)'] = simulate(lambda r: False, cond='R_gold')
result['oracle_stop_R_gold (idea4 oracle)'] = simulate(lambda r: r['R_gold']['em'] == 1, cond='R_gold')
p_now, auc_now = cv_scores('y_now', False)
p_now7, auc_now7 = cv_scores('y_now', True)
p_next, auc_next = cv_scores('y_next', False)
result['auc'] = dict(y_now=round(auc_now, 3), y_now_with_draft=round(auc_now7, 3), y_next=round(auc_next, 3))
curves = {}
for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    curves[f'idea2_threshold_{tau}'] = simulate(lambda r, t=tau: p_now.get((r['qid'], r['t']), 0) >= t)
    curves[f'idea7_draft_threshold_{tau}'] = simulate(lambda r, t=tau: p_now7.get((r['qid'], r['t']), 0) >= t, draft_cost=True)
for c in [-0.1, 0.0, 0.05, 0.1]:
    curves[f'idea2_voi_margin_{c}'] = simulate(lambda r, c=c: (r['qid'], r['t']) in p_now and p_now[(r['qid'], r['t'])] >= p_next[(r['qid'], r['t'])]-c and p_now[(r['qid'], r['t'])] >= 0.3)
result['curves'] = curves
result['note'] = 'train48 bootstrap trajectories; CV probes (GroupKFold 8 by question); EM out of 48; inputs = mean per question. Rough POC.'
(OUT/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
