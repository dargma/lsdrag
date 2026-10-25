"""POC S1: can frozen 9B decision-state features predict stop labels? Question-grouped CV on train48 branch outcomes (CPU)."""
import json
import re
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

run = Path(sys.argv[1])
states = torch.load(run/'states.pt', weights_only=False, map_location='cpu')
final, base = {}, {}
for f in (run/'episodes').glob('*.json'):
    m = re.match(r'(.+)_t(\d+)_a3_r0\.json$', f.name)
    if m:
        final[(m[1], int(m[2]))] = json.loads(f.read_text())
    elif f.name.endswith('_collector.json'):
        base[f.name[:-len('_collector.json')]] = json.loads(f.read_text())
X, T, groups, y_now, y_hopeless = [], [], [], [], []
for s in states:
    key = (s['qid'], s['turn'])
    if key not in final:
        continue
    pack = s['pack'].float().reshape(-1, 4096) if s['pack'].dim() > 1 else s['pack'].float()[None]
    X.append(pack.numpy())
    T.append(s['turn'])
    groups.append(s['qid'])
    y_now.append(int(final[key]['em'] == 1))
    y_hopeless.append(int(base[s['qid']]['em'] == 0))
X = np.stack(X)
T = np.array(T, dtype=float)[:, None]
groups = np.array(groups)
print('states', len(X), 'pack slots', X.shape[1], 'P(FINAL-now correct)=%.3f' % np.mean(y_now), 'P(hopeless)=%.3f' % np.mean(y_hopeless))


def cv_auc(features, y, C):
    y = np.array(y)
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=8).split(features, y, groups):
        sc = StandardScaler().fit(features[tr])
        clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(features[tr]), y[tr])
        scores[te] = clf.predict_proba(sc.transform(features[te]))[:, 1]
    return roc_auc_score(y, scores)


result = {}
for label, y in [('final_now_correct', y_now), ('question_hopeless', y_hopeless)]:
    row = {'turn_only': round(cv_auc(T, y, 1.0), 3)}
    for name, feats in [('last_token', X[:, 0]), ('all_slots', X.reshape(len(X), -1))]:
        for C in (0.001, 0.01):
            row[f'{name}_C{C}'] = round(cv_auc(np.hstack([feats, T]), y, C), 3)
    result[label] = row
    print(label, row)
(run/'poc_stop_probe.json').write_text(json.dumps(dict(n=len(X), questions=len(set(groups)), auc=result,
    note='GroupKFold(8) by question; train48 one-deviation states; labels from deterministic FINAL branches and base outcome. Diagnostic only.'), indent=2)+'\n')
