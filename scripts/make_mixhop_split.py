"""Frozen mixed-hop split: existing 2hop split + equal numbers of 3hop and 4hop questions.

All questions come from MuSiQue answerable train (like the 2hop split). 3hop/4hop questions are sampled
stratified by sub-type (3hop1/3hop2, 4hop1/2/3) proportional to availability, disjoint across splits.
train 192+192+192, validation 64+64+64, test 160+80+80. Selection reads only ids, never answers.
"""
import json
import numpy as np
from common import ROOT, REPO

SEED = 20261103
SIZES = {'train': 192, 'validation': 64, 'test': 80}

def main():
    base = json.loads((REPO/'configs/split.json').read_text())
    ids = [json.loads(l)['id'] for l in (ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open()]
    rng = np.random.default_rng(SEED); out = {k: list(v) for k, v in base.items()}
    for hop in ['3hop', '4hop']:
        pools = {}
        for q in ids:
            t = q.split('__')[0]
            if t.startswith(hop):
                pools.setdefault(t, []).append(q)
        total = sum(len(v) for v in pools.values()); need = sum(SIZES.values())
        quota = {t: int(round(need*len(v)/total)) for t, v in sorted(pools.items())}
        quota[max(quota, key=quota.get)] += need-sum(quota.values())
        drawn = []
        for t in sorted(pools):
            drawn += list(rng.choice(sorted(pools[t]), quota[t], replace=False))
        drawn = list(rng.permutation(drawn)); i = 0
        for split, n in SIZES.items():
            out[split] += [str(x) for x in drawn[i:i+n]]; i += n
    allids = [q for v in out.values() for q in v]; assert len(allids) == len(set(allids))
    out['note'] = ('mixed-hop: 2hop part identical to configs/split.json; 3hop/4hop sampled from MuSiQue answerable train, '
                   f'stratified by sub-type, seed {SEED}')
    (REPO/'configs/split_mixhop.json').write_text(json.dumps(out, indent=1)+'\n')
    for k in SIZES:
        c = {}
        for q in out[k]:
            c[q.split('__')[0]] = c.get(q.split('__')[0], 0)+1
        print(k, len(out[k]), c)

if __name__ == '__main__':
    main()
