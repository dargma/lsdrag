"""Offline check of a step-level (process) reward before running it, on saved head-only GRPO rollouts.

Questions answered from existing train rollouts (gold support labels are train-only information):
 1. need:      how many decisions get zero learning signal under the current reward F1*0.8^calls?
 2. direction: mean advantage per (hop, action) under current vs process reward.
 3. effect:    within each 8-rollout group, do trajectories without wasted calls reach >= F1 with fewer calls?
 4. explore:   on 3/4hop, do longer trajectories (>=4 calls) do better, and how rare are they?
Process reward: per call +ALPHA*new_gold/|gold| - COST; outcome F1 at FINAL. Step advantage =
return-to-go minus the group mean return-to-go at the same step index.
Usage: python scripts/validate_process_reward.py --run mh_headonly_C1
"""
import argparse, collections, glob, json
from pathlib import Path
import numpy as np
from common import ROOT, REPO

ACT = ['keyword_search', 'semantic_search', 'read_chunk', 'FINAL']

def gold_map(ids):
    cid = {}
    for line in (ROOT/'corpora/musique_shared_20261023/documents.jsonl').open():
        d = json.loads(line); cid[(d['title'], d['text'])] = d['chunk_id']
    g = {}
    for line in (ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open():
        q = json.loads(line)
        if q['id'] in ids:
            g[q['id']] = {cid[(p['title'], p['paragraph_text'])] for p in q['paragraphs'] if p['is_supporting']}
    return g

def steps_of(e, gold):
    """Per call: (action, new gold count, new chunk count); then FINAL."""
    """A search is useful if it surfaces a gold paragraph not seen before; a read is useful if it reads
    a gold paragraph in full for the first time (snippets are only 300 characters). Credit counts are in
    gold-paragraph units for searches and half units for first full reads."""
    seen, gs, gread, out = set(), set(), set(), []
    for h in e['trajectory']:
        res = h['result']
        if h['tool_name'] == 'read_chunk':
            ids = set(h['arguments'].get('chunk_ids', [])); new = (ids & gold)-gread; gread |= ids & gold
            out.append((h['tool_name'], .5*len(new), len(ids-seen))); seen |= ids
        else:
            ids = {r['chunk_id'] for r in res if isinstance(r, dict) and 'chunk_id' in r} if isinstance(res, list) else set()
            out.append((h['tool_name'], len((ids & gold)-gs), len(ids-seen))); seen |= ids; gs |= ids & gold
    return out, len(gs)/max(1, len(gold))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--alpha', type=float, default=.5)
    ap.add_argument('--cost', type=float, default=.05); a = ap.parse_args()
    G = collections.defaultdict(list)
    for p in glob.glob(str(ROOT/'results'/a.run/'episodes'/'train_i*_r*_*.json')):
        e = json.loads(Path(p).read_text()); it = Path(p).name.split('_')[1]; G[(it, e['qid'])].append(e)
    gold = gold_map({q for _, q in G})
    R = {'groups': len(G), 'trajectories': sum(len(v) for v in G.values())}
    cur_zero = proc_zero = n_dec = 0
    dir_cur = collections.defaultdict(list); dir_proc = collections.defaultdict(list)
    effect = collections.defaultdict(list); explore = collections.defaultdict(list); covF1 = []
    for (it, q), g in G.items():
        hop = q[:4]; info = [steps_of(e, gold[q]) for e in g]
        rc = np.array([e['f1']*.8**e['tool_calls'] for e in g]); ac = rc-rc.mean()
        # process returns-to-go per step
        rtg = []
        for e, (st, cov) in zip(g, info):
            r = [a.alpha*ng/max(1, len(gold[q]))-a.cost for _, ng, _ in st]+[e['f1']]
            rtg.append(np.cumsum(r[::-1])[::-1])
        T = max(len(x) for x in rtg)
        base = [np.mean([x[t] for x in rtg if len(x) > t]) for t in range(T)]
        for i, (e, (st, cov)) in enumerate(zip(g, info)):
            acts = [s[0] for s in st]+['FINAL']
            for t, act in enumerate(acts):
                n_dec += 1; cur_zero += abs(ac[i]) < 1e-9; ap_ = rtg[i][t]-base[t]; proc_zero += abs(ap_) < 1e-9
                ctx = 'after_full_cov' if t > 0 and sum(s[1] for s in st[:t] if s[0] != 'read_chunk') >= len(gold[q]) else 'before_full_cov'
                key = (hop, act if act != 'FINAL' else f'FINAL[{ctx}]')
                dir_cur[key].append(ac[i]); dir_proc[key].append(ap_)
            wasted = sum(1 for _, ng, nc in st if ng == 0)
            effect[(hop, wasted == 0)].append((e['f1'], e['tool_calls'], q, it))
            explore[(hop, e['tool_calls'] >= 4)].append(e['f1'])
        if np.ptp([c for _, c in info]) > 0 and np.ptp([e['f1'] for e in g]) > 0:
            covF1.append(np.corrcoef([c for _, c in info], [e['f1'] for e in g])[0, 1])
    R['decisions'] = n_dec
    R['1_need'] = {'zero_signal_decisions_current': cur_zero/n_dec, 'zero_signal_decisions_process': proc_zero/n_dec,
                   'within_group_corr(coverage,F1)': float(np.mean(covF1)), 'groups_used_for_corr': len(covF1)}
    R['2_direction'] = {f'{h} {k}': {'n': len(dir_cur[(h, k)]), 'adv_current': round(float(np.mean(dir_cur[(h, k)])), 4),
                                     'adv_process': round(float(np.mean(dir_proc[(h, k)])), 4)} for h, k in sorted(dir_cur)}
    # 3. paired within group: trajectories without wasted calls vs with, same question & iteration
    paired = collections.defaultdict(list)
    for (it, q), g in G.items():
        info = [steps_of(e, gold[q]) for e in g]
        clean = [(e['f1'], e['tool_calls']) for e, (st, _) in zip(g, info) if all(ng > 0 for _, ng, _ in st)]
        dirty = [(e['f1'], e['tool_calls']) for e, (st, _) in zip(g, info) if any(ng == 0 for _, ng, _ in st)]
        if clean and dirty:
            paired[q[:4]].append((np.mean([x[0] for x in clean])-np.mean([x[0] for x in dirty]), np.mean([x[1] for x in clean])-np.mean([x[1] for x in dirty])))
    R['3_effect_clean_minus_wasteful_same_group'] = {h: {'groups': len(v), 'dF1': round(float(np.mean([x[0] for x in v])), 3), 'dcalls': round(float(np.mean([x[1] for x in v])), 2)} for h, v in sorted(paired.items())}
    allcalls = [e['tool_calls'] for g in G.values() for e in g]
    wasted_calls = sum(sum(1 for _, ng, _ in steps_of(e, gold[q])[0] if ng == 0) for (it, q), g in G.items() for e in g)
    R['3_wasted_call_share'] = wasted_calls/sum(allcalls)
    R['4_explore'] = {f'{h} calls>=4': {'share': len(explore[(h, True)])/(len(explore[(h, True)])+len(explore[(h, False)])),
                                        'F1_long': round(float(np.mean(explore[(h, True)])), 3) if explore[(h, True)] else None,
                                        'F1_short': round(float(np.mean(explore[(h, False)])), 3)} for h in ['2hop', '3hop', '4hop']}
    out = ROOT/'analysis'/'process_reward_validation.json'; out.write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1))

if __name__ == '__main__':
    main()
