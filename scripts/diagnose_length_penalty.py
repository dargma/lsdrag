"""Why does the length/cost penalty barely change the GRPO policy? CPU-only diagnosis of saved rollouts.

Usage: python scripts/diagnose_length_penalty.py --run RUN [--run RUN2] --out results/x.json
Reads ROOT/results/RUN/episodes/train_i{it}_r{rep}_{qid}.json.
"""
import argparse, collections, glob, json, re
from pathlib import Path
import numpy as np
from common import ROOT, CONFIG

def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(CONFIG['model'])

def load(run):
    groups = collections.defaultdict(dict)
    for p in glob.glob(str(ROOT/'results'/run/'episodes'/'train_i*_r*_*.json')):
        it, rep, qid = re.match(r'train_i(\d+)_r(\d+)_(.+)\.json', Path(p).name).groups()
        groups[(int(it), qid)][int(rep)] = json.loads(Path(p).read_text())
    return {k: [v[r] for r in sorted(v)] for k, v in groups.items() if len(v) >= 2}

def split_generated(e, tok):
    """Generated tokens by role: answer text, chunk-id strings, other JSON syntax/arguments."""
    out = collections.Counter()
    for u in e['reader_calls']:
        if not u.get('generated_tokens') or 'raw_json' not in u: continue
        n = u['generated_tokens']; raw = u['raw_json']
        ids = re.findall(r'c[0-9a-f]{24}', raw)
        idtok = sum(len(tok.encode(f'"{i}"', add_special_tokens=False)) for i in ids)
        try: j = json.loads(raw if raw.lstrip().startswith('{"answer"') or '"answer"' not in raw else raw)
        except Exception: j = {}
        if 'answer' in j:
            atok = len(tok.encode(j['answer'], add_special_tokens=False))
            out['final_answer_text'] += atok; out['final_support_ids'] += idtok; out['final_syntax'] += n-atok-idtok
        elif ids:
            out['read_chunk_ids'] += idtok; out['read_chunk_syntax'] += n-idtok
        else:
            out['search_arguments'] += n
    return out

def analyze(run, tok, lam_grid=(0, .01, .03, .1, .3, 1, 3)):
    G = load(run); R = {}
    # 1. diversity: are the trajectories in a group actually different?
    seq = lambda e: tuple(d['action'] for d in e['decisions'])
    same_actions = np.mean([len({seq(e) for e in g}) == 1 for g in G.values()])
    same_answer = np.mean([len({e['answer'] for e in g}) == 1 for g in G.values()])
    f1_tie = np.mean([np.ptp([e['f1'] for e in g]) < 1e-9 for g in G.values()])
    calls_tie = np.mean([len({e['tool_calls'] for e in g}) == 1 for g in G.values()])
    probs = [max(d['policy_probs']) for g in G.values() for e in g for d in e['decisions'] if not d['forced'] and sum(d['legal']) > 1]
    ent = [-sum(p*np.log(p) for p in d['policy_probs'] if p > 0) for g in G.values() for e in g for d in e['decisions'] if not d['forced'] and sum(d['legal']) > 1]
    R['groups'] = len(G); R['repeats'] = len(next(iter(G.values())))
    R['diversity'] = dict(identical_action_sequence=same_actions, identical_answer=same_answer, f1_all_equal=f1_tie,
                          tool_calls_all_equal=calls_tie, head_max_prob_mean=float(np.mean(probs)), head_entropy_mean_nats=float(np.mean(ent)),
                          head_max_prob_ge_0_9=float(np.mean(np.array(probs) >= .9)))
    # 2. reward composition: within-group spread of F1 vs .03*cost, and how cost splits
    dF1 = [np.std([e['f1'] for e in g]) for g in G.values()]
    dC = [np.std([.03*e['cost_units'] for e in g]) for g in G.values()]
    R['within_group_std'] = dict(f1=float(np.mean(dF1)), penalty_003=float(np.mean(dC)))
    eps = [e for g in G.values() for e in g]
    calls = np.array([e['tool_calls'] for e in eps]); gen = np.array([e['generated_tokens'] for e in eps]); inp = np.array([e['prompt_tokens'] for e in eps])
    first = np.array([e['reader_calls'][0]['prompt_tokens'] for e in eps]); steps = np.array([len(e['decisions']) for e in eps])
    cost = np.array([e['cost_units'] for e in eps])
    R['cost_split_mean'] = dict(cost_units=float(cost.mean()), tool_calls=float(calls.mean()), generated_div100=float(gen.mean()/100),
                                input_div10000=float(inp.mean()/1e4), input_fixed_prefix_share=float((first*steps).sum()/inp.sum()))
    gs = collections.Counter()
    for e in eps: gs.update(split_generated(e, tok))
    R['generated_tokens_by_role_per_episode'] = {k: v/len(eps) for k, v in sorted(gs.items())}
    # 3. does the penalty weight change the advantage at all? (group standardization is scale-free)
    def adv(g, lam):
        r = np.array([e['f1']-lam*e['cost_units'] for e in g]); s = r.std()
        return (r-r.mean())/(s+1e-6)
    base = {k: adv(g, .03) for k, g in G.items()}
    R['advantage_vs_lambda'] = {}
    for lam in lam_grid:
        A = {k: adv(g, lam) for k, g in G.items()}
        a = np.concatenate([A[k] for k in G]); c = np.concatenate([[e['tool_calls'] for e in G[k]] for k in G])
        sign_flip = np.mean(np.concatenate([np.sign(A[k]) != np.sign(base[k]) for k in G]))
        nonzero = np.mean([np.abs(A[k]).max() > 1e-3 for k in G])
        R['advantage_vs_lambda'][str(lam)] = dict(groups_with_signal=float(nonzero), corr_adv_toolcalls=float(np.corrcoef(a, c)[0, 1]) if a.std() else 0.,
                                                  sign_differs_from_003=float(sign_flip))
    # 4. in groups where F1 differs, is longer more accurate? (penalty then fights accuracy)
    cor = [np.corrcoef([e['tool_calls'] for e in g], [e['f1'] for e in g])[0, 1] for g in G.values()
           if np.ptp([e['f1'] for e in g]) > 1e-9 and len({e['tool_calls'] for e in g}) > 1]
    R['f1_differs'] = dict(groups=int(sum(np.ptp([e['f1'] for e in g]) > 1e-9 for g in G.values())), with_call_variation=len(cor),
                           mean_corr_toolcalls_f1=float(np.mean(cor)) if cor else None)
    # 5. which actions can shorten a trajectory? how often is FINAL legal-but-not-chosen and its prob?
    fin = [d['policy_probs'][3] for e in eps for d in e['decisions'] if not d['forced'] and d['legal'][3] and d['action'] != 3]
    R['final_prob_when_not_final'] = dict(n=len(fin), mean=float(np.mean(fin)) if fin else None)
    R['tool_calls_hist'] = {int(k): int(v) for k, v in zip(*np.unique(calls, return_counts=True))}
    R['forced_final_rate'] = float(np.mean([e['forced_final'] for e in eps]))
    return R

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', action='append', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    tok = tokenizer(); res = {r: analyze(r, tok) for r in a.run}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False)); print(json.dumps(res, indent=1, ensure_ascii=False))

if __name__ == '__main__':
    main()
