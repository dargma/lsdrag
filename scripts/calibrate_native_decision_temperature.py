"""Exact native decision entropy as a function of decision temperature, from saved native rollouts.

For every non-forced decision of the given native episodes, rebuild the context, score every legal
tool-name trie sequence teacher-forced, and get p_T(action) = prod over branch points of
softmax(logits[choices]/T). Entropy over actions is then directly comparable to the head's entropy.
Usage: python scripts/calibrate_native_decision_temperature.py --run smoke_fullft_native --target 0.29
"""
import argparse, glob, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from common import ROOT, REPO, CONFIG
from lsdrag.decision_policy import PolicyReader
from lsdrag.fullft import prefixes_for
from lsdrag.text_decision_benchmark import TOOLS, canonical
from lsdrag.text_integrated import ACTIONS, observed_ids

TEMPS = [1., 1.5, 2., 2.5, 3., 4., 5., 6., 8.]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--target', type=float, required=True)
    ap.add_argument('--limit', type=int, default=400, help='max decisions'); a = ap.parse_args()
    raw = {}
    for name in ['train', 'dev']:
        for line in (ROOT/f'source/musique/musique_ans_v1.0_{name}.jsonl').open():
            q = json.loads(line); raw[q['id']] = q
    reader = PolicyReader(CONFIG['model'], SimpleNamespace(max_output_tokens=512, context_limit=16384)); tok = reader.tokenizer; m = reader.model
    prompt = (REPO/'prompts/text_rag_integrated_v5.txt').read_text()
    seqs_all = {act: tok.encode(prefixes_for(act)[0], add_special_tokens=False) for act in ACTIONS}
    H = {T: [] for T in TEMPS}; greedy_share = []; n = 0
    for p in sorted(glob.glob(str(ROOT/'results'/a.run/'episodes'/'train_*.json'))):
        e = json.loads(Path(p).read_text()); hist = e['trajectory']
        msgs = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': raw[e['qid']]['question']}]
        for i in range(len(e['decisions'])):
            if i >= 8 or n >= a.limit:
                break
            h = hist[:i]
            if i > 0:
                x = hist[i-1]; tc = {'id': f'call_{i-1}', 'type': 'function', 'function': {'name': x['tool_name'], 'arguments': canonical(x['arguments'])}}
                msgs += [{'role': 'assistant', 'content': None, 'tool_calls': [tc]}, {'role': 'tool', 'tool_call_id': tc['id'], 'content': canonical(x['result'])}]
            legal = [act for act in ACTIONS if act != 'read_chunk' or observed_ids(h)]
            seqs = {act: seqs_all[act] for act in legal}
            ctx = reader.prepare(msgs, TOOLS)['input_ids'][0].tolist()
            parts = {act: [] for act in legal}   # list of (logits over choices, index of own token) per branch point
            with torch.inference_mode():
                for act, s in seqs.items():
                    ids = torch.tensor([ctx+s[:-1]], device=m.device)
                    hid = m.model(input_ids=ids, use_cache=False).last_hidden_state[0, len(ctx)-1:]
                    logits = m.lm_head(hid).float()
                    for j in range(len(s)):
                        sharing = [o for o in seqs.values() if len(o) > j and o[:j] == s[:j]]
                        choices = sorted({o[j] for o in sharing})
                        if len(choices) > 1:
                            parts[act].append((logits[j, choices], choices.index(s[j])))
            for T in TEMPS:
                lp = torch.tensor([sum(float((lg/T).log_softmax(-1)[k]) for lg, k in parts[act]) for act in legal])
                pr = lp.exp(); pr = pr/pr.sum(); H[T].append(float(-(pr*pr.clamp_min(1e-12).log()).sum()))
                if T == 1.:
                    greedy_share.append(float(pr.max()))
            n += 1
        if n >= a.limit:
            break
    res = {'decisions': n, 'target_head_entropy': a.target, 'p_max_at_T1': float(np.mean(greedy_share)),
           'entropy_by_T': {str(T): float(np.mean(v)) for T, v in H.items()}}
    ent = [res['entropy_by_T'][str(T)] for T in TEMPS]
    res['matched_T'] = float(np.interp(a.target, ent, TEMPS)) if ent[0] < a.target < ent[-1] else None
    out = ROOT/'analysis'/'native_decision_temperature.json'; out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
