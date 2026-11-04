"""Is chunked (KV-extended) prefill numerically equivalent to a full re-prefill for the head pack?

Takes saved contexts from a head-ARAG episode, rebuilds the token sequence of step t+1, and compares the
head probabilities from (a) full prefill, (b) full prefill repeated, (c) prefill of the longest common
token prefix with step t's produced tokens + suffix on top of that cache.
Usage: python scripts/check_prefill_numerics.py --run RUN --episodes 6
"""
import argparse, glob, json
from pathlib import Path
from types import SimpleNamespace
import torch
from common import ROOT, REPO, CONFIG, load_head
from lsdrag.decision_policy import pack_hidden, matched_span, distribution
from lsdrag.decision_policy_recovery import RecoveryPolicyReader
from lsdrag.text_decision_benchmark import TOOLS, canonical

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--episodes', type=int, default=6); ap.add_argument('--prefix', default='')
    ap.add_argument('--head', default=str(ROOT/'results/decision_shared_grpo192_kl0/grpo_iteration1.pt')); a = ap.parse_args()
    raw = {}
    for name in ['train', 'dev']:
        for line in (ROOT/f'source/musique/musique_ans_v1.0_{name}.jsonl').open():
            q = json.loads(line); raw[q['id']] = q
    reader = RecoveryPolicyReader(CONFIG['model'], SimpleNamespace(max_output_tokens=512, context_limit=16384)); tok = reader.tokenizer; m = reader.model
    head = load_head({'checkpoint': a.head}, 'cuda'); prompt = (REPO/'prompts/text_rag_integrated_v5.txt').read_text()
    def hidden(ids, split=None):
        with torch.inference_mode():
            if split is None:
                t = torch.tensor([ids], device=m.device); return m.model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=True).last_hidden_state[0]
            t = torch.tensor([ids[:split]], device=m.device); o = m.model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=True)
            s = torch.tensor([ids[split:]], device=m.device)
            o2 = m.model(input_ids=s, attention_mask=torch.ones(1, len(ids), device=m.device, dtype=torch.long), past_key_values=o.past_key_values, use_cache=True)
            return torch.cat([o.last_hidden_state[0], o2.last_hidden_state[0]])
    rows = []; lcp_gap = []
    for p in sorted(glob.glob(str(ROOT/'results'/a.run/'episodes'/f'{a.prefix}*.json')))[:a.episodes]:
        e = json.loads(Path(p).read_text()); q = raw[e['qid']]['question']; conts = [u for u in e['reader_calls'] if 'raw_json' in u]
        msgs = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': q}]
        for i, hst in enumerate(e['trajectory']):
            prev = reader.prepare(msgs, TOOLS)['input_ids'][0].tolist()
            produced = prev+conts[i]['prefix_token_ids']+conts[i]['token_ids']
            tc = {'id': f'call_{i}', 'type': 'function', 'function': {'name': hst['tool_name'], 'arguments': canonical(hst['arguments'])}}
            msgs += [{'role': 'assistant', 'content': None, 'tool_calls': [tc]}, {'role': 'tool', 'tool_call_id': tc['id'], 'content': canonical(hst['result'])}]
            ids = reader.prepare(msgs, TOOLS)['input_ids'][0].tolist()
            lcp = next((j for j in range(min(len(ids), len(produced))) if ids[j] != produced[j]), min(len(ids), len(produced)))
            lcp_gap.append(len(produced)-lcp)
            dec = tok.decode(ids, skip_special_tokens=False)
            qs = matched_span(tok, ids, q); obs = matched_span(tok, ids, canonical(hst['result']), last=True)
            legal = torch.tensor([e['decisions'][i+1]['legal']], device='cuda')
            probs = {}
            for k, sp in [('full', None), ('full_again', None), ('kv_extended', lcp)]:
                pk = pack_hidden(hidden(ids, sp), qs, obs).float()[None]
                with torch.no_grad(): probs[k] = distribution(head(pk, legal)[0], legal).probs[0]
            rows.append({'rerun_diff': float((probs['full']-probs['full_again']).abs().max()), 'kv_diff': float((probs['full']-probs['kv_extended']).abs().max()),
                         'argmax_same': bool(probs['full'].argmax() == probs['kv_extended'].argmax()), 'saved_recorded_prob_diff': float((probs['full'].cpu()-torch.tensor(e['decisions'][i+1]['policy_probs'])).abs().max())})
    import numpy as np
    out = {'pairs': len(rows), 'produced_tokens_not_reused_after_lcp': dict(zip(*np.unique(lcp_gap, return_counts=True))),
           **{k: {'max': max(r[k] for r in rows), 'mean': float(np.mean([r[k] for r in rows]))} for k in ['rerun_diff', 'kv_diff', 'saved_recorded_prob_diff']},
           'argmax_same': sum(r['argmax_same'] for r in rows)}
    out['produced_tokens_not_reused_after_lcp'] = {int(k): int(v) for k, v in out['produced_tokens_not_reused_after_lcp'].items()}
    print(json.dumps(out, indent=1))

if __name__ == '__main__':
    main()
