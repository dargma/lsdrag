"""GPU check of the head-ARAG execution path on a few train questions (greedy, KL0 head).

Per step: (1) the head's action is the tool that runs; (2) the forced prefix is the tool-call syntax for
that action; (3) the next step's tokenized context starts with [previous context + prefix + generated
tokens] token for token, so the KV cache can be extended instead of re-prefilled; (4) incremental prefill
(previous KV + new tokens only) gives the same hidden pack / head probabilities as the full re-prefill
that the current code does; (5) time of both.

Usage: python scripts/check_head_prefill_gpu.py --questions 8 --out RESULT.json
"""
import argparse, json, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from common import ROOT, REPO, CONFIG, SPLIT, load_head
from lsdrag.decision_policy import pack_hidden, matched_span, rollout, distribution
from lsdrag.decision_policy_recovery import RecoveryPolicyReader
from lsdrag.text_decision_benchmark import TOOLS
from lsdrag.text_integrated import ACTIONS
from lsdrag.v11_embedding_worker import V11ProcessEmbedder

class Recording(RecoveryPolicyReader):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.log = []
    def capture_pack(self, messages, history, deadline):
        pack, usage = super().capture_pack(messages, history, deadline)
        self.log.append({'kind': 'capture', 'ids': list(self.pending['ids']), 'pack': pack.float().cpu(),
                         'qs': usage['question_span'], 'obs': usage['observation_span'], 'seconds': usage['prefill_seconds']})
        return pack, usage
    def continue_cached(self, action, trajectory, deadline=float('inf')):
        value, usage, err = super().continue_cached(action, trajectory, deadline)
        self.log.append({'kind': 'continue', 'action': action, 'prefix': usage['prefix_token_ids'], 'gen': usage['token_ids']})
        return value, usage, err

def incremental(reader, prev_ids, new_ids):
    """Prefill prev_ids, then only the new suffix on top of that cache; return full-sequence hidden and suffix time."""
    m = reader.model
    with torch.inference_mode():
        t = torch.tensor([prev_ids], device=m.device)
        out = m.model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=True, return_dict=True)
        h0 = out.last_hidden_state[0]
        torch.cuda.synchronize(); tick = time.perf_counter()
        s = torch.tensor([new_ids[len(prev_ids):]], device=m.device)
        out2 = m.model(input_ids=s, attention_mask=torch.ones(1, len(new_ids), device=m.device, dtype=torch.long),
                       past_key_values=out.past_key_values, use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        return torch.cat([h0, out2.last_hidden_state[0]]), time.perf_counter()-tick

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--questions', type=int, default=8); ap.add_argument('--out', required=True)
    ap.add_argument('--head', default=str(ROOT/'results/decision_shared_grpo192_kl0/grpo_iteration1.pt')); a = ap.parse_args()
    raw = {}
    for name in ['train', 'dev']:
        for line in (ROOT/f'source/musique/musique_ans_v1.0_{name}.jsonl').open():
            q = json.loads(line); raw[q['id']] = q
    reader = Recording(CONFIG['model'], SimpleNamespace(max_output_tokens=512, context_limit=16384))
    embed = V11ProcessEmbedder(); head = load_head({'checkpoint': a.head}, 'cuda')
    corpus = Path(CONFIG['corpus']); rows = [json.loads(l) for l in (corpus/'documents.jsonl').open()]; vec = np.load(corpus/'vectors.npy', mmap_mode='r')
    prompt = (REPO/'prompts/text_rag_integrated_v5.txt').read_text()
    R = {'steps': 0, 'action_executed': 0, 'prefix_is_syntax': 0, 'token_prefix_consistent': 0, 'head_argmax_same': 0,
         'max_abs_prob_diff': 0., 'max_rel_pack_diff': 0., 'full_prefill_s': 0., 'incremental_s': 0., 'examples': []}
    try:
        for qid in SPLIT['train'][:a.questions]:
            reader.log = []
            ep, steps, _ = rollout(reader, raw[qid]['question'], rows, vec, embed, prompt, head, time.perf_counter()+600, stochastic=False)
            caps = [x for x in reader.log if x['kind'] == 'capture']; cons = [x for x in reader.log if x['kind'] == 'continue']
            executed = [h['tool_name'] for h in ep['trajectory']]
            for i, c in enumerate(cons):
                R['steps'] += 1
                R['action_executed'] += i >= len(executed) and c['action'] == 'FINAL' or i < len(executed) and executed[i] == c['action']
                want = '' if c['action'] == 'FINAL' else '<tool_call>\n{"name": "'+c['action']+'", "arguments": {"'
                R['prefix_is_syntax'] += reader.tokenizer.decode(c['prefix']) == want
                if i+1 >= len(caps): continue
                produced = caps[i]['ids']+c['prefix']+c['gen']; nxt = caps[i+1]['ids']
                same = nxt[:len(produced)] == produced; R['token_prefix_consistent'] += same
                if not same and len(R['examples']) < 3:
                    k = next(j for j in range(len(produced)) if j >= len(nxt) or nxt[j] != produced[j])
                    R['examples'].append({'qid': qid, 'step': i, 'produced_tail': reader.tokenizer.decode(produced[k-5:k+10]), 'next_tail': reader.tokenizer.decode(nxt[k-5:k+10])})
                base = produced if same else caps[i]['ids']
                h, dt = incremental(reader, base, nxt); pack = pack_hidden(h, caps[i+1]['qs'], caps[i+1]['obs']).float().cpu()
                full = caps[i+1]['pack']
                legal = steps[i+1]['legal'][None].cuda()
                with torch.no_grad():
                    p_full = distribution(head(full[None].cuda(), legal)[0], legal).probs; p_inc = distribution(head(pack[None].cuda(), legal)[0], legal).probs
                R['head_argmax_same'] += int(p_full.argmax() == p_inc.argmax())
                R['max_abs_prob_diff'] = max(R['max_abs_prob_diff'], float((p_full-p_inc).abs().max()))
                R['max_rel_pack_diff'] = max(R['max_rel_pack_diff'], float((pack-full).norm()/full.norm()))
                R['full_prefill_s'] += caps[i+1]['seconds']; R['incremental_s'] += dt
            R.setdefault('episodes', []).append({'qid': qid, 'f1_answer': ep['answer'], 'calls': ep['tool_calls'], 'errors': ep['errors']})
    finally:
        embed.close()
    R['pairs_compared'] = R['steps']-len(R['episodes'])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(R, indent=1, ensure_ascii=False)); print(json.dumps(R, indent=1, ensure_ascii=False))

if __name__ == '__main__':
    main()
