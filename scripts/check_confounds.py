"""Two offline confound checks for the full-FT GRPO results (no new GRPO runs).

A. Decision-temperature confound (native arm trained its tool-name branch at T=4, so decision-logit gradients
   were scaled by 1/4). On the same dev192 iteration-0 decision states, how much did the decision policy move?
   - native: base LLM vs A1 iteration-4 LLM, exact action distribution over legal trie branches at T=1
   - head:   loaded KL0 head vs C1 iteration-4 head on base-LLM features (C1 had the LLM frozen)
B. Head-capacity confound (GRPO trained only action+skip, trunk frozen). Can the frozen-trunk head represent the
   stopping rule "FINAL iff all gold paragraphs already surfaced"? Fit on C1 train decision states (base LLM
   features), question-grouped 80/20 split: last layer only vs full head vs untrained KL0 head.
Usage: python scripts/check_confounds.py --states-a 400 --states-b 4000
"""
import argparse, copy, glob, json, random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from common import ROOT, REPO, CONFIG, load_head
from lsdrag.decision_policy import PolicyReader, distribution
from lsdrag.fullft import prefixes_for
from lsdrag.text_decision_benchmark import TOOLS, canonical
from lsdrag.text_integrated import ACTIONS, observed_ids, options_for

KL0 = ROOT/'results/decision_shared_grpo192_kl0/grpo_iteration1.pt'

def load_questions():
    raw = {}
    for name in ['train', 'dev']:
        for line in (ROOT/f'source/musique/musique_ans_v1.0_{name}.jsonl').open():
            q = json.loads(line); raw[q['id']] = q
    return raw

def states(e, question, prompt):
    """Yield (messages, history, decision index) for every non-forced decision of an episode."""
    msgs = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': question}]; hist = e['trajectory']
    for i, d in enumerate(e['decisions']):
        if d['forced'] or i > len(hist):
            break
        if i > 0:
            x = hist[i-1]; tc = {'id': f'call_{i-1}', 'type': 'function', 'function': {'name': x['tool_name'], 'arguments': canonical(x['arguments'])}}
            msgs = msgs+[{'role': 'assistant', 'content': None, 'tool_calls': [tc]}, {'role': 'tool', 'tool_call_id': tc['id'], 'content': canonical(x['result'])}]
        yield msgs, hist[:i], i

def native_probs(model, tok, reader, msgs, hist):
    legal = [a for a in ACTIONS if a != 'read_chunk' or observed_ids(hist)]
    seqs = {a: tok.encode(prefixes_for(a)[0], add_special_tokens=False) for a in legal}
    ctx = reader.prepare(msgs, TOOLS)['input_ids'][0].tolist(); lp = {}
    with torch.inference_mode():
        for a, s in seqs.items():
            ids = torch.tensor([ctx+s[:-1]], device=model.device)
            logits = model.lm_head(model.model(input_ids=ids, use_cache=False).last_hidden_state[0, len(ctx)-1:]).float()
            tot = 0.
            for j in range(len(s)):
                ch = sorted({o[j] for o in seqs.values() if len(o) > j and o[:j] == s[:j]})
                if len(ch) > 1:
                    tot += float(logits[j, ch].log_softmax(-1)[ch.index(s[j])])
            lp[a] = tot
    p = np.exp(np.array([lp.get(a, -np.inf) for a in ACTIONS])); return p/p.sum()

def head_pack(reader, msgs, hist, question):
    pack, _ = reader.capture_pack(msgs, hist, float('inf')); reader.pending = None
    legal = torch.tensor([[any(o['action'] == a for o in options_for(question, hist)) for a in ACTIONS]], device=pack.device)
    return pack.float(), legal

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s); pos = s[y == 1]; neg = s[y == 0]
    return float((pos[:, None] > neg[None, :]).mean()+.5*(pos[:, None] == neg[None, :]).mean()) if len(pos) and len(neg) else None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--states-a', type=int, default=400); ap.add_argument('--states-b', type=int, default=4000); a = ap.parse_args()
    random.seed(0); torch.manual_seed(0); raw = load_questions(); prompt = (REPO/'prompts/text_rag_integrated_v5.txt').read_text()
    reader = PolicyReader(CONFIG['model'], SimpleNamespace(max_output_tokens=512, context_limit=16384)); tok = reader.tokenizer; base = reader.model
    out = {}
    # ---- A. decision movement on the same states
    from transformers import AutoModelForCausalLM
    a1 = AutoModelForCausalLM.from_pretrained(ROOT/'results/mh_native_A1/model_iteration4', dtype=torch.bfloat16, device_map='cuda').eval()
    eps = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(ROOT/'results/mh_native_A1/eval/i0/*hop*.json')))]
    S = [(e['qid'], m, h) for e in eps for m, h, _ in states(e, raw[e['qid']]['question'], prompt)]; random.shuffle(S); S = S[:a.states_a]
    d_native = []
    for qid, m, h in S:
        p0 = native_probs(base, tok, reader, m, h); p1 = native_probs(a1, tok, reader, m, h); d_native.append((p0, p1))
    del a1; torch.cuda.empty_cache()
    kl0 = load_head({'checkpoint': str(KL0)}, 'cuda'); c1 = load_head({'checkpoint': str(ROOT/'results/mh_headonly_C1/head_iteration4.pt')}, 'cuda')
    epsh = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(ROOT/'results/mh_headonly_C1/eval/i0/*hop*.json')))]
    Sh = [(e['qid'], m, h) for e in epsh for m, h, _ in states(e, raw[e['qid']]['question'], prompt)]; random.shuffle(Sh); Sh = Sh[:a.states_a]
    d_head = []
    for qid, m, h in Sh:
        x, legal = head_pack(reader, m, h, raw[qid]['question'])
        with torch.no_grad():
            p0 = distribution(kl0(x[None], legal)[0], legal).probs[0].cpu().numpy(); p1 = distribution(c1(x[None], legal)[0], legal).probs[0].cpu().numpy()
        d_head.append((p0, p1))
    def move(D):
        P0 = np.stack([x for x, _ in D]); P1 = np.stack([y for _, y in D])
        return {'states': len(D), 'argmax_changed': float((P0.argmax(1) != P1.argmax(1)).mean()), 'mean_abs_dp': float(np.abs(P1-P0).sum(1).mean()/2),
                'mean_P_FINAL_before': float(P0[:, 3].mean()), 'mean_P_FINAL_after': float(P1[:, 3].mean()),
                'entropy_before': float((-(P0*np.log(np.clip(P0, 1e-12, 1))).sum(1)).mean()), 'entropy_after': float((-(P1*np.log(np.clip(P1, 1e-12, 1))).sum(1)).mean())}
    out['A_decision_movement'] = {'native_LLM_base_to_A1_i4 (T=1)': move(d_native), 'head_KL0_to_C1_i4': move(d_head),
                                  'note': 'native decision gradients were scaled by 1/T_dec = 1/4 during A1 training'}
    # ---- B. head capacity for the oracle stop rule
    cid = {}
    for line in (Path(CONFIG['corpus'])/'documents.jsonl').open():
        d = json.loads(line); cid[(d['title'], d['text'])] = d['chunk_id']
    tr = [json.loads(Path(p).read_text()) for p in glob.glob(str(ROOT/'results/mh_headonly_C1/episodes/train_i*.json'))]
    rows = []
    for e in tr:
        gold = {cid[(p['title'], p['paragraph_text'])] for p in raw[e['qid']]['paragraphs'] if p['is_supporting']}
        for m, h, i in states(e, raw[e['qid']]['question'], prompt):
            seen = set()
            for x in h:
                if x['tool_name'] != 'read_chunk' and isinstance(x['result'], list):
                    seen |= {r['chunk_id'] for r in x['result'] if isinstance(r, dict) and 'chunk_id' in r}
            rows.append((e['qid'], m, h, int(gold <= seen)))
    pos = [r for r in rows if r[3]]; neg = [r for r in rows if not r[3]]; random.shuffle(neg)
    rows = pos[:a.states_b//2]+neg[:a.states_b-min(len(pos), a.states_b//2)]
    X, L, Y, Q = [], [], [], []
    for qid, m, h, y in rows:
        x, legal = head_pack(reader, m, h, raw[qid]['question']); X.append(x.cpu()); L.append(legal.cpu()); Y.append(y); Q.append(qid)
    X = torch.stack(X).cuda(); L = torch.cat(L).cuda(); Y = torch.tensor(Y, device='cuda', dtype=torch.float)
    qs = sorted(set(Q)); random.shuffle(qs); test_q = set(qs[:len(qs)//5]); te = torch.tensor([q in test_q for q in Q], device='cuda')
    def pfinal(h, idx):
        return distribution(h(X[idx], L[idx])[0], L[idx]).probs[:, 3]
    def fit(train_all):
        h = copy.deepcopy(kl0).train()
        for n, p in h.named_parameters():
            p.requires_grad_(train_all or n.startswith(('action.', 'skip.')))
        opt = torch.optim.Adam([p for p in h.parameters() if p.requires_grad], lr=1e-3); tr_idx = (~te).nonzero().squeeze(1)
        w = torch.where(Y[tr_idx] > 0, (1-Y[tr_idx]).sum()/Y[tr_idx].sum().clamp_min(1), torch.ones_like(Y[tr_idx]))
        for _ in range(400):
            opt.zero_grad(); pf = pfinal(h, tr_idx).clamp(1e-6, 1-1e-6)
            loss = -(w*(Y[tr_idx]*pf.log()+(1-Y[tr_idx])*(1-pf).log())).mean(); loss.backward(); opt.step()
        return h.eval()
    res = {}
    te_idx = te.nonzero().squeeze(1); tr_idx = (~te).nonzero().squeeze(1)
    for name, h in [('KL0 head, untrained', kl0), ('last layer only (action+skip, 10.8k)', fit(False)), ('full head incl. trunk (340k)', fit(True))]:
        with torch.no_grad():
            st, sr = pfinal(h, te_idx).cpu().numpy(), pfinal(h, tr_idx).cpu().numpy()
        yt, yr = Y[te_idx].cpu().numpy(), Y[tr_idx].cpu().numpy()
        res[name] = {'test_AUC': auc(yt, st), 'train_AUC': auc(yr, sr), 'test_balanced_acc@0.5': float(((st > .5) == yt)[yt == 1].mean()/2+((st > .5) == yt)[yt == 0].mean()/2)}
    out['B_head_capacity_stop_rule'] = {'states': len(rows), 'positives': int(Y.sum()), 'test_states': int(te.sum()), 'results': res}
    path = ROOT/'analysis'/'confound_checks.json'; path.write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))

if __name__ == '__main__':
    main()
