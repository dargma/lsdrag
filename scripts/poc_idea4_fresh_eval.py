"""Idea 4 on fresh questions: re-answer head/native episodes from probe-selected chunks.

The probe is refit from the frozen idea-4 pool features (L16 mean). Selection rule is fixed in advance: top5.
sep = separate question-first forward over observed chunks (as in the train48 POC).
ctx = features from the agent's own final context (prompt + question + tool calls/results), i.e. the prefill
      the agent already computed; one extra forward here only to read it out.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[1]
MODEL = ROOT/'hf/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'
RUN = ROOT/'results/poc_idea4_fresh48_20261026'
OUT = ROOT/'results/poc_idea4_fresh48_reanswer_20261026'
LAYER, TOPK = 16, 5
INSTRUCTION = ('Answer the question using only the paragraphs. Reply with the shortest possible answer '
               '(an entity, date or number), no explanation.')
sys.path.insert(0, str(REPO/'scripts'))
sys.path.insert(0, str(ROOT/'source/musique_scorer'))
from evaluate_decision_shared_corpus import document_id  # noqa: E402
from metrics.answer import compute_exact, compute_f1  # noqa: E402


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    feats = np.load(ROOT/'results/poc_idea4_selector_20261025/features.npz')
    tr = feats['kind'] == 'pool'
    Xp = feats['X'][tr][:, 4096:8192].astype(np.float32)  # L16 mean block
    scaler = StandardScaler().fit(Xp)
    clf = LogisticRegression(C=0.01, max_iter=3000).fit(scaler.transform(Xp), feats['y'][tr])
    prob = lambda F: clf.predict_proba(scaler.transform(np.asarray(F, dtype=np.float32)))[:, 1]

    docs = {}
    for line in (ROOT/'corpora/musique_shared_20261023/documents.jsonl').open():
        r = json.loads(line)
        docs[r['chunk_id']] = r
    raw = {}
    for line in (ROOT/'source/musique/musique_ans_v1.0_dev.jsonl').open():
        q = json.loads(line)
        raw[q['id']] = q
    system = (REPO/'prompts/text_rag_integrated_v5.txt').read_text()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda',
                                               attn_implementation='sdpa', local_files_only=True).eval()

    def hidden(text):
        enc = tok(text, return_offsets_mapping=True, return_tensors='pt')
        offsets = enc.pop('offset_mapping')[0].tolist()
        with torch.no_grad():
            h = model(**enc.to('cuda'), output_hidden_states=True).hidden_states[LAYER][0].float().cpu().numpy()
        return h, offsets, int(enc['input_ids'].shape[1])

    def span_mean(h, offsets, spans):
        idx = [i for i, (a, b) in enumerate(offsets) if b > a and any(a >= s and b <= e for s, e in spans)]
        return h[idx].mean(0) if idx else None

    def sep_features(question, chunk_ids):
        pieces = [f"Title: {docs[c]['title']}\n{docs[c]['text']}" for c in chunk_ids]
        content = f"Question: {question}\n\n{INSTRUCTION}\n\n" + '\n\n'.join(pieces)
        text = tok.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        h, offsets, n = hidden(text)
        out, cursor = [], 0
        for p in pieces:
            start = text.index(p, cursor)
            cursor = start+len(p)
            out.append(span_mean(h, offsets, [(start, cursor)]))
        return out, n

    def ctx_features(question, ep, chunk_ids):
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': question}]
        for t, step in enumerate(ep['trajectory']):
            tc = {'id': f'call_{t}', 'type': 'function', 'function': {'name': step['tool_name'], 'arguments': step['arguments']}}
            messages += [{'role': 'assistant', 'content': None, 'tool_calls': [tc]},
                         {'role': 'tool', 'tool_call_id': tc['id'], 'content': canonical(step['result'])}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        spans = {c: [] for c in chunk_ids}
        for step in ep['trajectory']:
            for r in step['result'] if isinstance(step['result'], list) else []:
                if isinstance(r, dict) and r.get('chunk_id') in spans:
                    piece = canonical(r)
                    s = text.find(piece)
                    while s >= 0:
                        spans[r['chunk_id']].append((s, s+len(piece)))
                        s = text.find(piece, s+1)
        h, offsets, n = hidden(text)
        return [span_mean(h, offsets, spans[c]) for c in chunk_ids], n

    def answer(question, chunk_ids):
        context = '\n\n'.join(f"Title: {docs[c]['title']}\n{docs[c]['text']}" for c in chunk_ids)
        text = tok.apply_chat_template([{'role': 'user', 'content': f"{INSTRUCTION}\n\n{context}\n\nQuestion: {question}"}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=48, do_sample=False)
        return tok.decode(gen[0, ids['input_ids'].shape[1]:], skip_special_tokens=True).strip(), int(ids['input_ids'].shape[1])

    rows, auc_y, auc_sep, auc_ctx = [], [], [], []
    for f in sorted((RUN/'episodes').glob('*.json')):
        ep = json.loads(f.read_text())
        q = raw[ep['qid']]
        golds = [q['answer']]+q['answer_aliases']
        gold_ids = {document_id(p) for p in q['paragraphs'] if p['is_supporting']}
        observed = []
        for step in ep['trajectory']:
            for r in step['result'] if isinstance(step['result'], list) else []:
                if isinstance(r, dict) and r.get('chunk_id') in docs and r['chunk_id'] not in observed:
                    observed.append(r['chunk_id'])
        base = dict(qid=ep['qid'], policy=ep['policy'], observed=len(observed), gold_observed=len(gold_ids & set(observed)),
                    agent_prompt_tokens=ep['prompt_tokens'], tool_calls=ep['tool_calls'])
        score = lambda a: dict(answer=a, em=max(compute_exact(g, a) for g in golds), f1=max(compute_f1(g, a) for g in golds))
        rows.append(base | dict(cond='agent', answer_prompt=0, extra_prefill=0) | score(ep['answer']))
        if not observed:
            continue
        a, n = answer(q['question'], observed)
        rows.append(base | dict(cond='R_all', kept=len(observed), answer_prompt=n, extra_prefill=0) | score(a))
        sep, n_sep = sep_features(q['question'], observed)
        ctx, n_ctx = ctx_features(q['question'], ep, observed)
        for name, F, extra in [('sel_sep_top5', sep, n_sep), ('sel_ctx_top5', ctx, n_ctx)]:
            ok = [i for i, v in enumerate(F) if v is not None]
            p = np.full(len(observed), -1.)
            if ok:
                p[ok] = prob([F[i] for i in ok])
            keep = set(np.argsort(-p)[:TOPK].tolist())
            kept = [c for i, c in enumerate(observed) if i in keep]
            a, n = answer(q['question'], kept)
            rows.append(base | dict(cond=name, kept=len(kept), answer_prompt=n, extra_prefill=extra if name == 'sel_sep_top5' else 0,
                                    ctx_missing=len(observed)-len(ok)) | score(a))
            (auc_sep if name == 'sel_sep_top5' else auc_ctx).extend(p.tolist())
        auc_y.extend(int(c in gold_ids) for c in observed)
        print(ep['policy'], ep['qid'], flush=True)

    summary = {}
    for pol in sorted({r['policy'] for r in rows}):
        n_q = len({r['qid'] for r in rows if r['policy'] == pol})
        for cond in ['agent', 'R_all', 'sel_sep_top5', 'sel_ctx_top5']:
            sub = [r for r in rows if r['policy'] == pol and r['cond'] == cond]
            summary[f'{pol}/{cond}'] = dict(n=len(sub), EM=sum(r['em'] for r in sub), F1=round(sum(r['f1'] for r in sub)/n_q, 3),
                                           input_per_q=round(sum(r['agent_prompt_tokens']+r['answer_prompt']+r['extra_prefill'] for r in sub)/n_q),
                                           answer_prompt=round(np.mean([r['answer_prompt'] for r in sub])) if sub else None)
    y = np.array(auc_y)
    result = dict(questions=len({r['qid'] for r in rows}), summary=summary,
                  chunk_auc=dict(sep=round(float(roc_auc_score(y, auc_sep)), 3), ctx=round(float(roc_auc_score(y, auc_ctx)), 3), chunks=len(y), gold_rate=round(float(y.mean()), 3)),
                  rule='top5 by L16-mean probe refit on idea-4 pool features; fixed before this run',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (OUT/'rows.jsonl').open('w') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False)+'\n')
    (OUT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
