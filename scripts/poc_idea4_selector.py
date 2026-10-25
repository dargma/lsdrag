"""Idea 4 POC: can the frozen 9B's own hidden states tell which observed chunks to keep before answering?

Train a chunk probe on MuSiQue 20-paragraph pools from train questions that share no atomic sub-question with any
experiment split; test on the chunks the agent actually observed on train48, then re-answer with only kept chunks.
Context format puts the question first so chunk states can condition on it and on earlier chunks (bridge info).
"""
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[1]
MODEL = ROOT/'hf/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'
OUT = ROOT/'results/poc_idea4_selector_20261025'
LAYERS = [8, 16, 24, 32]
POOL_QUESTIONS = 300
INSTRUCTION = ('Answer the question using only the paragraphs. Reply with the shortest possible answer '
               '(an entity, date or number), no explanation.')
sys.path.insert(0, str(REPO/'scripts'))
sys.path.insert(0, str(ROOT/'source/musique_scorer'))
from evaluate_decision_shared_corpus import document_id  # noqa: E402
from metrics.answer import compute_exact, compute_f1  # noqa: E402


def atoms(qid):
    return set(re.findall(r'\d+', qid.split('__', 1)[1]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    docs = {}
    for line in (ROOT/'corpora/musique_shared_20261023/documents.jsonl').open():
        r = json.loads(line)
        docs[r['chunk_id']] = r
    raw = {}
    for line in (ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open():
        q = json.loads(line)
        raw[q['id']] = q
    split = json.loads((REPO/'lab/registry/decision_state_policy_iteration_20261023/manifest.json').read_text())
    used = set(split['train']) | set(split['validation']) | set(split['test'])
    blocked = set().union(*(atoms(q) for q in used))
    agent_eps = {f.name[:-len('_collector.json')]: json.loads(f.read_text())
                 for f in sorted((ROOT/'results/qwen35_9b_outcomes48_20261024/episodes').glob('*_collector.json'))}
    assert set(agent_eps) <= set(split['train'])
    candidates = sorted(q for q in raw if q.startswith('2hop__') and q not in used and not (atoms(q) & blocked))
    random.Random(20261025).shuffle(candidates)
    pool_qids = candidates[:POOL_QUESTIONS]

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda',
                                               attn_implementation='sdpa', local_files_only=True).eval()

    def render(question, chunks, question_first):
        body = '\n\n'.join(f"Title: {c['title']}\n{c['text']}" for c in chunks)
        content = (f"Question: {question}\n\n{INSTRUCTION}\n\n{body}" if question_first
                   else f"{INSTRUCTION}\n\n{body}\n\nQuestion: {question}")
        return tok.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    def chunk_features(question, chunks):
        text = render(question, chunks, True)
        enc = tok(text, return_offsets_mapping=True, return_tensors='pt')
        offsets = enc.pop('offset_mapping')[0].tolist()
        with torch.no_grad():
            hs = model(**enc.to('cuda'), output_hidden_states=True).hidden_states
        feats, cursor = [], 0
        for c in chunks:
            piece = f"Title: {c['title']}\n{c['text']}"
            start = text.index(piece, cursor)
            end = start+len(piece)
            cursor = end
            idx = [i for i, (a, b) in enumerate(offsets) if a >= start and b <= end and b > a]
            feats.append(np.concatenate([hs[l][0, idx].float().mean(0).cpu().numpy() for l in LAYERS]
                                        + [hs[LAYERS[-1]][0, idx[-1]].float().cpu().numpy()]).astype(np.float16))
        return np.stack(feats)

    def answer(question, chunks):
        text = render(question, chunks, False)
        ids = tok(text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=48, do_sample=False)
        return tok.decode(gen[0, ids['input_ids'].shape[1]:], skip_special_tokens=True).strip(), int(ids['input_ids'].shape[1])

    def as_chunk(p):
        return dict(chunk_id=document_id(p), title=p['title'], text=p['paragraph_text'])

    X, y, g, kind, meta = [], [], [], [], []
    for qid in pool_qids:
        q = raw[qid]
        chunks = [as_chunk(p) for p in q['paragraphs']]
        F = chunk_features(q['question'], chunks)
        for c, p, f in zip(chunks, q['paragraphs'], F):
            X.append(f); y.append(int(p['is_supporting'])); g.append(qid); kind.append('pool'); meta.append(c['chunk_id'])
    print('pool done', len(X), flush=True)
    for qid, ep in agent_eps.items():
        q = raw[qid]
        gold = {document_id(p) for p in q['paragraphs'] if p['is_supporting']}
        observed = []
        for step in ep['trajectory']:
            for r in step['result']:
                if isinstance(r, dict) and r.get('chunk_id') in docs and r['chunk_id'] not in observed:
                    observed.append(r['chunk_id'])
        if not observed:
            continue
        F = chunk_features(q['question'], [docs[c] for c in observed])
        for c, f in zip(observed, F):
            X.append(f); y.append(int(c in gold)); g.append(qid); kind.append('agent'); meta.append(c)
    print('agent done', len(X), flush=True)
    np.savez_compressed(OUT/'features.npz', X=np.stack(X), y=np.array(y), g=np.array(g), kind=np.array(kind), chunk=np.array(meta))
    (OUT/'pool_qids.json').write_text(json.dumps(pool_qids)+'\n')

    # Probe: train on pool, score agent chunks.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    X, y, g, kind, meta = np.stack(X).astype(np.float32), np.array(y), np.array(g), np.array(kind), np.array(meta)
    tr, te = kind == 'pool', kind == 'agent'
    d = 4096
    blocks = {f'L{l}_mean': slice(i*d, (i+1)*d) for i, l in enumerate(LAYERS)} | {'L32_last': slice(len(LAYERS)*d, (len(LAYERS)+1)*d)}
    result = {'pool_questions': len(pool_qids), 'pool_chunks': int(tr.sum()), 'agent_chunks': int(te.sum()),
              'agent_gold_rate': float(y[te].mean()), 'auc_agent': {}}
    best, best_p = None, None
    for name, sl in blocks.items():
        sc = StandardScaler().fit(X[tr][:, sl])
        clf = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(X[tr][:, sl]), y[tr])
        p = clf.predict_proba(sc.transform(X[te][:, sl]))[:, 1]
        auc = roc_auc_score(y[te], p)
        result['auc_agent'][name] = round(float(auc), 3)
        if best is None or auc > best:
            best, best_p, best_name = auc, p, name
    # Question-similarity baseline (same embedding as semantic_search).
    vectors = np.load(ROOT/'corpora/musique_shared_20261023/vectors.npy', mmap_mode='r')
    row_of = {cid: i for i, cid in enumerate(docs)}
    from lsdrag.embeddings import Embedder
    embed = Embedder('gemma', cache_dir='/content/local_fast/lsdrag-v11/hf', device='cuda')
    qvec = {qid: np.asarray(embed.encode([raw[qid]['question']], 'query'))[0] for qid in agent_eps}
    sim = np.array([float(np.dot(vectors[row_of[c]], qvec[q])) for c, q in zip(meta[te], g[te])])
    result['auc_agent']['question_similarity_baseline'] = round(float(roc_auc_score(y[te], sim)), 3)
    result['best_block'] = best_name

    # End-to-end on train48: keep chunks by probe and re-answer.
    scores = dict(zip(zip(g[te], meta[te]), best_p))
    rows, summary = [], {}
    for qid, ep in agent_eps.items():
        q = raw[qid]
        golds = [q['answer']]+q['answer_aliases']
        observed = [c for (qq, c) in scores if qq == qid]
        ranked = sorted(observed, key=lambda c: -scores[(qid, c)])
        for cond, keep in [('probe_top2', ranked[:2]), ('probe_top3', ranked[:3]), ('probe_top5', ranked[:5]),
                           ('probe_p>=0.3', [c for c in ranked if scores[(qid, c)] >= 0.3] or ranked[:2]),
                           ('probe_p>=0.5', [c for c in ranked if scores[(qid, c)] >= 0.5] or ranked[:2])]:
            keep_in_order = [c for c in observed if c in keep]
            a, pt = answer(q['question'], [docs[c] for c in keep_in_order])
            rows.append(dict(qid=qid, cond=cond, kept=len(keep_in_order), em=max(compute_exact(x, a) for x in golds),
                             f1=max(compute_f1(x, a) for x in golds), prompt_tokens=pt, answer=a))
    for cond in sorted({r['cond'] for r in rows}):
        sub = [r for r in rows if r['cond'] == cond]
        summary[cond] = dict(EM=sum(r['em'] for r in sub), F1=round(sum(r['f1'] for r in sub)/len(sub), 3),
                             kept=round(sum(r['kept'] for r in sub)/len(sub), 1), answer_prompt=round(sum(r['prompt_tokens'] for r in sub)/len(sub)))
    result['train48_reanswer'] = summary
    result['reference'] = 'train48: agent EM21; R_all EM23 (~3.6k answer prompt); question-similarity top5 EM19; R_gold EM30'
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with (OUT/'reanswer_rows.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    (OUT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
