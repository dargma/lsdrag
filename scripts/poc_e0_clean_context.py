"""POC E0: re-answer from a clean context built only from chunks the agent already observed (no new search, no training).

Conditions per question (same 9B, same answer prompt):
  observed_all  - every unique observed chunk, full text, no tool-call history
  rerank_top5/3 - observed chunks ranked by question-chunk embedding similarity (no gold)
  oracle_clean  - only observed chunks that are gold support (offline gold; upper bound)
Agent answers come from the saved episodes and are rescored with the same scorer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[1]
MODEL = ROOT/'hf/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'
CORPUS = ROOT/'corpora/musique_shared_20261023'
INSTRUCTION = ('Answer the question using only the paragraphs. Reply with the shortest possible answer '
               '(an entity, date or number), no explanation.')
sys.path.insert(0, str(REPO/'src'))
sys.path.insert(0, str(REPO/'scripts'))
sys.path.insert(0, str(ROOT/'source/musique_scorer'))
from evaluate_decision_shared_corpus import document_id  # noqa: E402
from metrics.answer import compute_exact, compute_f1  # noqa: E402


def episodes(source):
    """(set name, qid, agent episode) from train48 collector episodes or dev B0 episodes."""
    if source == 'train48':
        run = ROOT/'results/qwen35_9b_outcomes48_20261024/episodes'
        for f in sorted(run.glob('*_collector.json')):
            yield f.name[:-len('_collector.json')], json.loads(f.read_text())
    else:
        run = ROOT/'results/poc_baselines_dev64_20261025/episodes'
        for f in sorted(run.glob('before_*.json')):
            yield f.name[len('before_'):-len('.json')], json.loads(f.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    docs = {}
    for line in (CORPUS/'documents.jsonl').open():
        r = json.loads(line)
        docs[r['chunk_id']] = r
    vectors = np.load(CORPUS/'vectors.npy', mmap_mode='r')
    row_of = {cid: i for i, cid in enumerate(docs)}
    raw = {}
    for line in (ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open():
        q = json.loads(line)
        raw[q['id']] = q
    from lsdrag.embeddings import Embedder
    embed = Embedder('gemma', cache_dir='/content/local_fast/lsdrag-v11/hf', device='cuda')
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda',
                                               attn_implementation='sdpa', local_files_only=True).eval()

    def answer(question, chunk_ids):
        context = '\n\n'.join(f"Title: {docs[c]['title']}\n{docs[c]['text']}" for c in chunk_ids)
        messages = [{'role': 'user', 'content': f"{INSTRUCTION}\n\n{context}\n\nQuestion: {question}"}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=48, do_sample=False)
        return tok.decode(gen[0, ids['input_ids'].shape[1]:], skip_special_tokens=True).strip(), int(ids['input_ids'].shape[1]), int(gen.shape[1]-ids['input_ids'].shape[1])

    rows = []
    for source in ['train48', 'dev_b0']:
        for qid, ep in episodes(source):
            q = raw[qid]
            golds = [q['answer']]+q['answer_aliases']
            score = lambda a: (max(compute_f1(g, a) for g in golds), max(compute_exact(g, a) for g in golds))
            gold_ids = {document_id(p) for p in q['paragraphs'] if p['is_supporting']}
            observed = []
            for step in ep['trajectory']:
                for r in step['result']:
                    if isinstance(r, dict) and r.get('chunk_id') in docs and r['chunk_id'] not in observed:
                        observed.append(r['chunk_id'])
            qv = np.asarray(embed.encode([q['question']], 'query'))[0]
            ranked = sorted(observed, key=lambda c: -float(np.dot(vectors[row_of[c]], qv)))
            f1, em = score(ep['answer'])
            base = dict(source=source, qid=qid, observed=len(observed), gold_observed=len(gold_ids & set(observed)), gold_total=len(gold_ids))
            rows.append(base | dict(condition='agent', f1=f1, em=em, prompt_tokens=ep['prompt_tokens'], generated_tokens=ep['generated_tokens'], answer=ep['answer']))
            for cond, ids in [('observed_all', observed), ('rerank_top5', ranked[:5]), ('rerank_top3', ranked[:3]),
                              ('oracle_clean', [c for c in observed if c in gold_ids])]:
                a, pt, gt = answer(q['question'], ids)
                f1, em = score(a)
                rows.append(base | dict(condition=cond, f1=f1, em=em, prompt_tokens=pt, generated_tokens=gt, answer=a, chunks=len(ids)))
            print(source, qid, 'done', flush=True)
    with (out/'rows.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    summary = {}
    for source in ['train48', 'dev_b0']:
        for cond in ['agent', 'observed_all', 'rerank_top5', 'rerank_top3', 'oracle_clean']:
            sub = [r for r in rows if r['source'] == source and r['condition'] == cond]
            if sub:
                summary[f'{source}/{cond}'] = dict(n=len(sub), f1=round(sum(r['f1'] for r in sub)/len(sub), 4),
                                                   em=sum(r['em'] for r in sub), prompt_tokens=round(sum(r['prompt_tokens'] for r in sub)/len(sub)))
        sub = [r for r in rows if r['source'] == source and r['condition'] == 'agent']
        summary[f'{source}/all_gold_observed'] = sum(r['gold_observed'] == r['gold_total'] for r in sub)
    summary['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary['note'] = ('Re-answer prompt tokens exclude the retrieval trajectory that produced the chunks; agent prompt tokens are the full '
                       'episode. Diagnostic only; dev_b0 is the 17-question partial P1 set; train48 policy is bootstrap.')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
