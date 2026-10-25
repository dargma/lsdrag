"""POC P3: answer-only ceiling with gold support paragraphs given directly (diagnostic, not a policy)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[1]
INSTRUCTION = ('Answer the question using only the paragraphs. Reply with the shortest possible answer '
               '(an entity, date or number), no explanation.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    raw = {}
    for line in (ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open():
        q = json.loads(line)
        if q['id'] in spec['qids']:
            raw[q['id']] = q
    assert set(raw) == set(spec['qids'])
    sys.path.insert(0, str(ROOT/'source/musique_scorer'))
    from metrics.answer import compute_f1, compute_exact
    tok = AutoTokenizer.from_pretrained(spec['model'], local_files_only=True)
    model = Qwen3_5ForCausalLM.from_pretrained(spec['model'], dtype=torch.bfloat16, device_map='cuda',
                                               attn_implementation='sdpa', local_files_only=True).eval()
    rows = []
    for qid in spec['qids']:
        q = raw[qid]
        gold = [p for p in q['paragraphs'] if p['is_supporting']]
        for condition, paragraphs in [('gold_support', gold), ('all20', q['paragraphs'])]:
            context = '\n\n'.join(f"Title: {p['title']}\n{p['paragraph_text']}" for p in paragraphs)
            messages = [{'role': 'user', 'content': f"{INSTRUCTION}\n\n{context}\n\nQuestion: {q['question']}"}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, return_tensors='pt').to('cuda')
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=48, do_sample=False)
            answer = tok.decode(gen[0, ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            golds = [q['answer']]+q['answer_aliases']
            rows.append(dict(qid=qid, condition=condition, answer=answer, gold=q['answer'],
                             f1=max(compute_f1(g, answer) for g in golds), em=max(compute_exact(g, answer) for g in golds),
                             prompt_tokens=int(ids['input_ids'].shape[1]), generated_tokens=int(gen.shape[1]-ids['input_ids'].shape[1])))
    with (out/'oracle_rows.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    summary = {}
    for c in ['gold_support', 'all20']:
        sub = [r for r in rows if r['condition'] == c]
        summary[c] = dict(n=len(sub), f1=sum(r['f1'] for r in sub)/len(sub), em=sum(r['em'] for r in sub)/len(sub),
                          em_count=sum(r['em'] for r in sub), prompt_tokens=sum(r['prompt_tokens'] for r in sub)/len(sub))
    summary.update(model=spec['model'], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   note='Diagnostic ceiling: gold support given directly (gold_support) or the original 20-paragraph pool (all20). Not a policy; different prompt from the agent.')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
