"""Stage 1 (GPU): at every decision state of the train48 base trajectories, re-answer from a clean context (R).

R_all  = question + every chunk observed so far (full text), the rule recomposition
R_gold = question + observed chunks that are gold support (offline oracle for idea 4)
Also stores answer log-probabilities (idea 7 features). Diagnostic only.
"""
import json
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[1]
RUN = ROOT/'results/qwen35_9b_outcomes48_20261024'
MODEL = ROOT/'hf/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'
OUT = ROOT/'results/poc_stage1_state_answers_20261025'
INSTRUCTION = ('Answer the question using only the paragraphs. Reply with the shortest possible answer '
               '(an entity, date or number), no explanation.')
sys.path.insert(0, str(REPO/'scripts'))
sys.path.insert(0, str(ROOT/'source/musique_scorer'))
from evaluate_decision_shared_corpus import document_id  # noqa: E402
from metrics.answer import compute_exact, compute_f1  # noqa: E402


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    docs = {}
    for line in (ROOT/'corpora/musique_shared_20261023/documents.jsonl').open():
        r = json.loads(line)
        docs[r['chunk_id']] = r
    raw = {}
    for line in (ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open():
        q = json.loads(line)
        raw[q['id']] = q
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda',
                                               attn_implementation='sdpa', local_files_only=True).eval()

    def answer(question, chunk_ids):
        context = '\n\n'.join(f"Title: {docs[c]['title']}\n{docs[c]['text']}" for c in chunk_ids)
        messages = [{'role': 'user', 'content': f"{INSTRUCTION}\n\n{context}\n\nQuestion: {question}"}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=48, do_sample=False, output_scores=True, return_dict_in_generate=True)
        new = gen.sequences[0, ids['input_ids'].shape[1]:]
        logps = [float(torch.log_softmax(s[0].float(), -1)[t]) for s, t in zip(gen.scores, new)]
        return dict(answer=tok.decode(new, skip_special_tokens=True).strip(), prompt_tokens=int(ids['input_ids'].shape[1]),
                    generated_tokens=len(new), first_logp=logps[0] if logps else 0., mean_logp=sum(logps)/max(len(logps), 1))

    rows = []
    for f in sorted((RUN/'episodes').glob('*_collector.json')):
        qid = f.name[:-len('_collector.json')]
        ep = json.loads(f.read_text())
        q = raw[qid]
        golds = [q['answer']]+q['answer_aliases']
        gold_ids = {document_id(p) for p in q['paragraphs'] if p['is_supporting']}
        observed = []
        prompts = [sum(u.get('prompt_tokens', 0) for u in ep['reader_calls'][2*i:2*i+2]) for i in range(len(ep['decisions']))]
        for t in range(len(ep['decisions'])):
            if t > 0:
                for r in ep['trajectory'][t-1]['result']:
                    if isinstance(r, dict) and r.get('chunk_id') in docs and r['chunk_id'] not in observed:
                        observed.append(r['chunk_id'])
            row = dict(qid=qid, t=t, calls_total=len(ep['trajectory']), agent_em=max(compute_exact(g, ep['answer']) for g in golds),
                       agent_f1=max(compute_f1(g, ep['answer']) for g in golds), agent_prompt_tokens=ep['prompt_tokens'],
                       capture_prompt_tokens=prompts[t], cumulative_agent_prompt=sum(prompts[:t+1]),
                       observed=len(observed), gold_observed=len(gold_ids & set(observed)), gold_total=len(gold_ids))
            for cond, ids in [('R_all', observed), ('R_gold', [c for c in observed if c in gold_ids])]:
                a = answer(q['question'], ids)
                a.update(em=max(compute_exact(g, a['answer']) for g in golds), f1=max(compute_f1(g, a['answer']) for g in golds))
                row[cond] = a
            rows.append(row)
        print(qid, len(ep['decisions']), 'states', flush=True)
    with (OUT/'rows.jsonl').open('w') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False)+'\n')
    print('states', len(rows))


if __name__ == '__main__':
    main()
