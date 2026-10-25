"""S0: kernel equality, LoRA(B=0) identity, rollout/train speed and memory for the 6-hour plan."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from types import SimpleNamespace
from lsdrag.decision_qwen35_reader import Qwen35Reader
from lsdrag.lora_grpo import attach_lora, sample_episode, decision_logps
from lsdrag.text_decision_benchmark import TextCorpus, TOOLS
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
from evaluate_qwen35_h3 import load_head
from run_lora_grpo import MODEL, HEAD, ROOT, REPO

OUT = ROOT/'results/lora_grpo_s0_20261025'


def main():
    tag = sys.argv[1]  # 'fallback' or 'fla'
    OUT.mkdir(parents=True, exist_ok=True)
    reader = Qwen35Reader(str(MODEL), SimpleNamespace(max_output_tokens=512, context_limit=16384))
    prompt = (REPO/'prompts/text_rag_integrated_v5.txt').read_text()
    ep = json.loads((ROOT/'results/poc_baselines_dev64_20261025/episodes/before_2hop__23139_135703.json').read_text())
    messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': 'Who lent its name to the country with the largest economy in Africa?'}]
    for t, s in enumerate(ep['trajectory']):
        tc = {'id': f'call_{t}', 'type': 'function', 'function': {'name': s['tool_name'], 'arguments': json.dumps(s['arguments'], sort_keys=True)}}
        messages += [{'role': 'assistant', 'content': None, 'tool_calls': [tc]},
                     {'role': 'tool', 'tool_call_id': tc['id'], 'content': json.dumps(s['result'], sort_keys=True, ensure_ascii=False)}]
    inputs = reader.prepare(messages, TOOLS)
    torch.cuda.synchronize(); tick = time.perf_counter()
    with torch.inference_mode():
        logits = reader.model(**inputs).logits[0, -64:].float().cpu()
    torch.cuda.synchronize()
    res = dict(tag=tag, tokens=int(inputs['input_ids'].shape[1]), prefill_seconds=time.perf_counter()-tick)
    torch.save(logits, OUT/f'logits_{tag}.pt')
    ref = OUT/'logits_fallback.pt'
    if tag != 'fallback' and ref.exists():
        r = torch.load(ref)
        res.update(max_logit_diff_vs_fallback=float((logits-r).abs().max()), argmax_agree=float((logits.argmax(-1) == r.argmax(-1)).float().mean()))
    if tag == 'fallback':
        print(json.dumps(res)); (OUT/f's0_{tag}.json').write_text(json.dumps(res, indent=2)); return
    lora = attach_lora(reader.model)
    with torch.inference_mode():
        l2 = reader.model(**inputs).logits[0, -64:].float().cpu()
    res['lora_b0_max_diff'] = float((l2-logits).abs().max())
    reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    rows = [json.loads(l) for l in (ROOT/'corpora/musique_shared_20261023/documents.jsonl').open()]
    vectors = np.load(ROOT/'corpora/musique_shared_20261023/vectors.npy', mmap_mode='r')
    embed = V11ProcessEmbedder()
    corpus = TextCorpus(rows, vectors, embed)
    head = load_head(dict(kind='qwen35_base', checkpoint=str(HEAD)), 'cuda').requires_grad_(True)
    gen = torch.Generator(device='cuda').manual_seed(0)
    question = 'Who lent its name to the country with the largest economy in Africa?'
    try:
        for mode, h in [('native', None), ('head', head)]:
            reader.model.eval()
            tick = time.perf_counter()
            e, recs = sample_episode(reader, corpus, prompt, question, h, 1.0, gen)
            res[f'{mode}_episode_seconds'] = time.perf_counter()-tick
            res[f'{mode}_episode'] = dict(answer=e['answer'], actions=e['actions'], prompt_tokens=e['prompt_tokens'], errors=e['errors'])
            reader.model.train()
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); tick = time.perf_counter()
            errs = []
            for r in recs:
                tok, hl = decision_logps(reader.model, h, r, 'cuda')
                loss = 0.
                if tok is not None:
                    errs += (tok.detach().cpu()-torch.tensor(r['old_logps'])).abs().tolist(); loss = loss+tok.sum()
                if hl is not None:
                    errs.append(abs(float(hl)-r['head_old_logp'])); loss = loss+hl
                loss.backward()
            torch.cuda.synchronize()
            res[f'{mode}_train_seconds'] = time.perf_counter()-tick
            res[f'{mode}_train_peak_gib'] = torch.cuda.max_memory_allocated()/2**30
            res[f'{mode}_replay_mean_abs'] = float(np.mean(errs)); res[f'{mode}_replay_max_abs'] = float(np.max(errs))
            res[f'{mode}_lora_grad_nonzero'] = sum(int(p.grad is not None and bool(p.grad.abs().sum() > 0)) for p in lora)
            reader.model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
    finally:
        embed.close()
    print(json.dumps(res, indent=2)); (OUT/f's0_{tag}.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
