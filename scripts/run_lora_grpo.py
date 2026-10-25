"""Time-boxed LoRA GRPO: --mode native (LoRA only) or head (LoRA + value_aux head jointly).

Each iteration: sample G episodes for each of Q train questions, advantage = reward - group mean,
one epoch of clipped token/decision updates (minibatch 16 decisions), save adapter/head/episodes.
New iterations start only while the time box has room for one more (estimated from the last one).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from types import SimpleNamespace
from lsdrag.decision_qwen35_reader import Qwen35Reader
from lsdrag.lora_grpo import attach_lora, lora_state, sample_episode, reward, decision_logps
from lsdrag.text_decision_benchmark import TextCorpus
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
from evaluate_qwen35_h3 import load_head

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[1]
MODEL = ROOT/'hf/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'
HEAD = ROOT/'results/qwen35_9b_two_promising_20261024/value_aux.pt'
HEAD_SHA = '5191c60eee742695031856b2c66a2b3581995b65ffa78b6347cc9a84d9b476d7'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['native', 'head'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--minutes', type=float, required=True)
    ap.add_argument('--questions', type=int, default=16)
    ap.add_argument('--group', type=int, default=4)
    ap.add_argument('--max-iterations', type=int, default=100)
    ap.add_argument('--lora-lr', type=float, default=1e-5)
    ap.add_argument('--head-lr', type=float, default=1e-5)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=20261027)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    deadline = t0+60*args.minutes
    save(out/'config.json', vars(args) | dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                              module_sha256=hashlib.sha256((REPO/'src/lsdrag/lora_grpo.py').read_bytes()).hexdigest()))
    status = lambda phase, **kw: save(out/'status.json', dict(phase=phase, pid=os.getpid(), minutes=(time.monotonic()-t0)/60, **kw))
    status('loading')
    torch.manual_seed(args.seed)
    gen = torch.Generator(device='cuda').manual_seed(args.seed)

    split = json.loads((REPO/'lab/registry/decision_state_policy_iteration_20261023/manifest.json').read_text())
    order = sorted(split['train'])
    random.Random(args.seed).shuffle(order)
    raw = {}
    for line in (ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open():
        q = json.loads(line)
        if q['id'] in split['train']:
            raw[q['id']] = q
    sys.path.insert(0, str(ROOT/'source/musique_scorer'))
    from metrics.answer import compute_exact, compute_f1

    reader = Qwen35Reader(str(MODEL), SimpleNamespace(max_output_tokens=512, context_limit=16384))
    model = reader.model
    lora = attach_lora(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    groups = [dict(params=lora, lr=args.lora_lr)]
    head = None
    if args.mode == 'head':
        assert hashlib.sha256(HEAD.read_bytes()).hexdigest() == HEAD_SHA
        head = load_head(dict(kind='qwen35_base', checkpoint=str(HEAD)), 'cuda')
        head.requires_grad_(True)
        groups.append(dict(params=list(head.parameters()), lr=args.head_lr))
    opt = torch.optim.AdamW(groups, weight_decay=0.)
    rows = [json.loads(l) for l in (ROOT/'corpora/musique_shared_20261023/documents.jsonl').open()]
    vectors = np.load(ROOT/'corpora/musique_shared_20261023/vectors.npy', mmap_mode='r')
    embed = V11ProcessEmbedder()
    corpus = TextCorpus(rows, vectors, embed)
    prompt = (REPO/'prompts/text_rag_integrated_v5.txt').read_text()
    metrics, cursor, last_iter = [], 0, None
    try:
        for it in range(args.max_iterations):
            tick = time.monotonic()
            if last_iter and tick+last_iter > deadline:
                break
            qids = order[cursor:cursor+args.questions]
            cursor += args.questions
            model.eval()
            if head is not None:
                head.eval()
            batch, eps = [], []
            for qi, qid in enumerate(qids):
                q = raw[qid]
                golds = [q['answer']]+q['answer_aliases']
                group = []
                for g in range(args.group):
                    ep, recs = sample_episode(reader, corpus, prompt, q['question'], head, args.temperature, gen)
                    em = max(compute_exact(a, ep['answer']) for a in golds)
                    f1 = max(compute_f1(a, ep['answer']) for a in golds)
                    ep.update(qid=qid, iteration=it, sample=g, em=em, f1=f1, reward=reward(em, f1, ep['prompt_tokens'], bool(ep['errors'])))
                    group.append((ep, recs))
                mean = float(np.mean([e['reward'] for e, _ in group]))
                for e, recs in group:
                    e['advantage'] = e['reward']-mean
                    eps.append(e)
                    if abs(e['advantage']) > 1e-9:
                        batch += [r | dict(advantage=e['advantage']) for r in recs]
                status('rollout', iteration=it, questions_done=qi+1, of=len(qids))
            save(out/f'episodes_iter{it}.json', eps)
            rollout_min = (time.monotonic()-tick)/60
            # One epoch of clipped updates; units = policy tokens + head decisions.
            model.train()
            if head is not None:
                head.train()
            random.Random(args.seed+it).shuffle(batch)
            stats = dict(clipped=0, units=0, loss=[], max_abs_logp_err=0., sum_abs_logp_err=0., n_err=0)
            for mb in [batch[i:i+16] for i in range(0, len(batch), 16)]:
                units = sum(len(r['positions'])+('head_action' in r and head is not None) for r in mb)
                if units == 0:
                    continue
                opt.zero_grad()
                total = 0.
                for r in mb:
                    tok, hl = decision_logps(model, head, r, 'cuda')
                    terms, olds = [], []
                    if tok is not None:
                        terms.append(tok); olds.append(torch.tensor(r['old_logps'], device='cuda'))
                    if hl is not None:
                        terms.append(hl[None]); olds.append(torch.tensor([r['head_old_logp']], device='cuda'))
                    if not terms:
                        continue
                    new, old = torch.cat(terms), torch.cat(olds)
                    err = (new.detach()-old).abs()
                    stats['max_abs_logp_err'] = max(stats['max_abs_logp_err'], float(err.max()))
                    stats['sum_abs_logp_err'] += float(err.sum()); stats['n_err'] += len(err)
                    ratio = (new-old).exp()
                    a = r['advantage']
                    obj = torch.minimum(ratio*a, ratio.clamp(.8, 1.2)*a)
                    stats['clipped'] += int(((ratio < .8) | (ratio > 1.2)).sum())
                    loss = -obj.sum()/units
                    assert torch.isfinite(loss)
                    loss.backward()
                    total += float(loss)
                params = lora+(list(head.parameters()) if head is not None else [])
                torch.nn.utils.clip_grad_norm_(params, 1.)
                opt.step()
                stats['units'] += units
                stats['loss'].append(total)
            torch.save(lora_state(model), out/f'lora_iter{it}.pt')
            if head is not None:
                torch.save(head.state_dict(), out/f'head_iter{it}.pt')
            m = dict(iteration=it, questions=len(qids), episodes=len(eps), decisions_trained=len(batch),
                     reward=float(np.mean([e['reward'] for e in eps])), EM=int(sum(e['em'] for e in eps)), F1=float(np.mean([e['f1'] for e in eps])),
                     prompt_tokens=float(np.mean([e['prompt_tokens'] for e in eps])), tool_calls=float(np.mean([e['tool_calls'] for e in eps])),
                     errors=sum(bool(e['errors']) for e in eps), groups_with_variance=sum(len({round(e['reward'], 9) for e in eps if e['qid'] == q}) > 1 for q in qids),
                     clip_fraction=stats['clipped']/max(stats['units'], 1), mean_abs_logp_err=stats['sum_abs_logp_err']/max(stats['n_err'], 1),
                     max_abs_logp_err=stats['max_abs_logp_err'], optimizer_steps=len(stats['loss']),
                     rollout_minutes=rollout_min, iteration_minutes=(time.monotonic()-tick)/60)
            metrics.append(m)
            save(out/'metrics.json', metrics)
            print(json.dumps(m), flush=True)
            last_iter = time.monotonic()-tick
            if time.monotonic() > deadline:
                break
        status('complete', iterations=len(metrics))
    except BaseException as e:
        status('failed', error=repr(e))
        raise
    finally:
        embed.close()


if __name__ == '__main__':
    main()
