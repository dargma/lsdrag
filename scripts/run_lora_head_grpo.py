"""Joint GRPO: the original head update (KL .02 to the BC head) plus LoRA on the 4B LLM.

Same train192, reward F1 - .03*cost and group-standardized advantage as
run_decision_shared_grpo.py. The head now acts on hidden states of the LoRA model,
and payload tokens are sampled so the LLM gets its own clipped policy gradient.
No global GPU lock: this runs beside the head-only pipeline on a shared GPU.
"""
import argparse, copy, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from lsdrag.decision_policy import PolicyHead, rollout, distribution, returns_for_episode
from lsdrag.decision_grpo import group_advantages
from lsdrag.lora4b import SamplingReader, attach_lora, lora_state, segment_logps
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
from common import ROOT, REPO, CONFIG, SPLIT, sha, save

SOURCES = ['scripts/run_lora_head_grpo.py', 'scripts/common.py', 'src/lsdrag/lora4b.py', 'src/lsdrag/decision_policy.py',
           'src/lsdrag/decision_policy_recovery.py', 'src/lsdrag/decision_grpo.py', 'prompts/text_rag_integrated_v5.txt']

def read(p):
    return json.loads(Path(p).read_text())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='decision_shared_lora_grpo192')
    ap.add_argument('--questions', type=int, default=192, help='first N of train192')
    ap.add_argument('--iterations', type=int, default=1)
    ap.add_argument('--repeats', type=int, default=4)
    ap.add_argument('--kl', type=float, default=.02)
    ap.add_argument('--lr-head', type=float, default=1e-4)
    ap.add_argument('--lr-lora', type=float, default=2e-5)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--temperature', type=float, default=1.)
    ap.add_argument('--lora-batch', type=int, default=16, help='trajectories per LoRA optimizer step')
    ap.add_argument('--max-seconds', type=int, default=10800)
    a = ap.parse_args()
    base = dict(seed=CONFIG['grpo']['seed'], model=CONFIG['model'], corpus=CONFIG['corpus'], reference=CONFIG['bc_head'])
    train = SPLIT['train'][:a.questions]
    out = ROOT/'results'/a.name; out.mkdir(parents=True, exist_ok=False)
    m = dict(vars(a), train=train, seed=base['seed'], model=base['model'], corpus=base['corpus'], reference=base['reference'],
             reference_sha256=sha(Path(base['reference'])), source_sha256={p: sha(REPO/p) for p in SOURCES},
             reward='F1 - .03*cost_units; group-standardized advantage shared by head decisions and LLM tokens',
             note='Head update identical to the original GRPO except kl coefficient argument. LoRA: PPO clip .2 on sampled free tokens, no KL to the base LLM, one pass per iteration.')
    save(out/'manifest.json', m)
    torch.set_num_threads(2); torch.manual_seed(m['seed']); start = time.perf_counter(); embed = None
    counts = {'episodes': 0, 'generated': 0, 'inputs': 0}
    def phase(name, **kw):
        save(out/'status.json', {'phase': name, 'pid': None if name in ['complete', 'failed'] else os.getpid(), **counts, 'seconds': time.perf_counter()-start, **kw})
    phase('loading'); wanted = set(train); raw = {}
    for split in ['train', 'dev']:
        for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
            q = json.loads(line)
            if q['id'] in wanted:
                raw[q['id']] = q
    sys.path.insert(0, str(ROOT/'source/musique_scorer')); from metrics.answer import compute_f1
    try:
        reader = SamplingReader(m['model'], SimpleNamespace(max_output_tokens=512, context_limit=16384))
        model = reader.model; lora = attach_lora(model, a.rank, 2*a.rank)
        lora_opt = torch.optim.AdamW(lora, lr=a.lr_lora, weight_decay=0.)
        embed = V11ProcessEmbedder(); deadline = time.perf_counter()+a.max_seconds
        prompt = (REPO/'prompts/text_rag_integrated_v5.txt').read_text()
        anchor = PolicyHead(2560, 'mlp').cuda(); anchor.load_state_dict(torch.load(m['reference'], weights_only=True, map_location='cuda')); anchor.eval()
        head = copy.deepcopy(anchor)
        with torch.no_grad():
            head.value.weight.zero_(); head.value.bias.zero_()
        for n, p in head.named_parameters():
            p.requires_grad_(n.startswith(('action.', 'skip.')))
        head_opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=a.lr_head)
        rows = [json.loads(line) for line in (Path(m['corpus'])/'documents.jsonl').open()]
        vec = np.load(Path(m['corpus'])/'vectors.npy', mmap_mode='r')
        curves = []
        for iteration in range(a.iterations):
            batch, trajectories = [], []
            reader.temperature = a.temperature
            for qi, qid in enumerate(train):
                q = raw[qid]
                for rep in range(a.repeats):
                    torch.manual_seed(m['seed']+iteration*10000+qi*10+rep)
                    ep, steps, _ = rollout(reader, q['question'], rows, vec, embed, prompt, head, deadline, stochastic=True, epsilon=0.)
                    f1 = max(compute_f1(x, ep['answer']) for x in [q['answer']]+q['answer_aliases'])
                    reward = f1-.03*ep['cost_units']
                    segments = [{k: u.pop(k) for k in ['context_ids', 'prefix_token_ids', 'token_ids', 'token_logps', 'token_free']}
                                for u in ep['reader_calls'] if 'context_ids' in u]
                    trajectories.append({'qid': qid, 'rep': rep, 'reward': reward, 'segments': [s for s in segments if s['token_ids']]})
                    for s, ret in zip(steps, returns_for_episode(steps, f1, .03)):
                        if not s['forced']:
                            batch.append(s | {'return': ret, 'qid': qid, 'rep': rep, 'episode_reward': reward, 'episode_steps': sum(not t['forced'] for t in steps)})
                    ep.update(qid=qid, policy=f'train_i{iteration}_r{rep}', f1=f1, reward=reward)
                    save(out/f'episodes/train_i{iteration}_r{rep}_{qid}.json', ep)
                    counts['episodes'] += 1; counts['generated'] += ep['generated_tokens']; counts['inputs'] += ep['prompt_tokens']
                phase('collection', iteration=iteration, questions=qi+1)
            reader.temperature = 0.
            group = {qid: group_advantages([next(t['reward'] for t in trajectories if t['qid'] == qid and t['rep'] == r) for r in range(a.repeats)]) for qid in train}
            # Head: identical update to run_decision_shared_grpo.py.
            x = torch.stack([s['pack'] for s in batch]).float().cuda(); legal = torch.stack([s['legal'] for s in batch]).cuda()
            act = torch.tensor([s['action'] for s in batch], device='cuda'); old = torch.tensor([s['old_logp'] for s in batch], device='cuda')
            adv = torch.tensor([group[s['qid']][s['rep']] for s in batch], device='cuda', dtype=torch.float)
            roww = torch.tensor([len(batch)/(len(train)*a.repeats*s['episode_steps']) for s in batch], device='cuda')
            with torch.no_grad():
                before = distribution(head(x, legal)[0], legal).log_prob(act); assert torch.allclose(before, old, atol=2e-4, rtol=2e-4), 'behavior logprob mismatch'
                ap_ = distribution(anchor(x, legal)[0], legal).probs
            head_losses = []
            for epoch in range(4):
                torch.manual_seed(m['seed']+iteration*100+epoch)
                for idx in torch.randperm(len(batch), device='cuda').split(32):
                    head_opt.zero_grad(); dist = distribution(head(x[idx], legal[idx])[0], legal[idx]); prob = dist.probs
                    kl = (prob*(prob.clamp_min(1e-9).log()-ap_[idx].clamp_min(1e-9).log())).sum(-1)
                    ratio = (dist.log_prob(act[idx])-old[idx]).exp(); actor = -torch.minimum(ratio*adv[idx], ratio.clamp(.8, 1.2)*adv[idx])
                    loss = (roww[idx]*(actor+a.kl*kl-.01*dist.entropy())).mean(); loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in head.parameters() if p.requires_grad], 1.); head_opt.step(); head_losses.append(float(loss.detach()))
            with torch.no_grad():
                changed = float((distribution(head(x, legal)[0], legal).probs.argmax(-1) != ap_.argmax(-1)).float().mean())
            # LoRA: one pass, clipped token ratio, each trajectory weighted equally.
            phase('lora_update', iteration=iteration)
            order = [t for t in trajectories if abs(group[t['qid']][t['rep']]) > 1e-8 and t['segments']]
            np.random.default_rng(m['seed']+iteration).shuffle(order)
            replay_gap, clip_frac, lora_losses = [], [], []
            for first in range(0, len(order), a.lora_batch):
                chunk = order[first:first+a.lora_batch]; lora_opt.zero_grad(); total = 0.
                for t in chunk:
                    A = float(group[t['qid']][t['rep']]); n = sum(sum(s['token_free']) for s in t['segments'])
                    if not n:
                        continue
                    for s in t['segments']:
                        new = segment_logps(model, s, a.temperature)
                        old_lp = torch.tensor(s['token_logps'], device=new.device); keep = torch.tensor(s['token_free'], device=new.device)
                        ratio = (new-old_lp).exp()
                        if first == 0:
                            replay_gap.append(float((new.detach()-old_lp).abs()[keep].mean()) if keep.any() else 0.)
                        clip_frac.append(float(((ratio.detach()-1).abs() > .2)[keep].float().mean()) if keep.any() else 0.)
                        tok = -torch.minimum(ratio*A, ratio.clamp(.8, 1.2)*A)
                        loss = (tok*keep).sum()/(n*len(chunk)); loss.backward(); total += float(loss.detach())
                torch.nn.utils.clip_grad_norm_(lora, 1.); lora_opt.step(); lora_losses.append(total)
                phase('lora_update', iteration=iteration, trajectories_done=first+len(chunk), of=len(order))
            torch.save(head.state_dict(), out/f'grpo_iteration{iteration}.pt'); torch.save(lora_state(model), out/f'lora_iteration{iteration}.pt')
            rewards = [t['reward'] for t in trajectories]
            curves.append({'iteration': iteration, 'states': len(batch), 'trajectories': len(trajectories), 'lora_trajectories': len(order),
                           'mean_reward': float(np.mean(rewards)),
                           'head': {'loss': float(np.mean(head_losses)), 'changed_vs_anchor': changed},
                           'lora': {'loss': float(np.mean(lora_losses)) if lora_losses else None, 'steps': len(lora_losses),
                                    'replay_abs_logp_gap_first_batch': float(np.mean(replay_gap)) if replay_gap else None,
                                    'ratio_clip_fraction': float(np.mean(clip_frac)) if clip_frac else None}})
            save(out/'curves.json', curves); phase('updated', iteration=iteration)
        last = a.iterations-1
        save(out/'summary.json', {'counts': counts, 'curves': curves, 'head_sha256': sha(out/f'grpo_iteration{last}.pt'), 'lora_sha256': sha(out/f'lora_iteration{last}.pt')})
        phase('complete')
    except BaseException as e:
        phase('failed', error=repr(e)); raise
    finally:
        if embed:
            embed.close()

if __name__ == '__main__':
    main()
