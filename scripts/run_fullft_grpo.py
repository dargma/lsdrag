"""GRPO with full fine-tuning of the 4B LLM, native ARAG (--arm native) or head ARAG (--arm head).

Reward (default, --reward discount)  R = F1 * gamma**tool_calls: use tools exactly as needed.
        Alternatives: latency  R = F1 * max(0, 1 + lam*(1 - L/L_ref)), L = .026*generated + .218*input_k (fitted seconds proxy, R2 .98);
        no bonus if a non-empty answer drops support_ids. Advantage A = R - group mean (no std division).
Both arms, one pass per iteration over trajectories with A != 0, minibatches of --batch trajectories:
  per trajectory, decision term(s) + free payload tokens, each -min(r*A, clip(r,.8,1.2)*A) + beta*k3(ref||new),
  summed and divided by the fixed minibatch size (no per-length normalization, no entropy bonus).
  native: decision = sum of tool-name branch-token logprobs of the LLM; ref = iteration-0 LLM.
  head:   decision = head logprob (stored packs); ref = loaded head; LLM gets payload tokens only.
LLM: fp32 master weights + fp32 AdamW (lr --lr); the bf16 model used for rollouts is the rounded copy.
(bf16 stochastic rounding was tried and rejected: its noise moved log-probs ~0.1 nats per step.)
Usage (smoke): python scripts/run_fullft_grpo.py --arm native --name smoke_native --questions 16 --iterations 1
"""
import argparse, copy, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from lsdrag.decision_policy import rollout, distribution
from lsdrag.fullft import SampledReader, segment_logps, MasterAdamW, evidence_step_rewards
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
from common import ROOT, REPO, CONFIG, sha, save, load_head

SOURCES = ['scripts/run_fullft_grpo.py', 'scripts/common.py', 'src/lsdrag/fullft.py', 'src/lsdrag/decision_policy.py',
           'src/lsdrag/decision_policy_recovery.py', 'prompts/text_rag_integrated_v5.txt']
KL0_HEAD = ROOT/'results/decision_shared_grpo192_kl0/grpo_iteration1.pt'

def k3(ref, new):
    d = ref-new
    return d.exp()-d-1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['native', 'head'], required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--questions', type=int, default=192, help='train questions per iteration')
    ap.add_argument('--split', default=str(REPO/'configs/split.json'))
    ap.add_argument('--resample', action='store_true', help='each iteration draws --questions from the train pool, equal per hop type')
    ap.add_argument('--iterations', type=int, default=4)
    ap.add_argument('--repeats', type=int, default=8)
    ap.add_argument('--lam', type=float, default=1.)
    ap.add_argument('--gamma', type=float, default=.8, help='discount reward: F1*gamma**tool_calls')
    ap.add_argument('--token-gamma', type=float, default=1., help='discount reward also multiplies token_gamma**generated_tokens')
    ap.add_argument('--answer-max-chars', type=int, default=0, help='grammar cap on FINAL answer length (0 = none)')
    ap.add_argument('--proc-alpha', type=float, default=.5); ap.add_argument('--proc-cost', type=float, default=.05)
    ap.add_argument('--reward', choices=['discount', 'latency', 'calls', 'process'], default='discount',
                    help='latency: F1*max(0,1+lam*(1-L/L_ref)), L=.026*gen+.218*input_k seconds proxy; calls: F1*(1+lam*(1-calls/8))')
    ap.add_argument('--latency-ref', type=float, default=3.5, help='seconds; dev64 native/head iteration-0 mean is ~3.4-3.6')
    ap.add_argument('--beta', type=float, default=.02)
    ap.add_argument('--lr', type=float, default=1e-6)
    ap.add_argument('--lr-head', type=float, default=1e-5)
    ap.add_argument('--temperature', type=float, default=1.)
    ap.add_argument('--decision-temperature', type=float, default=None, help='native tool-name sampling T (entropy matched to head)')
    ap.add_argument('--batch', type=int, default=16, help='trajectories per optimizer step')
    ap.add_argument('--head', default=str(KL0_HEAD))
    ap.add_argument('--eval-at', default='', help='comma list of iterations after which to run dev64 greedy, 0 = before training')
    ap.add_argument('--save-model-at', default='', help='comma list of iterations whose full LLM weights are saved')
    ap.add_argument('--max-updates', type=int, default=0, help='smoke: stop each update after N optimizer steps')
    ap.add_argument('--freeze-llm', action='store_true', help='arm C: reader LLM frozen, only the head is trained (head arm only)')
    ap.add_argument('--max-seconds', type=int, default=60*60*20)
    a = ap.parse_args()
    out = ROOT/'results'/a.name; out.mkdir(parents=True, exist_ok=False)
    SPLIT = json.loads(Path(a.split).read_text()); seed = CONFIG['grpo']['seed']
    pool = SPLIT['train']; train = pool[:a.questions]
    def iteration_questions(iteration):
        if not a.resample:
            return train
        rng = np.random.default_rng(seed+iteration); hops = sorted({q.split('__')[0][:4] for q in pool}); k = a.questions//len(hops)
        return [q for h in hops for q in rng.choice(sorted(q for q in pool if q.startswith(h)), k, replace=False).tolist()]
    m = dict(vars(a), train_pool=pool, validation=SPLIT['validation'], seed=seed, model=CONFIG['model'], corpus=CONFIG['corpus'], source_sha256={p: sha(REPO/p) for p in SOURCES},
             head_sha256=sha(Path(a.head)) if a.arm == 'head' else None)
    save(out/'manifest.json', m)
    torch.set_num_threads(2); torch.manual_seed(seed); start = time.perf_counter(); embed = None
    counts = {'episodes': 0, 'generated': 0, 'inputs': 0}
    def phase(name, **kw):
        save(out/'status.json', {'phase': name, 'pid': None if name in ['complete', 'failed'] else os.getpid(), **counts,
                                 'seconds': time.perf_counter()-start, 'gpu_peak_gb': torch.cuda.max_memory_allocated()/2**30, **kw})
    phase('loading'); raw = {}
    wanted = set(pool) | set(SPLIT['validation'])
    for split in ['train', 'dev']:
        for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
            q = json.loads(line)
            if q['id'] in wanted:
                raw[q['id']] = q
    sys.path.insert(0, str(ROOT/'source/musique_scorer')); from metrics.answer import compute_f1, compute_exact
    def score(q, ans):
        golds = [q['answer']]+q['answer_aliases']
        return max(compute_f1(g, ans) for g in golds), max(compute_exact(g, ans) for g in golds)
    try:
        reader = SampledReader(m['model'], SimpleNamespace(max_output_tokens=512, context_limit=16384)); model = reader.model
        reader.answer_max_chars = a.answer_max_chars or None
        assert not a.freeze_llm or a.arm == 'head', '--freeze-llm needs the head arm'
        assert a.reward != 'process' or a.freeze_llm, 'process reward is implemented for the head-only arm'
        gold = {}
        if a.reward == 'process':   # train-only supporting paragraph ids, used for the step reward, never for evaluation
            cid = {}
            for line in (Path(m['corpus'])/'documents.jsonl').open():
                d = json.loads(line); cid[(d['title'], d['text'])] = d['chunk_id']
            gold = {q: {cid[(p['title'], p['paragraph_text'])] for p in raw[q]['paragraphs'] if p['is_supporting']} for q in pool}
        if a.freeze_llm:
            ref = opt = None; params = []
        else:
            ref = copy.deepcopy(model).eval(); [p.requires_grad_(False) for p in ref.parameters()]
            model.gradient_checkpointing_enable(); model.config.use_cache = True
            params = [p for p in model.parameters()]; [p.requires_grad_(True) for p in params]
            opt = MasterAdamW(params, a.lr)   # fp32 master weights + fp32 AdamW; bf16 model is the rounded copy
        head = anchor = head_opt = None
        if a.arm == 'head':
            anchor = load_head({'checkpoint': a.head}, 'cuda'); head = copy.deepcopy(anchor)
            for n, p in head.named_parameters():
                p.requires_grad_(n.startswith(('action.', 'skip.')))
            head_opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=a.lr_head, weight_decay=0.)
        embed = V11ProcessEmbedder(); deadline = time.perf_counter()+a.max_seconds
        prompt = (REPO/'prompts/text_rag_integrated_v5.txt').read_text()
        rows = [json.loads(line) for line in (Path(m['corpus'])/'documents.jsonl').open()]
        vec = np.load(Path(m['corpus'])/'vectors.npy', mmap_mode='r')
        def run(qid, temperature, tag):
            reader.temperature = temperature; reader.decision_temperature = a.decision_temperature; model.eval()
            ep, steps, _ = rollout(reader, raw[qid]['question'], rows, vec, embed, prompt, head, deadline,
                                   stochastic=temperature > 0 and a.arm == 'head', epsilon=0., native_fast=a.arm == 'native')
            segs = [u.pop('segment') for u in ep['reader_calls'] if 'segment' in u]
            f1, em = score(raw[qid], ep['answer'])
            ep.update(qid=qid, policy=tag, f1=f1, em=em, free_tokens=sum(k == 'free' for s in segs for k in s['kinds']),
                      fallback_tokens=sum(k == 'fallback' for s in segs for k in s['kinds']))
            counts['episodes'] += 1; counts['generated'] += ep['generated_tokens']; counts['inputs'] += ep['prompt_tokens']
            return ep, steps, segs
        def evaluate(iteration):
            rows_ = []
            for qid in SPLIT['validation']:
                torch.manual_seed(seed); ep, _, _ = run(qid, 0., f'dev64_i{iteration}'); save(out/f'eval/i{iteration}/{qid}.json', ep)
                rows_.append(ep); phase('eval', iteration=iteration, questions=len(rows_))
            s = {k: float(np.mean([e[k] for e in rows_])) for k in ['f1', 'em', 'tool_calls', 'generated_tokens', 'prompt_tokens', 'free_tokens', 'seconds']}
            s['latency_proxy'] = .026*s['generated_tokens']+.218*s['prompt_tokens']/1000
            s.update(empty_answer=float(np.mean([e['answer'] == '' for e in rows_])), three_plus_calls=float(np.mean([e['tool_calls'] >= 3 for e in rows_])),
                     errors=int(sum(bool(e['errors']) for e in rows_)))
            save(out/f'eval/i{iteration}/summary.json', s); return s
        evals = {}; curves = []
        at = lambda spec: {int(x) for x in spec.split(',') if x != ''}
        if 0 in at(a.eval_at):
            evals[0] = evaluate(0)
        for iteration in range(1, a.iterations+1):
            trajs = []
            train = iteration_questions(iteration); save(out/f'train_questions_i{iteration}.json', train)
            for qi, qid in enumerate(train):
                group = []
                for rep in range(a.repeats):
                    torch.manual_seed(seed+iteration*10000+qi*10+rep)
                    ep, steps, segs = run(qid, a.temperature, f'train_i{iteration}_r{rep}')
                    ep['latency_proxy'] = .026*ep['generated_tokens']+.218*ep['prompt_tokens']/1000
                    rtg = None
                    if a.reward == 'process':   # step rewards -> returns-to-go, one per decision (incl. forced)
                        sr = evidence_step_rewards(ep, gold[qid], a.proc_alpha, a.proc_cost); rtg = np.cumsum(sr[::-1])[::-1]
                        ep['step_rewards'] = sr; ep['reward'] = float(rtg[0])
                    elif a.reward == 'discount':   # correct answers only; each tool call keeps gamma of the credit, so correct > wrong always
                        ep['reward'] = ep['f1']*a.gamma**ep['tool_calls']*a.token_gamma**ep['generated_tokens']
                    else:
                        short = 1-ep['latency_proxy']/a.latency_ref if a.reward == 'latency' else 1-ep['tool_calls']/8
                        cites = ep['answer'] == '' or bool(ep['support_ids'])   # no speed bonus for dropping the citation
                        ep['reward'] = ep['f1']*max(0., 1+a.lam*short*cites)
                    save(out/f'episodes/train_i{iteration}_r{rep}_{qid}.json', ep)
                    group.append({'qid': qid, 'rep': rep, 'reward': ep['reward'], 'f1': ep['f1'], 'calls': ep['tool_calls'], 'latency': ep['latency_proxy'], 'segments': segs,
                                  'steps': [{'pack': s['pack'], 'legal': s['legal'], 'action': s['action'], 'old_logp': s['old_logp']} for s in steps if not s['forced']] if a.arm == 'head' else [],
                                  'rtg': rtg, 'nonforced': [i for i, s in enumerate(steps) if not s['forced']]})
                mean = np.mean([g['reward'] for g in group])
                for g in group:
                    g['adv'] = float(g['reward']-mean)
                if a.reward == 'process':   # step advantage = return-to-go minus group mean return-to-go at the same step index
                    T = max(len(g['rtg']) for g in group)
                    base = [np.mean([g['rtg'][s] for g in group if len(g['rtg']) > s]) for s in range(T)]
                    for g in group:
                        g['step_adv'] = [float(g['rtg'][i]-base[i]) for i in g['nonforced']]
                        g['adv'] = max([abs(x) for x in g['step_adv']], default=0.)
                trajs += group; phase('collection', iteration=iteration, questions=qi+1)
            # update
            model.train(); phase('update', iteration=iteration)
            order = [t for t in trajs if abs(t['adv']) > 1e-8]; np.random.default_rng(seed+iteration).shuffle(order)
            stats = {'gap': [], 'clip': [], 'kl': [], 'dec_ratio_dev': [], 'loss': []}; updates = 0
            for first in range(0, len(order), a.batch):
                if a.max_updates and updates >= a.max_updates:
                    break
                chunk = order[first:first+a.batch]; total = 0.
                if opt:
                    opt.zero_grad(set_to_none=True)
                if head_opt:
                    head_opt.zero_grad()
                for t in chunk:
                    A = t['adv']
                    for seg in (t['segments'] if opt else []):
                        r = segment_logps(model, seg, a.temperature)
                        if r is None:
                            continue
                        fn, fo, bn, bo = r
                        with torch.no_grad():
                            rr = segment_logps(ref, seg, a.temperature); fr, br = rr[0], rr[2]
                        loss = 0.
                        if len(fn):
                            ratio = (fn-fo).exp()
                            loss = loss+(-torch.minimum(ratio*A, ratio.clamp(.8, 1.2)*A)+a.beta*k3(fr, fn)).sum()
                            stats['gap'].append(float((fn.detach()-fo).abs().mean())); stats['clip'].append(float(((ratio.detach()-1).abs() > .2).float().mean()))
                            stats['kl'].append(float(k3(fr, fn.detach()).mean()))
                        if bn is not None:
                            ratio = (bn-bo).exp(); stats['dec_ratio_dev'].append(float((ratio.detach()-1).abs()))
                            loss = loss-torch.minimum(ratio*A, ratio.clamp(.8, 1.2)*A)+a.beta*k3(br, bn)
                        loss = loss/a.batch; loss.backward(); total += float(loss.detach())
                    if head is not None and t['steps']:
                        x = torch.stack([s['pack'] for s in t['steps']]).float().cuda(); legal = torch.stack([s['legal'] for s in t['steps']]).cuda()
                        act = torch.tensor([s['action'] for s in t['steps']], device='cuda'); old = torch.tensor([s['old_logp'] for s in t['steps']], device='cuda')
                        new = distribution(head(x, legal)[0], legal).log_prob(act)
                        with torch.no_grad():
                            refp = distribution(anchor(x, legal)[0], legal).log_prob(act)
                        ratio = (new-old).exp(); stats['dec_ratio_dev'] += (ratio.detach()-1).abs().tolist()
                        if 'step_adv' in t:
                            A = torch.tensor(t['step_adv'], device='cuda')
                        hl = (-torch.minimum(ratio*A, ratio.clamp(.8, 1.2)*A)+a.beta*k3(refp, new)).sum()/a.batch; hl.backward(); total += float(hl.detach())
                if opt:
                    torch.nn.utils.clip_grad_norm_(params, 1.); opt.step()
                if head_opt:
                    torch.nn.utils.clip_grad_norm_([p for p in head.parameters() if p.requires_grad], 1.); head_opt.step()
                updates += 1; stats['loss'].append(total)
                phase('update', iteration=iteration, trajectories_done=first+len(chunk), of=len(order))
            if opt:
                opt.zero_grad(set_to_none=True)
            model.eval()
            rw = np.array([t['reward'] for t in trajs]); f1 = np.array([t['f1'] for t in trajs]); calls = np.array([t['calls'] for t in trajs])
            groups = [[t for t in trajs if t['qid'] == q] for q in train]
            curve = {'iteration': iteration, 'trajectories': len(trajs), 'trained_trajectories': len(order), 'updates': updates,
                     'mean_reward': float(rw.mean()), 'mean_f1': float(f1.mean()), 'mean_calls': float(calls.mean()),
                     'mean_latency_proxy': float(np.mean([t['latency'] for t in trajs])), 'three_plus_calls': float(np.mean(calls >= 3)),
                     'groups_with_signal': float(np.mean([np.ptp([t['reward'] for t in g]) > 1e-8 for g in groups])),
                     'groups_f1_varies': float(np.mean([np.ptp([t['f1'] for t in g]) > 1e-8 for g in groups])),
                     **{k: float(np.mean(v)) if v else None for k, v in stats.items()}}
            if head is not None:
                with torch.no_grad():
                    allsteps = [s for t in trajs for s in t['steps']]
                    x = torch.stack([s['pack'] for s in allsteps]).float().cuda(); legal = torch.stack([s['legal'] for s in allsteps]).cuda()
                    curve['head_changed_vs_loaded'] = float((distribution(head(x, legal)[0], legal).probs.argmax(-1) != distribution(anchor(x, legal)[0], legal).probs.argmax(-1)).float().mean())
                torch.save(head.state_dict(), out/f'head_iteration{iteration}.pt')
            curves.append(curve); save(out/'curves.json', curves)
            if iteration in at(a.save_model_at):
                model.save_pretrained(out/f'model_iteration{iteration}', safe_serialization=True)
            if iteration in at(a.eval_at):
                evals[iteration] = evaluate(iteration)
            save(out/'evals.json', evals); phase('updated', iteration=iteration)
        phase('complete')
    except BaseException as e:
        phase('failed', error=repr(e)); raise
    finally:
        if embed:
            embed.close()

if __name__ == '__main__':
    main()
