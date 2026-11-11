"""Replay recorded dev192 agent trajectories in vLLM to time the LLM work of each arm.

The head/pointer policy is not ported to vLLM. Instead every recorded turn is replayed with its exact
input token ids and the number of tokens the LLM actually generated in that turn:

native turn  one request: prompt = context + forced syntax tokens, generate n tokens
             (n = tool-name branch tokens + argument/answer tokens, including chunk IDs).
head turn    request A: prompt = context, generate 1 token (stands for the prefill whose hidden state the
             head reads; counted against the head arm), then for search/FINAL request B: prompt = context +
             forced syntax prefix, generate n argument/answer tokens. A read turn has request A only
             (the pointer picks chunk numbers; its ~1 ms is not added).

Requests run one at a time (batch 1), turns of an episode in order, prefix cache reset before every
episode, condition order rotated per question. Decode cost does not depend on token identity, so the
base model weights are used for every condition. Tool execution time is not replayed; it is reported
from the original runs. Usage (GPU must be free):
  PYTHONPATH=/content/local_fast/vllm_env python3 scripts/bench_v22_vllm_replay.py [--limit N]
"""
import argparse
import glob
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path('/content/local_fast/lsdrag-text-v1')
MODEL = ROOT/'hf/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554'
CONDITIONS = {
    'N0': ('results/v22_s2_dev192/episodes/N0_*.json', 'native'),
    'H0P': ('results/v22_s2_dev192/episodes/H0P_*.json', 'head'),
    'NRL2': ('results/v22_s4_main_native_dev192_i2/episodes/NRL2_*.json', 'native'),
    'HRL2': ('results/v22_s4_main_head_dev192_i2/episodes/HRL2_*.json', 'head'),
}
GEN = ('branch', 'free', 'fallback')


def plan(episode, arm):
    """List of (prompt_token_ids, max_tokens, kind) for one recorded episode."""
    calls = episode['reader_calls']
    segments = [u['segment'] for u in calls if u.get('segment')]
    requests, si = [], 0
    for i, u in enumerate(calls):
        if 'segment' in u or u.get('mode') in ('chunk_pointer', 'native_sampled_trie') or u.get('generated_tokens'):
            continue
        # a capture (prefill) entry starts a turn; its prompt_tokens is the context length
        n_ctx = u['prompt_tokens']
        nxt = calls[i+1] if i+1 < len(calls) else None
        if nxt is None:
            break
        if arm == 'head' and nxt.get('mode') == 'chunk_pointer':
            ctx = next(s['context_ids'] for s in segments[si:] if len(s['context_ids']) > n_ctx)[:n_ctx]
            requests.append((ctx, 1, 'head_prefill'))
            continue
        seg = segments[si]
        si += 1
        assert len(seg['context_ids']) == n_ctx, (len(seg['context_ids']), n_ctx)
        forced = [t for t, k in zip(seg['tokens'], seg['kinds']) if k == 'forced']
        n = sum(k in GEN for k in seg['kinds'])
        if arm == 'head':
            requests.append((seg['context_ids'], 1, 'head_prefill'))
        if n:
            requests.append((seg['context_ids']+forced, n, 'generate'))
    return requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='questions to time (0 = all 192)')
    ap.add_argument('--warmup', type=int, default=4)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams, TokensPrompt
    data = {}
    for name, (pattern, arm) in CONDITIONS.items():
        eps = {json.load(open(p))['qid']: json.load(open(p)) for p in glob.glob(str(ROOT/pattern))}
        data[name] = {q: (plan(e, arm), e) for q, e in eps.items()}
    qids = sorted(set.intersection(*(set(v) for v in data.values())))
    assert len(qids) == 192
    if a.limit:
        qids = [q for h in ('2hop', '3hop', '4hop') for q in [x for x in qids if x.startswith(h)][:a.limit//3]]
    for name in data:  # recorded generation must be fully replayed
        for q in qids:
            reqs, e = data[name][q]
            assert sum(n for _, n, k in reqs if k == 'generate') == e['generated_tokens'], (name, q)
    llm = LLM(model=str(MODEL), dtype='bfloat16', gpu_memory_utilization=0.55, max_model_len=20480,
              enable_prefix_caching=True, seed=0)
    def run(reqs):
        llm.reset_prefix_cache()
        t0 = time.perf_counter()
        for ids, n, _ in reqs:
            llm.generate([TokensPrompt(prompt_token_ids=ids)],
                         SamplingParams(max_tokens=n, min_tokens=n, ignore_eos=True, temperature=0.), use_tqdm=False)
        return time.perf_counter()-t0
    names = list(CONDITIONS)
    for q in qids[:a.warmup]:
        for name in names:
            run(data[name][q][0])
    rows = []
    for i, q in enumerate(qids):
        order = names[i % 4:]+names[:i % 4]
        for name in order:
            reqs, e = data[name][q]
            sec = run(reqs)
            rows.append(dict(qid=q, condition=name, llm_seconds=sec, requests=len(reqs),
                             generated=e['generated_tokens'], prompt_tokens=sum(len(r[0]) for r in reqs),
                             tool_seconds_recorded=sum(t['seconds'] for t in e['trajectory'])))
    out = ROOT/f'results/v22_vllm_replay_{datetime.now(timezone.utc):%Y%m%dT%H%M}'
    out.mkdir(parents=True)
    (out/'episodes.json').write_text(json.dumps(rows)+'\n')
    by = {n: {r['qid']: r for r in rows if r['condition'] == n} for n in names}
    summary = {}
    for n in names:
        v = list(by[n].values())
        summary[n] = dict(llm_seconds=float(np.mean([r['llm_seconds'] for r in v])),
                          llm_seconds_median=float(np.median([r['llm_seconds'] for r in v])),
                          tool_seconds_recorded=float(np.mean([r['tool_seconds_recorded'] for r in v])),
                          generated_tokens=float(np.mean([r['generated'] for r in v])),
                          requests=float(np.mean([r['requests'] for r in v])))
    rng = np.random.default_rng(20261110)
    pairs = {}
    for h, nat in (('H0P', 'N0'), ('HRL2', 'NRL2'), ('HRL2', 'N0')):
        d = np.array([by[h][q]['llm_seconds']-by[nat][q]['llm_seconds'] for q in qids])
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        pairs[f'{h}-{nat}'] = dict(mean=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                   relative=float(d.mean()/summary[nat]['llm_seconds']), faster=int((d < 0).sum()), n=len(d))
    gen = [r for r in rows if r['generated'] > 0]
    res = dict(questions=len(qids), engine='vllm', model=str(MODEL), summary=summary, paired_llm_seconds=pairs,
               note='Replay of recorded turns with exact input ids and generated-token counts; head arm adds one '
                    'prefill request per turn for the head; tool time is from the original HF runs, not replayed.')
    import vllm
    res['vllm_version'] = vllm.__version__
    (out/'latency.json').write_text(json.dumps(res, indent=2)+'\n')
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
