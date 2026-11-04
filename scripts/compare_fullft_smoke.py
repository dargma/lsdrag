"""Compare native vs head full-FT GRPO smoke runs: cost, exploration, update health, and lambda choice.

Usage: python scripts/compare_fullft_smoke.py --run native=smoke_fullft_native --run head=smoke_fullft_head --out FILE.md
"""
import argparse, collections, glob, json, re
from pathlib import Path
import numpy as np
from common import ROOT

def episodes(run):
    G = collections.defaultdict(list)
    for p in sorted(glob.glob(str(ROOT/'results'/run/'episodes'/'train_i1_r*_*.json'))):
        e = json.loads(Path(p).read_text()); G[e['qid']].append(e)
    return G

def decision_entropy(e):
    ents = [sum(u['decision_entropy']) for u in e['reader_calls'] if 'decision_entropy' in u]
    if ents:  # native: per decision, sum of branch-point entropies along the sampled path (chain rule -> action entropy)
        return ents
    return [-sum(p*np.log(p) for p in d['policy_probs'] if p > 0) for d in e['decisions'] if not d['forced'] and sum(d['legal']) > 1]

def lam_stats(G, lam):
    adv, calls, signal = [], [], []
    for g in G.values():
        r = np.array([e['f1']*(1+lam*(1-e['tool_calls']/8)) for e in g]); a = r-r.mean()
        adv += a.tolist(); calls += [e['tool_calls'] for e in g]; signal.append(np.ptp(r) > 1e-8)
    adv = np.array(adv); calls = np.array(calls)
    # mean advantage of a trajectory with more calls than its group's correct minimum: how strongly length is pushed
    return dict(signal=float(np.mean(signal)), corr=float(np.corrcoef(adv, calls)[0, 1]) if adv.std() and calls.std() else 0., adv_abs=float(np.abs(adv).mean()))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', action='append', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    runs = [r.split('=', 1) for r in a.run]; L = []
    rows = {}
    for label, run in runs:
        d = ROOT/'results'/run; st = json.loads((d/'status.json').read_text()); cv = json.loads((d/'curves.json').read_text())[-1] if (d/'curves.json').exists() else {}
        ev = json.loads((d/'eval/i0/summary.json').read_text()) if (d/'eval/i0/summary.json').exists() else {}
        G = episodes(run); eps = [e for g in G.values() for e in g]
        ent = [x for e in eps for x in decision_entropy(e)]
        seqs = lambda g: len({tuple(dd['action'] for dd in e['decisions']) for e in g})
        rows[label] = dict(status=st['phase'], seconds=st['seconds'], gpu_peak_gb=st.get('gpu_peak_gb'), eval0=ev, curve=cv,
                           train=dict(episodes=len(eps), f1=np.mean([e['f1'] for e in eps]), calls=np.mean([e['tool_calls'] for e in eps]),
                                      empty=np.mean([e['answer'] == '' for e in eps]), errors=sum(bool(e['errors']) for e in eps),
                                      free=np.mean([e['free_tokens'] for e in eps]), fallback=np.mean([e['fallback_tokens'] for e in eps]),
                                      seconds_per_episode=np.mean([e['seconds'] for e in eps]),
                                      decision_entropy=float(np.mean(ent)) if ent else None,
                                      distinct_action_seqs=float(np.mean([seqs(g) for g in G.values()])),
                                      f1_varies=float(np.mean([np.ptp([e['f1'] for e in g]) > 1e-9 for g in G.values()]))),
                           lam={lam: lam_stats(G, lam) for lam in (0., .3, 1.)})
    f = lambda x, n=2: '—' if x is None else (f'{x:.{n}f}' if isinstance(x, float) else str(x))
    labels = [l for l, _ in runs]
    L += ['# full FT GRPO smoke: ' + ' vs '.join(labels), '', '| 항목 | ' + ' | '.join(labels) + ' |', '|---|' + '---:|'*len(labels)]
    def row(name, get, n=2):
        L.append(f'| {name} | ' + ' | '.join(f(get(rows[l]), n) for l in labels) + ' |')
    row('상태', lambda r: r['status'])
    row('전체 시간 (분)', lambda r: r['seconds']/60, 1)
    row('GPU 최대 메모리 (GB)', lambda r: r['gpu_peak_gb'], 1)
    row('dev64 i0 F1 (%)', lambda r: 100*r['eval0'].get('f1', float('nan')))
    row('dev64 i0 도구 호출', lambda r: r['eval0'].get('tool_calls'))
    row('dev64 i0 빈 답 비율', lambda r: r['eval0'].get('empty_answer'))
    row('dev64 i0 3회+ 비율', lambda r: r['eval0'].get('three_plus_calls'))
    row('train T=1 F1 (%)', lambda r: 100*r['train']['f1'])
    row('train T=1 도구 호출', lambda r: r['train']['calls'])
    row('train T=1 빈 답 비율', lambda r: r['train']['empty'])
    row('train 오류 에피소드', lambda r: r['train']['errors'])
    row('결정 엔트로피 (nats)', lambda r: r['train']['decision_entropy'], 3)
    row('그룹당 서로 다른 행동열 (8궤적 중)', lambda r: r['train']['distinct_action_seqs'])
    row('F1이 갈린 그룹 비율', lambda r: r['train']['f1_varies'])
    row('free 토큰 / 궤적', lambda r: r['train']['free'], 1)
    row('fallback 토큰 / 궤적', lambda r: r['train']['fallback'], 2)
    row('궤적당 rollout 초', lambda r: r['train']['seconds_per_episode'])
    row('업데이트 step 수', lambda r: r['curve'].get('updates'))
    row('replay |Δlogp| (free)', lambda r: r['curve'].get('gap'), 4)
    row('clip 비율 (free)', lambda r: r['curve'].get('clip'), 4)
    row('k3 KL (free)', lambda r: r['curve'].get('kl'), 5)
    row('결정 |ratio−1|', lambda r: r['curve'].get('dec_ratio_dev'), 4)
    for lam in (0., .3, 1.):
        row(f'λ={lam}: 신호 있는 그룹', lambda r, lam=lam: r['lam'][lam]['signal'])
        row(f'λ={lam}: corr(A, 호출 수)', lambda r, lam=lam: r['lam'][lam]['corr'], 3)
    text = '\n'.join(L)+'\n'; Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(text); print(text)
    Path(a.out).with_suffix('.json').write_text(json.dumps(rows, indent=1, default=float))

if __name__ == '__main__':
    main()
