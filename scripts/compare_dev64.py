"""Paired dev64 comparison of evaluated policies against the native baseline.

Reads per-question episodes from evaluation runs under ROOT/results and writes a
Markdown report: summary table, paired deltas with bootstrap 95% CIs, per-question
changes, and where token savings come from.

Usage: python scripts/compare_dev64.py --policy LABEL=RUN:POLICY ... --out results/x.md
The first --policy is the reference (baseline).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common import ROOT, SPLIT

def load_gold():
    sys.path.insert(0, str(ROOT/'source/musique_scorer'))
    gold = {}
    for split in ['train', 'dev']:
        for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
            q = json.loads(line)
            if q['id'] in SPLIT['validation']:
                gold[q['id']] = [q['answer']]+q['answer_aliases']
    return gold

def load(run, policy, gold):
    from metrics.answer import compute_exact
    rows = {}
    for qid in SPLIT['validation']:
        e = json.loads((ROOT/'results'/run/'episodes'/f'{policy}_{qid}.json').read_text())
        rows[qid] = dict(f1=e['f1'], em=max(compute_exact(a, e['answer']) for a in gold[qid]), calls=e['tool_calls'],
                         gen=e['generated_tokens'], inp=e['prompt_tokens'], err=bool(e['errors']))
    return rows

def boot(delta, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), (n, len(delta)))
    means = delta[idx].mean(1)
    return np.percentile(means, [2.5, 97.5])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', action='append', required=True, help='LABEL=RUN:POLICY; first is the reference')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    gold = load_gold(); qids = SPLIT['validation']
    specs = [(s.split('=', 1)[0], *s.split('=', 1)[1].split(':', 1)) for s in a.policy]
    data = {label: load(run, pol, gold) for label, run, pol in specs}
    ref = specs[0][0]
    col = lambda label, k: np.array([data[label][q][k] for q in qids], dtype=float)
    out = [f'# dev64 비교: {" / ".join(l for l, _, _ in specs)}', '',
           f'MuSiQue 2hop dev64({len(qids)}문항), greedy. 기준: {ref}. 1문항 = F1 {100/len(qids):.2f}%p.', '',
           '## 요약', '', '| 조건 | F1 (%) | EM (%) | 생성 토큰 | 입력 토큰 | 도구 호출 | 오류 |', '|---|---:|---:|---:|---:|---:|---:|']
    for label, _, _ in specs:
        out.append(f"| {label} | {100*col(label,'f1').mean():.2f} | {100*col(label,'em').mean():.2f} | {col(label,'gen').mean():.1f} | "
                   f"{col(label,'inp').mean():,.0f} | {col(label,'calls').mean():.2f} | {int(col(label,'err').sum())} |")
    out += ['', f'## {ref} 대비 (문항별 paired)', '',
            '| 조건 | ΔF1 (%p) [95% CI] | ΔEM (%p) | F1 오름/내림/같음 | EM 0→1 / 1→0 | 생성 토큰 비율 | 입력 토큰 비율 |', '|---|---|---:|---|---|---:|---:|']
    for label, _, _ in specs[1:]:
        d = col(label, 'f1')-col(ref, 'f1'); lo, hi = boot(d)
        em_up = int(((col(ref, 'em') == 0) & (col(label, 'em') == 1)).sum()); em_dn = int(((col(ref, 'em') == 1) & (col(label, 'em') == 0)).sum())
        out.append(f"| {label} | {100*d.mean():+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] | {100*(col(label,'em')-col(ref,'em')).mean():+.2f} | "
                   f"{int((d>1e-9).sum())}/{int((d<-1e-9).sum())}/{int((abs(d)<=1e-9).sum())} | {em_up} / {em_dn} | "
                   f"{col(label,'gen').sum()/col(ref,'gen').sum():.3f} | {col(label,'inp').sum()/col(ref,'inp').sum():.3f} |")
    out += ['', '## 토큰 변화의 출처 (기준 대비, 입력+생성 토큰 합)', '',
            '| 조건 | 둘 다 정답(EM) | 둘 다 오답 | 정답이 바뀜 | 합계 |', '|---|---:|---:|---:|---:|']
    for label, _, _ in specs[1:]:
        tok = (col(label, 'inp')+col(label, 'gen'))-(col(ref, 'inp')+col(ref, 'gen'))
        both_ok = (col(ref, 'em') == 1) & (col(label, 'em') == 1); both_bad = (col(ref, 'em') == 0) & (col(label, 'em') == 0)
        changed = ~(both_ok | both_bad)
        out.append(f"| {label} | {tok[both_ok].sum():+,.0f} ({int(both_ok.sum())}문항) | {tok[both_bad].sum():+,.0f} ({int(both_bad.sum())}문항) | "
                   f"{tok[changed].sum():+,.0f} ({int(changed.sum())}문항) | {tok.sum():+,.0f} |")
    out += ['', '## F1이 바뀐 문항', '']
    for label, _, _ in specs[1:]:
        changed = [q for q in qids if abs(data[label][q]['f1']-data[ref][q]['f1']) > 1e-9]
        out.append(f'**{label}** ({len(changed)}문항)')
        out.append('')
        out.append(f'| 문항 | {ref} F1 | {label} F1 | 도구 호출 |')
        out.append('|---|---:|---:|---|')
        for q in changed:
            out.append(f"| `{q}` | {data[ref][q]['f1']:.2f} | {data[label][q]['f1']:.2f} | {data[ref][q]['calls']} → {data[label][q]['calls']} |")
        out.append('')
    text = '\n'.join(out)+'\n'
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(text)

if __name__ == '__main__':
    main()
