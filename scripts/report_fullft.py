"""Final report for the full-FT GRPO runs (A1 native, B1 head+LLM, C1 head only): template tables 1-7.

Reads ROOT/results/RUN/{eval/i*/<qid>.json, eval/i*/summary.json, curves.json, manifest.json, status.json}.
Usage: python scripts/report_fullft.py --out ROOT/analysis/fullft_report.md [--final 4]
"""
import argparse, collections, json, re
from pathlib import Path
import numpy as np
from common import ROOT, REPO

SPLIT = json.loads((REPO/'configs/split_mixhop.json').read_text())
RUNS = [('A1 native', 'mh_native_A1'), ('B1 head+LLM', 'mh_head_B1'), ('C1 head only', 'mh_headonly_C1'), ('C2 head only lr 2e-4', 'mh_headonly_C2_lr2e-4')]
HOP = lambda q: q.split('__')[0][:4]

def gold_ids():
    cid = {}
    for line in (ROOT/'corpora/musique_shared_20261023/documents.jsonl').open():
        d = json.loads(line); cid[(d['title'], d['text'])] = d['chunk_id']
    want = set(SPLIT['validation']); gold = {}
    for line in (ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open():
        q = json.loads(line)
        if q['id'] in want:
            gold[q['id']] = {cid[(p['title'], p['paragraph_text'])] for p in q['paragraphs'] if p['is_supporting']}
    return gold

def coverage(e, gold):
    """(calls until all gold paragraphs observed or None, number of gold observed)."""
    seen = set(); k = None
    for i, h in enumerate(e['trajectory']):
        res = h['result']
        if isinstance(res, list):
            seen |= {r['chunk_id'] for r in res if isinstance(r, dict) and 'chunk_id' in r}
        if k is None and gold <= seen:
            k = i+1
    return k, len(gold & seen)

def episodes(run, it):
    d = ROOT/'results'/run/'eval'/f'i{it}'
    if it == 0 and run.startswith('mh_headonly_C') and not d.exists():   # same head + frozen LLM + greedy -> identical to C1 i0
        d = ROOT/'results'/'mh_headonly_C1'/'eval'/'i0'
    if not all((d/f'{q}.json').exists() for q in SPLIT['validation']):
        return None
    return {q: json.loads((d/f'{q}.json').read_text()) for q in SPLIT['validation']}

def boot(x, n=10000, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x))); return np.percentile(x[idx].mean(1), [2.5, 97.5])

def roles(e):
    out = collections.Counter()
    for u in e['reader_calls']:
        raw = u.get('raw_json'); n = u.get('generated_tokens', 0)
        if not raw or not n or 'decision_tokens' in u:
            continue
        ids = len(re.findall(r'c[0-9a-f]{24}', raw)); idtok = min(n, 23*ids)  # a quoted 25-char chunk id is ~23 Qwen tokens (digits split)
        try: j = json.loads(raw)
        except Exception: j = {}
        if 'answer' in j:
            out['answer+syntax'] += n-idtok; out['support_ids'] += idtok
        elif 'chunk_ids' in j:
            out['read_chunk'] += n
        else:
            out['search_args'] += n
    return out

def summary(E):
    v = list(E.values()); g = lambda k: np.array([e[k] for e in v], dtype=float)
    calls = g('tool_calls')
    return dict(f1=100*g('f1').mean(), em=100*g('em').mean(), calls=calls.mean(), three=np.mean(calls >= 3), gen=g('generated_tokens').mean(),
                free=g('free_tokens').mean(), inp=g('prompt_tokens').mean(), empty=np.mean([e['answer'] == '' for e in v]),
                lat=np.mean([.026*e['generated_tokens']+.218*e['prompt_tokens']/1000 for e in v]), sec=g('seconds').mean(),
                hist=[np.mean(calls <= 1), np.mean(calls == 2), np.mean(calls == 3), np.mean((calls >= 4) & (calls < 8)), np.mean(calls >= 8)])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--final', type=int, default=4); a = ap.parse_args()
    L = ['# full FT GRPO 결과: native vs head ARAG, head-only', '',
         '보상 R = F1·0.8^(도구 호출 수), mixed-hop(2/3/4hop), 반복마다 192문항×8궤적×4회, dev192 greedy. 판정은 iteration %d 고정, 기준점은 같은 arm의 iteration 0.' % a.final, '']
    data = {}
    for label, run in RUNS:
        d = ROOT/'results'/run
        if not (d/'status.json').exists():
            continue
        data[label] = dict(run=run, status=json.loads((d/'status.json').read_text()),
                           curves=json.loads((d/'curves.json').read_text()) if (d/'curves.json').exists() else [],
                           manifest=json.loads((d/'manifest.json').read_text()), ev={i: episodes(run, i) for i in range(a.final+1)})
    # Table 1
    L += ['## 표 1. 주 결과', '', '| 조건 | F1 (%) | EM (%) | 도구 호출 | 3회+ 비율 | 생성 토큰 | 입력 토큰 | 지연 proxy (s) | 빈 답 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label, D in data.items():
        for it in sorted(i for i, E in D['ev'].items() if E):
            s = summary(D['ev'][it])
            L.append(f"| {label} i{it} | {s['f1']:.2f} | {s['em']:.2f} | {s['calls']:.2f} | {100*s['three']:.0f}% | {s['gen']:.1f} | {s['inp']:,.0f} | {s['lat']:.2f} | {100*s['empty']:.0f}% |")
    # Table 2
    L += ['', '## 표 2. 판정 (iteration %d vs 0, paired bootstrap 95%% CI)' % a.final, '',
          '| arm | ΔF1 (%p) [CI] | Δ호출 [CI] | 호출 변화율 | Δ지연 proxy | F1 오름/내림/같음 | EM 0→1 / 1→0 | 판정 |', '|---|---|---|---:|---:|---|---|---|']
    for label, D in data.items():
        E0, E1 = D['ev'].get(0), D['ev'].get(a.final)
        if not (E0 and E1):
            L.append(f'| {label} | 미완료 | | | | | | |'); continue
        q = SPLIT['validation']; f = lambda E, k: np.array([E[x][k] for x in q], dtype=float)
        df = f(E1, 'f1')-f(E0, 'f1'); dc = f(E1, 'tool_calls')-f(E0, 'tool_calls'); lf, hf = boot(df); lc, hc = boot(dc)
        rate = dc.mean()/f(E0, 'tool_calls').mean(); lat = lambda E: .026*f(E, 'generated_tokens')+.218*f(E, 'prompt_tokens')/1000
        dl = lat(E1).mean()/lat(E0).mean()-1
        up = int(((f(E0, 'em') == 0) & (f(E1, 'em') == 1)).sum()); dn = int(((f(E0, 'em') == 1) & (f(E1, 'em') == 0)).sum())
        acc_ok = df.mean() >= 0 and lf > -3/100*1
        shorter = hc < 0
        if shorter and -.30 <= rate <= -.10 and acc_ok: verdict = 'best (호출 −10~30%, 정확도 ≥ 동등)'
        elif rate < -.5 and df.mean() < 0: verdict = '경보 2: 급격히 짧아지고 정확도 하락'
        elif abs(rate) < .05: verdict = '경보 1: 호출 수 변화 없음'
        elif shorter and acc_ok: verdict = '짧아짐 + 정확도 유지 (목표 범위 밖)'
        else: verdict = '미달'
        L.append(f"| {label} | {100*df.mean():+.2f} [{100*lf:+.2f}, {100*hf:+.2f}] | {dc.mean():+.2f} [{lc:+.2f}, {hc:+.2f}] | {100*rate:+.1f}% | {100*dl:+.1f}% | "
                 f"{int((df>1e-9).sum())}/{int((df<-1e-9).sum())}/{int((abs(df)<=1e-9).sum())} | {up} / {dn} | {verdict} |")
    # Table 3
    L += ['', '## 표 3. 학습 추이', '', '| arm | it | train F1 | train 호출 | 3회+ | 평균 보상 | 신호 그룹 | KL | replay gap | head 변화율 | updates |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fm = lambda x, n=3: '—' if x is None else f'{x:.{n}f}'
    for label, D in data.items():
        for c in D['curves']:
            L.append(f"| {label} | {c['iteration']} | {100*c['mean_f1']:.1f} | {c['mean_calls']:.2f} | {fm(c.get('three_plus_calls'), 2)} | {c['mean_reward']:.3f} | {c['groups_with_signal']:.2f} | "
                     f"{fm(c.get('kl'), 4)} | {fm(c.get('gap'), 4)} | {fm(c.get('head_changed_vs_loaded'), 4)} | {c['updates']} |")
    # Table 4
    L += ['', '## 표 4. 도구 호출 수 분포 (dev192 문항 비율)', '', '| 조건 | 0~1회 | 2회 | 3회 | 4~7회 | 8회(강제) |', '|---|---:|---:|---:|---:|---:|']
    for label, D in data.items():
        for it in sorted(i for i, E in D['ev'].items() if E):
            h = summary(D['ev'][it])['hist']; L.append(f'| {label} i{it} | ' + ' | '.join(f'{100*x:.0f}%' for x in h) + ' |')
    # Table 5
    L += ['', '## 표 5. 생성 토큰 구성 (궤적당, dev192)', '', '| 조건 | 검색 인자 | read_chunk | FINAL(답+문법) | support_ids(추정) | 합계 |', '|---|---:|---:|---:|---:|---:|']
    for label, D in data.items():
        for it in sorted(i for i, E in D['ev'].items() if E):
            c = collections.Counter()
            for e in D['ev'][it].values(): c.update(roles(e))
            n = len(D['ev'][it]); L.append(f"| {label} i{it} | {c['search_args']/n:.1f} | {c['read_chunk']/n:.1f} | {c['answer+syntax']/n:.1f} | {c['support_ids']/n:.1f} | {sum(c.values())/n:.1f} |")
    # Table 6
    L += ['', '## 표 6. 정확도 변화의 출처 (iteration %d vs 0, 문항 수)' % a.final, '', '| arm | 빈 답→정답 | 빈 답→오답 | 오답→정답 | 정답→오답 | 정답→빈 답 | 호출 줄고 정답 유지 |', '|---|---:|---:|---:|---:|---:|---:|']
    for label, D in data.items():
        E0, E1 = D['ev'].get(0), D['ev'].get(a.final)
        if not (E0 and E1): continue
        cnt = collections.Counter()
        for q in SPLIT['validation']:
            a0, a1 = E0[q], E1[q]; ok0, ok1 = a0['em'] == 1, a1['em'] == 1; emp0, emp1 = a0['answer'] == '', a1['answer'] == ''
            if emp0 and ok1: cnt['e2c'] += 1
            if emp0 and not emp1 and not ok1: cnt['e2w'] += 1
            if not emp0 and not ok0 and ok1: cnt['w2c'] += 1
            if ok0 and not ok1 and not emp1: cnt['c2w'] += 1
            if ok0 and emp1: cnt['c2e'] += 1
            if ok0 and ok1 and a1['tool_calls'] < a0['tool_calls']: cnt['short'] += 1
        L.append(f"| {label} | {cnt['e2c']} | {cnt['e2w']} | {cnt['w2c']} | {cnt['c2w']} | {cnt['c2e']} | {cnt['short']} |")
    # Table 8: per hop and tool-use efficiency
    gold = gold_ids()
    L += ['', '## 표 8. hop별 결과와 도구 사용 효율 (dev192)', '',
          '| 조건 | hop | F1 (%) | 도구 호출 | 근거 모두 관측 | 낭비 호출/문항 (근거 다 본 뒤) | 부족 종료 (근거 덜 보고 FINAL) |', '|---|---|---:|---:|---:|---:|---:|']
    for label, D in data.items():
        for it in sorted(i for i, E in D['ev'].items() if E):
            for hop in ['2hop', '3hop', '4hop', '전체']:
                E = [e for q, e in D['ev'][it].items() if hop == '전체' or HOP(q) == hop]
                cov = [coverage(e, gold[e['qid']]) for e in E]
                full = [k for k, _ in cov if k]; waste = np.mean([e['tool_calls']-k if k else 0 for e, (k, _) in zip(E, cov)])
                under = np.mean([k is None and not e['forced_final'] for e, (k, _) in zip(E, cov)])
                L.append(f"| {label} i{it} | {hop} | {100*np.mean([e['f1'] for e in E]):.2f} | {np.mean([e['tool_calls'] for e in E]):.2f} | {100*len(full)/len(E):.0f}% | {waste:.2f} | {100*under:.0f}% |")
    # Table 7
    L += ['', '## 표 7. 공평성·건전성', '', '| 항목 | ' + ' | '.join(data) + ' |', '|---|' + '---:|'*len(data)]
    def row(name, fn):
        L.append(f'| {name} | ' + ' | '.join(str(fn(D)) for D in data.values()) + ' |')
    row('상태', lambda D: D['status']['phase'])
    row('결정 T / LLM 학습', lambda D: f"{D['manifest'].get('decision_temperature') or 'head'} / {'고정' if D['manifest'].get('freeze_llm') else 'full FT'}")
    row('lr (LLM / head)', lambda D: f"{'—' if D['manifest'].get('freeze_llm') else D['manifest']['lr']} / {D['manifest']['lr_head'] if D['manifest']['arm'] == 'head' else '—'}")
    row('총 궤적 / 학습 궤적', lambda D: f"{sum(c['trajectories'] for c in D['curves'])} / {sum(c['trained_trajectories'] for c in D['curves'])}")
    row('업데이트 step', lambda D: sum(c['updates'] for c in D['curves']))
    row('최종 KL / replay gap', lambda D: f"{fm(D['curves'][-1].get('kl'), 4)} / {fm(D['curves'][-1].get('gap'), 4)}" if D['curves'] else '—')
    row('GPU 최대 (GB) / 시간 (h)', lambda D: f"{D['status'].get('gpu_peak_gb', 0):.0f} / {D['status']['seconds']/3600:.1f}")
    row('dev192 오류 (최종)', lambda D: (lambda E: sum(bool(e['errors']) for e in E.values()) if E else '—')(D['ev'].get(a.final)))
    L += ['', '주: A1과 C1은 GPU를 함께 써서 wall-clock 초는 비교하지 않음. 시간은 지연 proxy(0.026·생성 + 0.218·입력 k토큰, R² 0.98)로 비교함.']
    text = '\n'.join(L)+'\n'; Path(a.out).write_text(text); print(text)

if __name__ == '__main__':
    main()
