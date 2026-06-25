"""long-context/멀티홉 평가 (내부). 단답 grounding이 아니라 '여러 출처 통합' 능력 측정.

지표:
- multi-source recall : gold_sources(2+ 페이지) 중 검색 trajectory에 등장한 비율(부분점수) + 전부-등장률.
- synthesis coverage  : aggregation은 gold_set(표 전체 항목) 중 답변이 포함한 비율(=전체를 읽고 나열했나).
                        그 외(crossref/multihop)는 gold_answers token F1.
- 평균 LLM 호출수·시간.

실행: PYTHONPATH=. RAG_CONFIG=<cfg> RAG_READER_PROVIDER=<openai|claude_code> python benchmarks/eval_longctx_benchmark.py
"""
from __future__ import annotations

import json
import os
import string
import time
from collections import defaultdict

from src.agent.runner import build_agent
from src.config import Config

_ART = {"a", "an", "the"}


def _norm(s):
    s = (s or "").lower()
    s = "".join(c if c not in string.punctuation else " " for c in s)
    return [t for t in s.split() if t not in _ART]


def f1_em(pred, golds):
    """SQuAD식 token F1/EM (여러 gold 중 max). eval_benchmark와 동일 로직(인라인)."""
    best_f1, best_em = 0.0, 0
    p = _norm(pred)
    for g in golds:
        gt = _norm(g)
        em = int(p == gt or " ".join(gt) in " ".join(p))
        n = sum(min(p.count(t), gt.count(t)) for t in set(p) & set(gt))
        if n == 0 or not p or not gt:
            f1 = 1.0 if (not gt and not p) else 0.0
        else:
            prec, rec = n / len(p), n / len(gt)
            f1 = 2 * prec * rec / (prec + rec)
        best_f1 = max(best_f1, f1)
        best_em = max(best_em, em)
    return best_f1, best_em


def multi_source_recall(item, retrieved):
    srcs = item.get("gold_sources") or []
    if not srcs:
        return 1.0, 1
    txt = retrieved.lower()
    found = sum(1 for s in srcs if s.lower() in txt)
    return found / len(srcs), int(found == len(srcs))


def synthesis(item, answer):
    if item.get("gold_set"):  # 표 전체 나열 — 항목 커버리지
        gs = item["gold_set"]
        a = _norm(answer)
        hit = sum(1 for k in gs if all(t in a for t in _norm(k)))
        return hit / len(gs)
    f1, _ = f1_em(answer, item["gold_answers"])
    return f1


def main():
    cfg = Config.load(os.environ.get("RAG_CONFIG", "config.yaml"))
    cfg._d.setdefault("agent", {})["max_loops"] = 10  # 멀티홉은 더 많은 스텝 허용
    if os.environ.get("RAG_READER_PROVIDER"):
        cfg._d.setdefault("reader", {})["provider"] = os.environ["RAG_READER_PROVIDER"]
    provider = cfg.reader_config()["provider"]
    agent = build_agent(cfg)
    data = json.load(open(os.path.join(os.path.dirname(__file__), "eval_longctx.json"), encoding="utf-8"))

    n = 0
    s_msr = s_all = s_syn = s_loops = 0.0
    s_time = 0.0
    by = defaultdict(lambda: [0, 0.0, 0.0, 0])  # cat -> [n, msr, syn, allsrc]
    rows = []
    for it in data["items"]:
        t0 = time.time()
        res = agent.run(it["question"])
        dt = time.time() - t0
        ans = res.get("answer", "") or ""
        retrieved = " ".join(str(t.get("tool_result", "")) for t in res.get("trajectory", []))
        msr, allsrc = multi_source_recall(it, retrieved)
        syn = synthesis(it, ans)
        loops = res.get("loops") or len(res.get("trajectory", [])) or 1
        n += 1; s_msr += msr; s_all += allsrc; s_syn += syn; s_loops += loops; s_time += dt
        c = by[it["category"]]; c[0] += 1; c[1] += msr; c[2] += syn; c[3] += allsrc
        rows.append((it["id"], it["category"], msr, allsrc, syn, ans[:70]))
        print(f"{it['id']} [{it['category']}]: msrc={msr:.2f} all={allsrc} syn={syn:.2f}", flush=True)

    mmsr, mall, msyn, ml, mt = s_msr/n, s_all/n, s_syn/n, s_loops/n, s_time/n
    print(f"SUMMARY provider={provider} n={n} multi_src_recall={mmsr:.4f} all_src_rate={mall:.4f} "
          f"synthesis={msyn:.4f} avg_llm_calls={ml:.2f} avg_time_s={mt:.2f}", flush=True)

    L = [f"# LONG-CONTEXT BENCHMARK — {n}문항 (Reader={provider})\n",
         f"> {data['doc']}\n> 여러 출처 통합/집계/참조추적 측정 — 단답 grounding과 별개.\n",
         "\n## 집계\n\n| 지표 | 값 |\n|----|----|\n"
         f"| multi-source recall (필요 출처 중 검색됨) | **{mmsr:.2%}** |\n"
         f"| all-sources rate (필요 출처 전부 검색) | **{mall:.2%}** |\n"
         f"| synthesis (집계 커버리지 / token F1) | **{msyn:.3f}** |\n"
         f"| 평균 LLM 호출 | **{ml:.2f}** |  평균 시간 | **{mt:.2f}s** |\n",
         "\n## 카테고리별\n\n| category | n | multi-src recall | all-src | synthesis |\n|----|:-:|:-:|:-:|:-:|"]
    for k, (c, m, sy, al) in sorted(by.items()):
        L.append(f"| {k} | {c} | {m/c:.0%} | {al/c:.0%} | {sy/c:.2f} |")
    L.append("\n## 문항별\n\n| ID | cat | multi-src | all-src | synthesis | 답변(발췌) |\n|----|----|:-:|:-:|:-:|----|")
    for cid, cat, m, al, sy, a in rows:
        L.append(f"| {cid} | {cat} | {m:.2f} | {al} | {sy:.2f} | {a.replace(chr(124),'/')} |")
    out = os.environ.get("RAG_BENCH_OUT") or os.path.join(cfg.base_dir, "benchmarks", "results", "LONGCTX.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
