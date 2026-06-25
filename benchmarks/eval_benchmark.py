"""벤치마크 평가 — 독립 골드셋으로 검색 Recall + 최종 답변 F1/EM 측정.

지표:
  · retrieval recall : 정답이 든 출처(gold_page 또는 gold_keywords)를 검색 trajectory가 실제로 가져왔나.
  · answer F1 / EM   : Reader 최종 답변 vs gold_answers (SQuAD식 토큰 F1, 여러 gold 중 max).
출력: examples/BENCHMARK.md (per-item + 집계).

실행: PYTHONPATH=. RAG_CONFIG=<engine>/config.yaml python tests/eval_benchmark.py
(빌드된 인덱스 필요. reader.provider는 config 따름 — openai 권장(빠름), claude_code도 가능.)
"""
from __future__ import annotations

import json
import os
import string
import sys

from src.agent.runner import build_agent
from src.config import Config

_ARTICLES = {"a", "an", "the"}


def _norm(s: str):
    s = (s or "").lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    return [t for t in s.split() if t not in _ARTICLES]


def f1_em(pred: str, golds):
    """SQuAD식: 여러 gold 중 best F1/EM."""
    best_f1, best_em = 0.0, 0
    p = _norm(pred)
    for g in golds:
        gt = _norm(g)
        em = int(p == gt or " ".join(gt) in " ".join(p))  # 완전일치 또는 gold가 pred에 포함
        common = {}
        for t in p:
            if t in gt:
                common[t] = 1
        n = sum(min(p.count(t), gt.count(t)) for t in set(p) & set(gt))
        if n == 0 or not p or not gt:
            f1 = 1.0 if (not gt and not p) else 0.0
        else:
            prec, rec = n / len(p), n / len(gt)
            f1 = 2 * prec * rec / (prec + rec)
        best_f1 = max(best_f1, f1)
        best_em = max(best_em, em)
    return best_f1, best_em


def retrieval_recall(item, retrieved_text: str) -> int:
    """gold 출처가 검색 본문에 등장했나(1/0)."""
    txt = retrieved_text.lower()
    if item.get("gold_page"):
        if item["gold_page"].lower() in txt:
            return 1
    kws = item.get("gold_keywords") or []
    if kws:
        return int(any(k.lower() in txt for k in kws))  # 핵심어 중 하나라도 검색되면 출처 적중
    return 1  # 출처 타깃이 없는 항목(부정 질의 등)은 recall 대상 제외 → 1로 둠


def main():
    cfg = Config.load(os.environ.get("RAG_CONFIG", "config.yaml"))
    cfg._d.setdefault("agent", {})["max_loops"] = 8
    if os.environ.get("RAG_READER_PROVIDER"):      # 평가 시 provider 비교용 override
        cfg._d.setdefault("reader", {})["provider"] = os.environ["RAG_READER_PROVIDER"]
    provider = cfg.reader_config()["provider"]
    agent = build_agent(cfg)
    set_path = os.environ.get("RAG_EVAL_SET") or os.path.join(os.path.dirname(__file__), "eval_set.json")
    data = json.load(open(set_path, encoding="utf-8"))

    # 공정한 F1을 위해 간결 답변 요청(생성형↔짧은 gold 불일치 완화). 검색 trajectory엔 영향 없음.
    concise = " Answer in as few words as possible (just the key term/label, no explanation)."
    import time
    from collections import defaultdict
    rows, sum_f1, sum_em, sum_rec, n = [], 0.0, 0, 0, 0
    sum_loops, sum_time = 0, 0.0
    by_cat, by_tool = defaultdict(lambda: [0, 0.0, 0, 0]), defaultdict(lambda: [0, 0.0, 0, 0])
    for it in data["items"]:
        t0 = time.time()
        res = agent.run(it["question"] + concise)
        dt = time.time() - t0
        loops = res.get("loops") or len(res.get("trajectory", [])) or 1   # LLM 호출 수(루프)
        ans = res.get("answer", "") or ""
        retrieved = " ".join(str(t.get("tool_result", "")) for t in res.get("trajectory", []))
        rec = retrieval_recall(it, retrieved)
        f1, em = f1_em(ans, it["gold_answers"])
        sum_f1 += f1; sum_em += em; sum_rec += rec; n += 1
        sum_loops += loops; sum_time += dt
        for agg, k in ((by_cat, it.get("category", "?")), (by_tool, it.get("tool", "?"))):
            agg[k][0] += 1; agg[k][1] += f1; agg[k][2] += em; agg[k][3] += rec
        rows.append((it["id"], it.get("category", ""), it.get("tool", ""), rec, f1, em, ans[:80]))
        print(f"{it['id']} [{it.get('category')}/{it.get('tool')}]: recall={rec} F1={f1:.2f} EM={em}", flush=True)

    mr, mf, me = sum_rec / n, sum_f1 / n, sum_em / n
    ml, mt = sum_loops / n, sum_time / n
    # 머신리더블 한 줄(비교표 집계용)
    print(f"SUMMARY provider={provider} n={n} recall={mr:.4f} f1={mf:.4f} em={me:.4f} "
          f"avg_llm_calls={ml:.2f} avg_time_s={mt:.2f}", flush=True)

    def _bd(agg):
        out = []
        for k, (c, sf, se, sr) in sorted(agg.items()):
            out.append(f"| {k} | {c} | {sr/c:.0%} | {sf/c:.2f} | {se/c:.0%} |")
        return "\n".join(out)
    lines = [f"# BENCHMARK — 독립 골드셋 {n}문항 (Reader provider = {provider})\n",
             f"> 문서: {data['doc']}. 검색 Recall(출처 검색율) + Reader 최종 답변 F1/EM(SQuAD식).\n"
             "> 이 셋은 모델-비의존(gold=원문) → **Reader LLM을 바꿔 같은 셋으로 비교**할 수 있다.\n",
             f"\n## 집계\n\n| 지표 | 값 |\n|------|----|\n"
             f"| 검색 Recall (gold 출처 검색율) | **{mr:.2%}** |\n"
             f"| 답변 F1 (토큰, gold max) | **{mf:.3f}** |\n"
             f"| 답변 EM (완전/포함 일치) | **{me:.2%}** |\n"
             f"| 평균 LLM 호출 수 (루프) | **{ml:.2f}** |\n"
             f"| 평균 답변 시간 (초) | **{mt:.2f}** |\n",
             "\n## 카테고리별 (컨텍스트: 표/그림/텍스트)\n\n| category | n | Recall | F1 | EM |\n|----|:-:|:-:|:-:|:-:|",
             _bd(by_cat),
             "\n## 도구별 (의도 도구 커버리지)\n\n| tool | n | Recall | F1 | EM |\n|----|:-:|:-:|:-:|:-:|",
             _bd(by_tool),
             "\n## 문항별\n\n| ID | cat | tool | Recall | F1 | EM | 답변(발췌) |\n|----|----|----|:-:|:-:|:-:|----|"]
    for cid, cat, tool, rec, f1, em, a in rows:
        lines.append(f"| {cid} | {cat} | {tool} | {rec} | {f1:.2f} | {em} | {a.replace(chr(124), '/')} |")
    out = os.environ.get("RAG_BENCH_OUT") or os.path.join(cfg.base_dir, "benchmarks", "results", "BENCHMARK.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\n=== recall={mr:.2%} F1={mf:.3f} EM={me:.2%} → {out} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
