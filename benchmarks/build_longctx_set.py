"""long-context/멀티홉 평가셋 생성기 (내부 테스트). 단답 grounding이 아니라
'여러 출처를 모아 통합/집계/참조추적'하는 능력을 측정.

유형(전부 원문 근거·gold 검증 → 무오류):
- aggregation : 표 전체를 읽어 모든 항목 나열. gold_set = 표의 행 키들(결정적). → synthesis F1(집합 일치).
- crossref    : 청크 본문이 다른 페이지("E2-XXXX")를 참조 → 두 출처(원본+참조) 필요. → multi-source recall.
- multihop    : 같은 주제 2청크(다른 페이지)를 모아야 답. LLM이 문장화, gold는 두 출처 union에서 검증.

출력: benchmarks/eval_longctx.json
실행: PYTHONPATH=. python benchmarks/build_longctx_set.py
"""
from __future__ import annotations

import glob
import json
import os
import re

from src.schema import ParsedDoc

IR = "data/ir_full"
PAGE = re.compile(r"\b([A-Z]\d{0,2}-\d{3,5})\b")  # ARM 페이지 라벨


def _strip(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def load_blocks():
    out = []
    for f in sorted(glob.glob(os.path.join(IR, "*.json"))):
        for b in ParsedDoc.from_dict(json.load(open(f, encoding="utf-8"))).blocks:
            if b.page_label:
                out.append(b)
    return out


_HEADERISH = {"attribute", "field", "name", "bit", "bits", "register", "registers",
              "meaning", "description", "value", "type", "encoding", "purpose"}


def table_rows(md):
    """markdown 표 → 행 첫 셀(키). 헤더 행(--- 직전 행)·컬럼헤더성 토큰 제외."""
    lines = [ln.strip() for ln in md.splitlines()]
    header_idx = {i - 1 for i, ln in enumerate(lines)
                  if i > 0 and set(ln) <= set("|-: ") and "-" in ln}
    keys = []
    for i, ln in enumerate(lines):
        if i in header_idx or not ln.startswith("|") or set(ln) <= set("|-: "):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        k = cells[0] if cells else ""
        if k and not set(k) <= set("-: ") and k.lower() not in _HEADERISH:
            keys.append(k)
    return keys


_BOILER = ("copyright", "non-confidential", "arm limited", "rights reserved",
           "all rights", "affiliates", "proprietary notice")


def _is_boiler(s):
    s = (s or "").lower()
    return any(b in s for b in _BOILER) or bool(re.fullmatch(r"[A-Z]\d{0,2}-\d{3,5}\.?", (s or "").strip()))


def _deboiler(s):
    return " ".join(ln for ln in (s or "").splitlines() if not _is_boiler(ln))


def gen_aggregation(blocks, want):
    items = []
    tabs = [b for b in blocks if b.block_type == "table"]
    for b in tabs:
        keys = table_rows(b.text)
        keys = [k for k in keys if 1 <= len(k.split()) <= 5]
        keys = list(dict.fromkeys(keys))  # dedup, 순서유지
        if 3 <= len(keys) <= 8:
            topic = _strip(b.heading or "")[:50]
            q = (f"Read the entire table on page {b.page_label}"
                 + (f" (section '{topic}')" if topic else "")
                 + f" and list ALL of its row entries (there are {len(keys)}).")
            items.append({"category": "aggregation", "question": q,
                          "gold_answers": keys, "gold_set": keys,
                          "gold_sources": [b.page_label]})
        if len(items) >= want:
            break
    return items[:want]


def gen_crossref(blocks, want):
    items, seen = [], set()
    for b in blocks:
        txt = _strip(b.text)
        refs = [r for r in PAGE.findall(txt) if r != b.page_label]
        refs = list(dict.fromkeys(refs))
        if refs and len(txt) > 80:
            ref = refs[0]
            topic = _strip(b.heading or "")[:50] or txt[:40]
            key = (b.page_label, ref)
            if key in seen:
                continue
            seen.add(key)
            q = (f"The content on page {b.page_label}"
                 + (f" about '{topic}'" if topic else "")
                 + " refers the reader to another printed page for related detail. "
                 "Which page does it point to? (Both pages must be consulted.)")
            items.append({"category": "crossref", "question": q,
                          "gold_answers": [ref], "gold_sources": [b.page_label, ref]})
        if len(items) >= want:
            break
    return items[:want]


def gen_multihop(blocks, want):
    """같은 heading을 가진 서로 다른 페이지 2청크 → LLM이 통합 질문 생성, gold는 union에서 검증."""
    import requests
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return []
    from collections import defaultdict
    byh = defaultdict(list)
    for b in blocks:
        if b.block_type == "text" and b.heading and len(_strip(b.text)) > 80:
            byh[_strip(b.heading)[:60]].append(b)
    pairs = []
    for h, bs in byh.items():
        pages = {}
        for b in bs:
            pages.setdefault(b.page_label, b)
        if len(pages) >= 2:
            ps = list(pages.values())[:2]
            pairs.append((h, ps[0], ps[1]))
    items = []
    for h, b1, b2 in pairs:
        if _is_boiler(h) or h.lower().startswith("chapter"):
            continue
        s1, s2 = _deboiler(_strip(b1.text))[:700], _deboiler(_strip(b2.text))[:700]
        if len(s1) < 120 or len(s2) < 120:   # 보일러 제거 후 실질 내용이 적으면 스킵
            continue
        prompt = (
            "Two excerpts from the SAME section of an ARM manual, on different pages.\n"
            f"PAGE {b1.page_label}:\n{s1}\n\nPAGE {b2.page_label}:\n{s2}\n\n"
            "Write ONE question a hardware engineer would ask that requires BOTH excerpts to answer "
            "(integration across the two pages). The gold answer must be a short span copied VERBATIM "
            "from EITHER excerpt.\n"
            'Reply JSON only: {"q":"...","gold":"<verbatim span>"}'
        )
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions",
                              headers={"Authorization": f"Bearer {key}"}, timeout=60,
                              json={"model": "gpt-4.1-mini", "temperature": 0.4,
                                    "messages": [{"role": "user", "content": prompt}]})
            o = json.loads(re.search(r"\{.*\}", r.json()["choices"][0]["message"]["content"], re.S).group(0))
        except Exception:
            continue
        q, gold = o.get("q", ""), o.get("gold", "")
        union = (s1 + " " + s2).lower()
        if q and gold and not _is_boiler(gold) and gold.lower() in union:
            items.append({"category": "multihop", "question": q,
                          "gold_answers": [gold], "gold_sources": [b1.page_label, b2.page_label]})
        if len(items) >= want:
            break
    return items[:want]


def main():
    blocks = load_blocks()
    print(f"IR 블록: {len(blocks)}")
    agg = gen_aggregation(blocks, 8)
    xref = gen_crossref(blocks, 6)
    mh = gen_multihop(blocks, 6)
    items = agg + xref + mh
    for i, it in enumerate(items, 1):
        it["id"] = f"L{i:02d}"
    out = os.path.join(os.path.dirname(__file__), "eval_longctx.json")
    json.dump({"doc": "ARMv8 매뉴얼의 '파싱된 일부'(~1800p, 20조각, 전체 6354p 아님) — data/ir_full",
               "note": "long-context/멀티홉 — 여러 출처 통합/집계/참조추적. 지표: multi-source recall + synthesis/token F1. (파싱된 구간 한정)",
               "items": items}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from collections import Counter
    print("총", len(items), "문항 |", dict(Counter(i["category"] for i in items)), "→", out)


if __name__ == "__main__":
    main()
