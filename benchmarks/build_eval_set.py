"""독립 Evaluation Set 생성기 — 파싱된 IR(원문)에서 gold를 추출/검증해 ~100문항 구성.

설계 원칙:
- 무오류: gold는 반드시 원문 청크에 실제로 존재(결정적 추출 또는 LLM생성 후 verbatim 검증).
- 모델 비의존: gold는 문서 내용에서 나옴 → Reader LLM(openai/claude/...)을 바꿔도 공통 기준.
- 균형: 컨텍스트(표/그림/텍스트) + 의도 도구(semantic/keyword/read_chunk/page_index/image_read)를 골고루.
- HW 엔지니어 시나리오: LLM이 실무 질문으로 문장화(원문 근거 한정).

출력: tests/eval_set_100.json  (각 item: id, category, tool, question, gold_answers, gold_page, gold_keywords)

실행: PYTHONPATH=. RAG_CONFIG=<engine>/config.yaml python tests/build_eval_set.py [--n 100]
(파싱 IR = data/ir_full/ 필요. LLM생성엔 OPENAI_API_KEY.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import string

from src.config import Config
from src.schema import ParsedDoc

_ART = {"a", "an", "the"}


def _norm(s):
    s = (s or "").lower()
    s = "".join(c if c not in string.punctuation else " " for c in s)
    return " ".join(t for t in s.split() if t not in _ART)


def _strip(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def load_blocks(ir_dir):
    blocks = []
    for f in sorted(glob.glob(os.path.join(ir_dir, "*.json"))):
        doc = ParsedDoc.from_dict(json.load(open(f, encoding="utf-8")))
        for b in doc.blocks:
            if b.page_label:
                blocks.append(b)
    return blocks


# ── 결정적 문항(무오류, gold=page_label) ──────────────────────────────
def det_items(blocks, want):
    items, seen = [], set()
    figs = [b for b in blocks if b.block_type == "figure" and b.heading]
    tabs = [b for b in blocks if b.block_type == "table" and len(_strip(b.text)) > 40]
    heads = [b for b in blocks if b.heading and len(b.heading.split()) >= 3]

    def add(cat, tool, q, page, kws):
        key = q.lower()
        if key in seen or not page:
            return
        seen.add(key)
        items.append({"category": cat, "tool": tool, "question": q,
                      "gold_answers": [page], "gold_page": page, "gold_keywords": kws})

    # 그림 위치 (page_index_search)
    for b in figs[:: max(1, len(figs) // (want // 3 + 1))]:
        topic = _strip(b.heading)[:60]
        add("figure-loc", "page_index_search",
            f"Which printed page label has the figure/diagram in the section about '{topic}'?",
            b.page_label, [topic.split()[0]] if topic else [])
    # 표 위치 (keyword_search)
    for b in tabs[:: max(1, len(tabs) // (want // 3 + 1))]:
        cell = _strip(b.text).split("|")
        phrase = next((c.strip() for c in cell if 3 <= len(c.strip().split()) <= 6), "")
        if phrase:
            add("table-loc", "keyword_search",
                f"Which printed page label contains the table mentioning '{phrase}'?",
                b.page_label, [phrase.split()[0]])
    # 섹션 위치 (page_index_search)
    for b in heads[:: max(1, len(heads) // (want // 3 + 1))]:
        h = _strip(b.heading)[:70]
        add("heading-loc", "page_index_search",
            f"On which printed page label is the section titled '{h}'?",
            b.page_label, [h.split()[0]])
    return items[:want]


# ── LLM 근거 문항(원문 한정, gold verbatim 검증) ───────────────────────
GEN_PROMPT = (
    "You create ONE benchmark QA item for a hardware/firmware engineer reading an ARM manual.\n"
    "SOURCE (printed page {label}, type {bt}):\n{text}\n\n"
    "Write a realistic question that engineer would ask, answerable ONLY from this SOURCE.\n"
    "The gold answer MUST be a short span (<=8 words) COPIED VERBATIM from the SOURCE text.\n"
    'Reply with JSON only: {{"q":"...","gold":"<verbatim span from source>",'
    '"tool":"semantic_search|keyword_search|read_chunk|page_index_search|image_read"}}'
)


def llm_items(blocks, cfg, want):
    import requests
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("(OPENAI_API_KEY 없음 → LLM 문항 생략, 결정적 문항만)")
        return []
    ep, model = "https://api.openai.com/v1/chat/completions", "gpt-4.1-mini"
    # 컨텍스트 다양화: 표/그림/텍스트 청크 샘플
    pool = []
    for bt, n in [("text", want // 2), ("table", want // 3), ("figure", want - want // 2 - want // 3)]:
        cand = [b for b in blocks if b.block_type == bt and len(_strip(b.text)) > 60]
        step = max(1, len(cand) // (n + 1))
        pool += [(b, bt) for b in cand[::step][:n]]
    items, seen = [], set()
    for b, bt in pool:
        src = _strip(b.text)[:1100]
        prompt = GEN_PROMPT.format(label=b.page_label, bt=bt, text=src)
        try:
            r = requests.post(ep, headers={"Authorization": f"Bearer {key}"}, timeout=60,
                              json={"model": model, "messages": [{"role": "user", "content": prompt}],
                                    "temperature": 0.4})
            out = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", out, re.S)
            obj = json.loads(m.group(0))
        except Exception:
            continue
        q, gold, tool = obj.get("q", ""), obj.get("gold", ""), obj.get("tool", "semantic_search")
        # 검증: gold가 원문에 verbatim(정규화) 존재해야 채택(무오류 보장)
        if not q or not gold or _norm(gold) not in _norm(src):
            continue
        if q.lower() in seen:
            continue
        seen.add(q.lower())
        cat = {"figure": "figure", "table": "table", "text": "concept"}[bt]
        # 의도 도구를 카테고리로 균형 고정(LLM 선택에 안 맡김 → 5종 골고루):
        #   concept→semantic_search, table→read_chunk/keyword 교대, figure→image_read
        if cat == "concept":
            tool = "semantic_search"
        elif cat == "table":
            tool = "read_chunk" if len(items) % 2 == 0 else "keyword_search"
        else:
            tool = "image_read"
        items.append({"category": cat, "tool": tool, "question": q,
                      "gold_answers": [gold], "gold_page": b.page_label,
                      "gold_keywords": [w for w in _norm(gold).split()[:3]]})
        if len(items) >= want:
            break
    return items


# ── 부정 문항(범위 밖, gold=없음) ─────────────────────────────────────
# 아키텍처 레퍼런스 매뉴얼에 '확실히 없는' 물리/보드/상용 사양(무오류 보장).
NEG = [
    "What is the die area of the processor in square millimetres?",
    "Which company manufactured this specific silicon chip?",
    "What is the typical power consumption in milliwatts?",
    "What is the pinout of the JTAG debug connector?",
    "Which physical GPIO pins are used for the debug UART?",
    "What is the maximum clock frequency in MHz for this part?",
    "What is the unit price of this processor?",
    "What is the operating temperature range in degrees Celsius?",
    "What package type (BGA/QFP) is this chip delivered in?",
    "How many transistors does this implementation contain?",
]


def neg_items():
    return [{"category": "negative", "tool": "any", "question": q,
             "gold_answers": ["no", "not in this document", "does not"],
             "gold_keywords": []} for q in NEG]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ir", default="data/ir_full")
    args = ap.parse_args()
    cfg = Config.load(os.environ.get("RAG_CONFIG", "config.yaml"))
    blocks = load_blocks(args.ir)
    print(f"IR 블록(page_label 보유): {len(blocks)}")

    neg = neg_items()
    det = det_items(blocks, want=45)
    llm = llm_items(blocks, cfg, want=args.n - len(det) - len(neg))
    alli = det + llm + neg
    for i, it in enumerate(alli, 1):
        it["id"] = f"E{i:03d}"
    out = os.path.join(os.path.dirname(__file__), "eval_set_100.json")
    json.dump({"doc": "ARMv8-Reference-Manual (expanded parse, data/ir_full)",
               "note": "독립 벤치마크. gold는 원문에서 추출/검증(무오류). Reader LLM 교체 시 공통 기준.",
               "items": alli}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from collections import Counter
    print("총", len(alli), "문항")
    print("category:", dict(Counter(i["category"] for i in alli)))
    print("tool:", dict(Counter(i["tool"] for i in alli)))
    print("→", out)


if __name__ == "__main__":
    main()
