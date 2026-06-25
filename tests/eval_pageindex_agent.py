"""page_index + agentic RAG 본연 역할 라이브 평가 (어려운 20 케이스).

각 케이스: 질문 + 의도(어떤 도구/단계를 거쳐야 맞는지). 실제 agent를 돌려
단계별 trajectory와 의도 충족 여부를 examples/EVAL_pageindex_agent.md 에 세부 기록.

실행: PYTHONPATH=. RAG_CONFIG=<engine>/config.yaml python tests/eval_pageindex_agent.py
(빌드된 인덱스 필요. 실제 Upstage·GPT-4.1 mini 호출.)
"""
from __future__ import annotations

import os
import sys
import traceback

from src.agent.runner import build_agent
from src.config import Config

# ── 20 케이스: (id, 난이도, 질문, 의도/체크) ─────────────────────────────
# expect_tools: 이 도구들이 trajectory에 나와야 함(부분집합)
# any_kw: 답에 이 중 하나라도 포함(정답성)
# cite: 답에 인쇄 페이지 라벨(E2-/F1-) 포함해야
# graceful: 문서에 없는 정보 → 날조 금지(불확실/없음 신호)
# HW 설계/펌웨어 엔지니어가 ARM ARM을 보며 실제로 던질 법한 시나리오.
CASES = [
    ("C01", "스핀락설계", "I'm implementing a spinlock with LDREX/STREX. Read the exclusive-access state diagram and tell me the two monitor states and which instruction enters the exclusive state.",
     {"expect_tools": {"image_read"}, "any_kw": ["Open Access", "Exclusive Access", "LoadExcl", "Load-Exclusive"]}),
    ("C02", "모니터클리어", "For my context-switch code I must know when the exclusive monitor is cleared. What clears it (e.g. CLREX) and what state does it go to?",
     {"any_kw": ["CLREX", "clear", "Open Access", "monitor"]}),
    ("C03", "메모리컨트롤러", "I'm designing a memory controller. From the memory attributes table, what does the Gathering (G/nG) attribute require me to do?",
     {"any_kw": ["gather", "G or nG", "merge", "combine", "access"]}),
    ("C04", "인터커넥트", "For my interconnect ordering model, explain the Reordering (R/nR) attribute as given in the attributes table.",
     {"any_kw": ["reorder", "R or nR", "order"]}),
    ("C05", "쓰기버퍼", "My write buffer design needs the Early Write Acknowledgement hint semantics. What does (No) Early Write Acknowledgement mean and on which page is it? Give the page label.",
     {"expect_tools": {"page_index_search", "keyword_search"}, "cite": True, "any_kw": ["Early Write", "acknowledge"]}),
    ("C06", "로컬vs글로벌", "I need to implement both a local and a global exclusive monitor in my cache. Compare their behaviour for exclusive access.",
     {"any_kw": ["local monitor", "global monitor"]}),
    ("C07", "저전력", "For a low-power spin-wait loop I want to use WFE/SEV. Which printed page covers Wait For Event / Send Event, and what is the event mechanism?",
     {"cite": True, "any_kw": ["Event", "WFE", "SEV", "WFE"]}),
    ("C08", "검증엔지니어", "As a verification engineer I need the exclusive-access state machine to build a checker. Read the figure on page E2-2801 and enumerate the transitions (which ops on which edges).",
     {"expect_tools": {"image_read"}, "any_kw": ["LoadExcl", "StoreExcl", "Exclusive", "Open"]}),
    ("C09", "설계리뷰표지", "For a design review I must cite sources. Which page label describes Store-Exclusive and the global monitor?",
     {"expect_tools": {"page_index_search"}, "cite": True, "any_kw": ["E2-28"]}),
    ("C10", "그림인벤토리", "I'm cataloguing the diagrams I need to study. List the figures in this document and the page each is on.",
     {"expect_tools": {"page_index_search"}, "any_kw": ["E2-2797", "E2-2801", "E2-2804", "page"]}),
    ("C11", "이미지+표통합", "For a coherency feature: using BOTH the exclusive-access diagram (read the figure) and the memory attributes table, relate the Exclusive Access state to the Gathering attribute.",
     {"expect_tools": {"image_read"}, "any_kw": ["Exclusive Access", "Gather"]}),
    ("C12", "페이지정밀", "I want to bookmark the exact printed page for the exclusive-access state diagram I'll reference in my spec. What is its page label?",
     {"any_kw": ["E2-2797", "E2-2801", "E2-2804"]}),
    ("C13", "전이명령", "In the state-machine figure, which single instruction causes the transition from Open Access to Exclusive Access?",
     {"expect_tools": {"image_read"}, "any_kw": ["LoadExcl", "Load-Exclusive", "Load Exclusive"]}),
    ("C14", "STREX실패", "When does a Store-Exclusive (STREX) fail / not update memory, per the monitor behaviour described here?",
     {"any_kw": ["StoreExcl", "fail", "monitor", "marked"]}),
    ("C15", "디바이스메모리", "For an MMIO/device region, what does the Device memory attribute section say I must consider? Cite a page label if possible.",
     {"any_kw": ["Device", "memory", "E2-"]}),
    ("C16", "표참조추적", "The Gathering table row points me elsewhere — which page does it tell me to see for more on Gathering?",
     {"any_kw": ["E2-2788", "page", "2788"]}),
    ("C17", "범위밖레지스터", "I need the SCTLR_EL1 reset value for my boot code. Does THIS document give it? Answer only from this document.",
     {"graceful": True}),
    ("C18", "없는그림", "For my coherency block I'm looking for Figure D9-99 on snoop filters — which page is it on?",
     {"graceful": True}),
    ("C19", "신입온보딩", "Onboarding a junior engineer: give a grounded overview of the AArch32 synchronization & semaphores material in this doc, citing at least one page label.",
     {"cite": True, "any_kw": ["synchron", "exclusive", "semaphore", "monitor"]}),
    ("C20", "마킹규칙", "For my exclusive monitor RTL, how does a LoadExcl mark the address (what granularity / 'marked address') per this document?",
     {"any_kw": ["mark", "address", "LoadExcl", "exclusive"]}),
]


def run_case(agent, q):
    res = agent.run(q)
    traj = []
    for t in res.get("trajectory", []):
        traj.append({
            "loop": t["loop"], "tool": t["tool_name"], "args": t.get("arguments"),
            "hits": t.get("hits"), "images": t.get("images"),
            "cached": t.get("cached"), "snippet": str(t.get("tool_result", ""))[:90],
        })
    return res.get("answer", ""), traj, res.get("loops")


def check(expect, answer, traj):
    ans = (answer or "").lower()
    used = {t["tool"] for t in traj}
    out = {}
    if "expect_tools" in expect:
        miss = expect["expect_tools"] - used
        out["expected tools"] = (not miss, f"used={sorted(used)} miss={sorted(miss)}")
    if "any_kw" in expect:
        hit = [k for k in expect["any_kw"] if k.lower() in ans]
        out["answer keyword"] = (bool(hit), f"matched={hit}")
    if expect.get("cite"):
        import re
        labels = re.findall(r"\b[A-Z]\d+-\d{3,}\b", answer or "")
        out["cites page label"] = (bool(labels), f"labels={labels[:4]}")
    if expect.get("graceful"):
        signal = any(s in ans for s in ["not", "no ", "does not", "cannot", "n/a",
                                        "no information", "not found", "not in", "unable", "없"])
        # 부정: 날조 방지 — 불확실 신호가 있거나 빈약하면 통과(구체 날조면 실패는 사람 판단)
        out["graceful (no fabrication)"] = (signal, f"uncertainty_signal={signal}")
    return out


def main():
    cfg = Config.load(os.environ.get("RAG_CONFIG", "config.yaml"))
    cfg._d.setdefault("agent", {})["max_loops"] = 8  # 평가용 상한(속도)
    agent = build_agent(cfg)

    lines = ["# EVAL — page_index + agentic RAG (어려운 20 케이스, 라이브)\n",
             "> 실제 Upstage·GPT-4.1 mini, 빌드된 ARM v8-A part1 인덱스. 단계별 trajectory와 의도 충족 기록.\n"]
    n_pass = n_total = 0
    summary = []
    for cid, diff, q, expect in CASES:
        try:
            answer, traj, loops = run_case(agent, q)
            checks = check(expect, answer, traj)
        except Exception:
            lines += [f"\n## {cid} [{diff}] — ⛔ ERROR\n", f"Q: {q}\n", f"```\n{traceback.format_exc()[:500]}\n```\n"]
            summary.append((cid, diff, "ERROR"))
            continue
        passed = all(ok for ok, _ in checks.values()) if checks else True
        n_total += 1; n_pass += int(passed)
        verdict = "✅ PASS" if passed else "⚠️ CHECK"
        summary.append((cid, diff, verdict))
        lines.append(f"\n## {cid} [{diff}] — {verdict}  (loops={loops})\n")
        lines.append(f"**Q**: {q}\n")
        lines.append("\n**단계별 trajectory (의도대로 도구를 골랐나):**\n")
        for s in traj:
            extra = []
            if s["hits"] is not None: extra.append(f"hits={s['hits']}")
            if s["images"]: extra.append(f"images={[os.path.basename(i) for i in s['images']]}")
            if s["cached"]: extra.append("cached")
            lines.append(f"- loop{s['loop']}: `{s['tool']}` {s['args']} {' '.join(extra)}")
        lines.append("\n**의도 충족 체크:**\n")
        for name, (ok, note) in checks.items():
            lines.append(f"- {'✅' if ok else '❌'} {name} — {note}")
        lines.append(f"\n**답변(발췌)**: {(answer or '')[:300]}\n")
        print(f"{cid} {diff}: {verdict} (loops={loops})", flush=True)

    head = (f"\n**요약: {n_pass}/{n_total} 케이스 의도 충족.**  "
            + " · ".join(f"{c}:{v.split()[0]}" for c, d, v in summary) + "\n")
    lines.insert(2, head)
    out_dir = os.path.join(cfg.base_dir, "examples")
    os.makedirs(out_dir, exist_ok=True)
    report = os.path.join(out_dir, "EVAL_pageindex_agent.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n=== {n_pass}/{n_total} pass. report → {report} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
