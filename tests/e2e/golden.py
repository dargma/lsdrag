"""골든 QA 셋 + 채점 + 모듈책임 분류 (08).

3유형: text(개념) / structure(page·figure) / image(다이어그램 해석).
채점은 정답 키워드 매칭(정확/부분). 실패는 어느 모듈 책임인지 분류 가능하게.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Golden:
    qid: str
    qtype: str            # "text" | "structure" | "image"
    question: str
    must_include: List[str]    # 답에 포함돼야 할 키워드(부분 매칭)
    expects_tool: str          # 이 질의가 반드시 거쳐야 할 tool


GOLDEN_SET: List[Golden] = [
    Golden("t1", "text", "What does the SCTLR_EL1 register control?",
           ["system"], "semantic_search"),
    Golden("s1", "structure", "Which page and figure shows the SCTLR_EL1 bitfields?",
           ["page 13", "figure 2"], "page_index_search"),
    Golden("i1", "image", "Interpret the SCTLR_EL1 bitfield diagram.",
           ["bitfield"], "image_read"),
]


@dataclass
class Score:
    qid: str
    passed: bool
    matched: int
    total: int
    blame: str = ""           # 실패 시 책임 모듈
    note: str = ""


def grade(g: Golden, answer: str, tools_used: List[str]) -> Score:
    ans = (answer or "").lower()
    matched = sum(1 for k in g.must_include if k.lower() in ans)
    total = len(g.must_include)
    passed = matched == total
    blame = note = ""
    if not passed:
        # 모듈책임 분류: 기대 tool이 안 불렸나? 불렸는데 답이 틀렸나?
        if g.expects_tool not in tools_used:
            blame = _tool_owner(g.expects_tool)
            note = f"expected tool '{g.expects_tool}' not used"
        else:
            blame = "agent/reader"
            note = "tool used but answer incomplete"
    return Score(g.qid, passed, matched, total, blame, note)


def _tool_owner(tool: str) -> str:
    return {
        "keyword_search": "indexing/retrieval",
        "semantic_search": "indexing/retrieval",
        "read_chunk": "indexing/retrieval",
        "page_index_search": "page_index/retrieval",
        "image_read": "parser(image)/retrieval",
    }.get(tool, "agent")


def summarize(scores: List[Score]) -> dict:
    by_type_pass = sum(1 for s in scores if s.passed)
    return {
        "total": len(scores),
        "passed": by_type_pass,
        "match_rate": round(by_type_pass / len(scores), 3) if scores else 0.0,
        "failures": [{"qid": s.qid, "blame": s.blame, "note": s.note} for s in scores if not s.passed],
    }
