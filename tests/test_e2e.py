"""08_e2e 게이트 (오프라인) — 전체 체인(IR→store→registry→agent→답변) + 채점 + 모듈책임 분류.

실제 LLM/Reader 없이 스크립트형 FakeLLM. 3유형(text/structure/image) 각 1건.
"""
import json
import os
import tempfile

import src.agent.runner  # noqa: F401  (컨텍스트 패치)
from arag.agent.base import BaseAgent
from arag.tools.base import BaseTool

from src.indexing import IndexStore
from src.page_index import PageIndex
from src.retrieval import PageIndexSearchTool, ImageReadTool, build_registry
from src.schema import ParsedBlock, ParsedDoc
from tests.e2e.golden import GOLDEN_SET, grade, summarize
from tests.test_indexing import fake_embed


def _build_store(img):
    store = IndexStore()
    store.add_doc(ParsedDoc(doc_id="arm1", title="ARM", blocks=[
        ParsedBlock(text="The SCTLR_EL1 register controls the system behavior.", page_no=13,
                    block_type="text", chunk_id="arm1:0"),
        ParsedBlock(text="bitfield layout", page_no=13, block_type="figure", figure_no="2",
                    image_path=img, chunk_id="arm1:1"),
    ]), fake_embed)
    return store


class _ChunkStub(BaseTool):
    def __init__(self, nm, chunks): self._n = nm; self._c = chunks
    @property
    def name(self): return self._n
    def get_schema(self): return {"type": "function", "function": {"name": self._n, "parameters": {}}}
    def execute(self, context, **k):
        context.add_retrieval_log(self._n, 1, {})
        return " ".join(self._c.values()), {"tool": self._n}


def _registry(store, img):
    return build_registry([
        _ChunkStub("semantic_search", store.chunks),
        _ChunkStub("keyword_search", store.chunks),
        _ChunkStub("read_chunk", store.chunks),
        PageIndexSearchTool(store.page_index, store.chunks),
        ImageReadTool(lambda p: "The bitfield diagram shows fields 0..63."),
    ])


class FakeLLM:
    def __init__(self, script): self.s = list(script); self.i = 0
    def chat(self, messages, tools=None):
        m = self.s[min(self.i, len(self.s) - 1)]; self.i += 1
        return {"message": m, "cost": 0.0}


def _tc(name, args):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": "x", "function": {"name": name, "arguments": json.dumps(args)}}]}


# 각 골든을 정상적으로 푸는 스크립트(기대 tool 호출 → 키워드 포함 답변)
SCRIPTS = {
    "t1": ([_tc("semantic_search", {"query": "SCTLR_EL1"}),
            {"role": "assistant", "content": "SCTLR_EL1 controls the system behavior."}]),
    "s1": ([_tc("page_index_search", {"page": 13, "figure_no": "2"}),
            {"role": "assistant", "content": "It is on page 13, figure 2."}]),
    "i1": ([_tc("image_read", {"image_path": "__IMG__"}),
            {"role": "assistant", "content": "The bitfield diagram shows the fields."}]),
}


def _run_one(g, store, img):
    script = [dict(m) for m in SCRIPTS[g.qid]]
    for m in script:
        for tc in m.get("tool_calls", []):
            tc["function"]["arguments"] = tc["function"]["arguments"].replace("__IMG__", img)
    agent = BaseAgent(llm_client=FakeLLM(script), tools=_registry(store, img),
                      system_prompt="t", max_loops=5)
    res = agent.run(g.question)
    tools_used = [t["tool_name"] for t in res["trajectory"]]
    return grade(g, res["answer"], tools_used)


def test_golden_set_passes_threshold():
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "f.png"); open(img, "wb").write(b"PNG")
        store = _build_store(img)
        scores = [_run_one(g, store, img) for g in GOLDEN_SET]
        s = summarize(scores)
        assert s["match_rate"] >= 0.99, s
        # 3유형 각각 최소 1건 통과
        types = {g.qtype for g, sc in zip(GOLDEN_SET, scores) if sc.passed}
        assert types == {"text", "structure", "image"}


def test_failure_blames_correct_module():
    from tests.e2e.golden import Golden
    # 구조 질의인데 page_index_search를 안 거치면 → page_index/retrieval 책임
    g = Golden("bad", "structure", "q", ["page 99"], "page_index_search")
    sc = grade(g, "some answer without the page", tools_used=["semantic_search"])
    assert not sc.passed and sc.blame == "page_index/retrieval"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("[08_e2e] 골든셋 채점 + 모듈책임 분류 통과")
