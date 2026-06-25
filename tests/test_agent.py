"""07_agent 게이트 — 스크립트형 FakeLLM으로 loop 완주 + tool 연결 + 이미지 추적(컨텍스트 주입).

실제 LLM/Reader 없이. src.agent.runner import가 vendor base의 AgentContext를
ImageAgentContext로 교체했음을 image_read 중복캐시로 검증.
"""
import json
import os
import tempfile

import arag.agent.base as agent_base
from arag.agent.base import BaseAgent

import src.agent.runner  # noqa: F401  # side effect: vendor AgentContext를 ImageAgentContext로 패치
from src.agent.context import ImageAgentContext
from src.page_index import PageIndex
from src.retrieval import ImageReadTool, PageIndexSearchTool, build_registry
from src.schema import ParsedBlock, ParsedDoc


def test_context_is_patched():
    assert agent_base.AgentContext is ImageAgentContext


class FakeLLM:
    """스크립트된 tool_call 시퀀스를 차례로 반환."""
    def __init__(self, script):
        self.script = list(script); self.i = 0
    def chat(self, messages, tools=None):
        msg = self.script[min(self.i, len(self.script) - 1)]; self.i += 1
        return {"message": msg, "cost": 0.0}


def _tool_call(name, args):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": f"c{name}", "function": {"name": name, "arguments": json.dumps(args)}}]}


def _final(text):
    return {"role": "assistant", "content": text}


def _registry(img_path, reader_calls):
    pi = PageIndex()
    pi.add_doc(ParsedDoc(doc_id="d1", title="t", blocks=[
        ParsedBlock(text="SCTLR_EL1 controls system.", page_no=13, chunk_id="d1:0"),
        ParsedBlock(text="bitfield", page_no=13, block_type="figure", figure_no="2",
                    image_path=img_path, chunk_id="d1:1"),
    ]))
    def reader(p):
        reader_calls.append(p); return "Figure shows bitfields 0..63."
    return build_registry([
        PageIndexSearchTool(pi, {"d1:0": "SCTLR_EL1 controls system.", "d1:1": "bitfield"}),
        ImageReadTool(reader),
    ])


def test_loop_completes_tools_fire_image_dedup():
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "f.png"); open(img, "wb").write(b"PNG")
        reader_calls = []
        reg = _registry(img, reader_calls)
        llm = FakeLLM([
            _tool_call("page_index_search", {"page": 13, "figure_no": "2"}),
            _tool_call("image_read", {"image_path": img}),
            _tool_call("image_read", {"image_path": img}),  # 중복 → 캐시
            _final("SCTLR_EL1 bitfields are on page 13, figure 2."),
        ])
        agent = BaseAgent(llm_client=llm, tools=reg, system_prompt="t", max_loops=6)
        result = agent.run("Where are SCTLR_EL1 bitfields?")

        assert "page 13" in result["answer"]
        names = [t["tool_name"] for t in result["trajectory"]]
        assert "page_index_search" in names and names.count("image_read") == 2
        # 핵심: 두번째 image_read는 캐시 → reader 1회만 = ImageAgentContext 주입 증거
        assert len(reader_calls) == 1
        img_results = [t["tool_result"] for t in result["trajectory"] if t["tool_name"] == "image_read"]
        assert any("already interpreted" in r for r in img_results)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("[07_agent] loop 완주 + tool 연결 + 이미지 추적 통과")
