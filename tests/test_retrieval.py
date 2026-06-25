"""06_retrieval 게이트 G1~G4 (오프라인). 신규 2종 정밀 + 5종 registry 통합.

vendor 3종은 모델 다운로드를 피하려고 stub BaseTool로 대체(계약 동형 검증 목적).
"""
import os
import tempfile
from typing import Any, Dict, Tuple

import src.agent  # noqa: F401
from arag.tools.base import BaseTool

from src.page_index import PageIndex
from src.retrieval import PageIndexSearchTool, ImageReadTool, build_registry
from src.schema import ParsedBlock, ParsedDoc


class _Ctx:
    """mock AgentContext (이미지 추적 포함 = 07 확장 모사)."""
    def __init__(self):
        self.read_chunk_ids = set(); self.read_image_ids = set(); self.logs = []
    def mark_chunk_as_read(self, c): self.read_chunk_ids.add(c)
    def is_image_read(self, p): return p in self.read_image_ids
    def mark_image_read(self, p): self.read_image_ids.add(p)
    def add_retrieval_log(self, name, tokens, meta): self.logs.append((name, meta))


class _Stub(BaseTool):
    def __init__(self, nm): self._n = nm
    @property
    def name(self): return self._n
    def get_schema(self): return {"type": "function", "function": {"name": self._n, "parameters": {}}}
    def execute(self, context, **k) -> Tuple[str, Dict[str, Any]]:
        if hasattr(context, "add_retrieval_log"): context.add_retrieval_log(self._n, 1, {})
        return f"{self._n} ran", {"tool": self._n}


def _pi_chunks():
    pi = PageIndex()
    pi.add_doc(ParsedDoc(doc_id="d1", title="t", blocks=[
        ParsedBlock(text="SCTLR_EL1 full body text.", page_no=13, block_type="text", chunk_id="d1:0"),
        ParsedBlock(text="bitfield", page_no=13, block_type="figure", figure_no="2",
                    image_path="data/images/x.png", chunk_id="d1:1"),
    ]))
    return pi, {"d1:0": "SCTLR_EL1 full body text.", "d1:1": "bitfield"}


# ── G2: page_index_search ────────────────────────────────────
def test_g2_page_figure_returns_body_and_image():
    pi, chunks = _pi_chunks()
    t = PageIndexSearchTool(pi, chunks); ctx = _Ctx()
    out, log = t.execute(ctx, page=13, figure_no="2")
    assert "data/images/x.png" in out and log["hits"] == 1
    assert isinstance(out, str) and isinstance(log, dict)

def test_g2_missing_location_empty_graceful():
    pi, chunks = _pi_chunks()
    out, log = PageIndexSearchTool(pi, chunks).execute(_Ctx(), page=999)
    assert "No structural match" in out and log["hits"] == 0


# ── G3: image_read dedup ─────────────────────────────────────
def test_g3_image_read_no_double_vlm():
    calls = {"n": 0}
    def reader(p): calls["n"] += 1; return "described"
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "x.png"); open(img, "wb").write(b"PNG")
        t = ImageReadTool(reader); ctx = _Ctx()
        t.execute(ctx, image_path=img)
        t.execute(ctx, image_path=img)  # 두번째는 캐시
        assert calls["n"] == 1

def test_g3_image_missing_file_graceful():
    out, log = ImageReadTool(lambda p: "x").execute(_Ctx(), image_path="/no/such.png")
    assert "not found" in out and "error" in log

def test_g3_vlm_failure_fallback():
    def boom(p): raise RuntimeError("vlm down")
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "x.png"); open(img, "wb").write(b"PNG")
        out, log = ImageReadTool(boom).execute(_Ctx(), image_path=img)
        assert "VLM failed" in out and log["error"] == "vlm_failed"


# ── G1 + G4: registry 5종 ────────────────────────────────────
def _five_tools():
    pi, chunks = _pi_chunks()
    return [_Stub("keyword_search"), _Stub("semantic_search"), _Stub("read_chunk"),
            PageIndexSearchTool(pi, chunks), ImageReadTool(lambda p: "d")]

def test_g1_registry_has_five_schemas():
    reg = build_registry(_five_tools())
    schemas = reg.get_all_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert len(schemas) == 5
    assert names == {"keyword_search", "semantic_search", "read_chunk",
                     "page_index_search", "image_read"}

def test_g4_all_execute_and_log():
    reg = build_registry(_five_tools()); ctx = _Ctx()
    reg.execute("keyword_search", ctx, keywords=["x"])
    reg.execute("semantic_search", ctx, query="x")
    reg.execute("read_chunk", ctx, chunk_id="d1:0")
    reg.execute("page_index_search", ctx, page=13, figure_no="2")
    reg.execute("image_read", ctx, image_path="/no/such.png")
    logged = {n for n, _ in ctx.logs}
    assert {"keyword_search", "semantic_search", "read_chunk", "page_index_search"} <= logged


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("[06_retrieval] G1~G4 + 신규 2종 정밀 통과")
