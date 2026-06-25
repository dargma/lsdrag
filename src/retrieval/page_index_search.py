"""page_index_search — 신규 tool (06). 구조 질의 → 위치 특정 + 본문 + 이미지명.

사실 3: 이 tool은 예외적으로 위치 특정 + 본문 + 이미지명을 함께 반환 → read_chunk 우회.
BaseTool 계약(사실 1) 충족. 코어 수정 없음(사실 2: register만).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import src.agent  # noqa: F401  (vendor path 부트스트랩)
from arag.tools.base import BaseTool

from src.page_index import PageIndex


class PageIndexSearchTool(BaseTool):
    def __init__(self, page_index: PageIndex, chunks: Dict[str, str]) -> None:
        self._pi = page_index
        self._chunks = chunks  # chunk_id -> full text

    @property
    def name(self) -> str:
        return "page_index_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "page_index_search",
                "description": (
                    "Structural lookup by page / figure number / heading / doc_id. "
                    "Returns the located chunk's FULL text plus any image filename — "
                    "no need to call read_chunk afterward. Use for 'which page/figure is X on'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer"},
                        "figure_no": {"type": "string"},
                        "heading": {"type": "string"},
                        "doc_id": {"type": "string"},
                    },
                },
            },
        }

    def execute(self, context: "Any", page: Optional[int] = None, figure_no: Optional[str] = None,
                heading: Optional[str] = None, doc_id: Optional[str] = None,
                **_: Any) -> Tuple[str, Dict[str, Any]]:
        hits = self._pi.query(page=page, figure_no=figure_no, heading=heading, doc_id=doc_id)
        if not hits:
            log = {"tool": self.name, "hits": 0}
            _safe_log(context, self.name, 0, log)
            return (f"No structural match (page={page}, figure_no={figure_no}, "
                    f"heading={heading}, doc_id={doc_id})."), log

        parts, images = [], []
        for e in hits:
            full = self._chunks.get(e.chunk_id, e.text)
            head = f"[{e.doc_id} p{e.page_no}" + (f" fig {e.figure_no}" if e.figure_no else "") + "]"
            parts.append(f"{head}\n{full}")
            if e.image_path:
                images.append(e.image_path)
            if hasattr(context, "mark_chunk_as_read"):
                context.mark_chunk_as_read(e.chunk_id)
        if images:
            parts.append("Images available (use image_read if needed): " + ", ".join(images))
        log = {"tool": self.name, "hits": len(hits), "images": images}
        _safe_log(context, self.name, sum(len(p) for p in parts), log)
        return "\n\n".join(parts), log


def _safe_log(context, name, tokens, meta):
    if hasattr(context, "add_retrieval_log"):
        try:
            context.add_retrieval_log(name, tokens, meta)
        except Exception:
            pass
