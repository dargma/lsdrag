"""page_index_search — 신규 tool (06). 구조 질의 → 위치 특정 + 본문 + 이미지명.

사실 3: 이 tool은 예외적으로 위치 특정 + 본문 + 이미지명을 함께 반환 → read_chunk 우회.
BaseTool 계약(사실 1) 충족. 코어 수정 없음(사실 2: register만).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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
                    "Returns the located chunk's FULL text. If any figure image sits on the "
                    "matched page(s), its filename is listed — then call image_read on it to read the figure. "
                    "Best for 'which page/figure is X on' and for pulling text + figure together."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "string",
                                 "description": "printed page label (e.g. 'E2-2797') or the local page number"},
                        "figure_no": {"type": "string", "description": "figure label if the document numbers it"},
                        "heading": {"type": "string", "description": "section heading substring"},
                        "doc_id": {"type": "string"},
                    },
                },
            },
        }

    # 토큰 효율: 한 호출이 반환하는 엔트리 수·본문 길이 상한(전체 페이지 덤프 방지).
    MAX_ENTRIES = 6
    MAX_BODY = 500

    def execute(self, context: "Any", page=None, figure_no: Optional[str] = None,
                heading: Optional[str] = None, doc_id: Optional[str] = None,
                top_k: int = MAX_ENTRIES, **_: Any) -> Tuple[str, Dict[str, Any]]:
        all_hits = self._pi.query(page=page, figure_no=figure_no, heading=heading, doc_id=doc_id)
        if not all_hits:
            log = {"tool": self.name, "hits": 0}
            _safe_log(context, self.name, 0, log)
            return (f"No structural match (page={page}, figure_no={figure_no}, "
                    f"heading={heading}, doc_id={doc_id})."), log

        cap = max(1, min(int(top_k or self.MAX_ENTRIES), 20))
        hits = all_hits[:cap]
        parts, images = [], []
        for e in hits:
            full = self._chunks.get(e.chunk_id, e.text)
            if len(full) > self.MAX_BODY:
                full = full[:self.MAX_BODY] + " …[truncated; use read_chunk for full]"
            loc = f"page {e.page_label}" if e.page_label else f"p{e.page_no}"
            head = f"[{e.doc_id} {loc}" + (f" fig {e.figure_no}" if e.figure_no else "") + "]"
            parts.append(f"{head}\n{full}")
            if e.image_path:
                images.append(e.image_path)
            if hasattr(context, "mark_chunk_as_read"):
                context.mark_chunk_as_read(e.chunk_id)
        if len(all_hits) > cap:
            parts.append(f"(+{len(all_hits) - cap} more matches on this query; refine page/heading to narrow.)")

        # 같은 페이지에 있는 figure 이미지를 자동 노출 → 에이전트가 image_read로 읽을 수 있게.
        # (figure가 번호 없거나 헤딩이 달라 직접 매칭 안 돼도, 텍스트가 걸린 페이지의 그림을 surface)
        hit_pages = {(e.doc_id, e.page_no) for e in all_hits}
        for f in self._pi.query(block_type="figure"):
            if (f.doc_id, f.page_no) in hit_pages and f.image_path and f.image_path not in images:
                images.append(f.image_path)
        images = images[:4]  # 토큰 효율: 노출 이미지 수 제한

        if images:
            parts.append("Figure image(s) on these pages — call image_read to read them: "
                         + ", ".join(images))
        log = {"tool": self.name, "hits": len(all_hits), "images": images}
        _safe_log(context, self.name, sum(len(p) for p in parts), log)
        return "\n\n".join(parts), log


def _safe_log(context, name, tokens, meta):
    if hasattr(context, "add_retrieval_log"):
        try:
            context.add_retrieval_log(name, tokens, meta)
        except Exception:
            pass
