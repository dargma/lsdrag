"""Page Index — 구조 메타데이터 질의 인덱스 (04). 2↔3 계약의 절반.

그래프 노드가 아니라 질의 가능한 메타데이터 인덱스. 그래프 없이 구조 질의를 커버한다.
조회 키: page / figure_no / heading / doc_id (부분 지정 허용).
각 엔트리는 chunk_id + image_path를 보유 → page_index_search가 본문+이미지명을 한 번에 반환.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from src.schema import ParsedDoc


@dataclass
class PageEntry:
    doc_id: str
    chunk_id: str
    page_no: int
    block_type: str
    text: str
    heading: Optional[str] = None
    figure_no: Optional[str] = None
    image_path: Optional[str] = None


class PageIndex:
    """문서별 구조 엔트리. add/remove는 doc_id 단위(증분 — 05와 정합)."""

    def __init__(self) -> None:
        self._entries: List[PageEntry] = []

    # ── 빌드/증분 ──────────────────────────────────────────────
    def add_doc(self, doc: ParsedDoc) -> None:
        self.remove_doc(doc.doc_id)  # 멱등: 재추가는 갱신
        for b in doc.blocks:
            self._entries.append(PageEntry(
                doc_id=doc.doc_id,
                chunk_id=b.chunk_id or f"{doc.doc_id}:{len(self._entries)}",
                page_no=b.page_no, block_type=b.block_type, text=b.text,
                heading=b.heading, figure_no=b.figure_no, image_path=b.image_path,
            ))

    def remove_doc(self, doc_id: str) -> int:
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.doc_id != doc_id]
        return before - len(self._entries)

    def doc_ids(self) -> List[str]:
        return sorted({e.doc_id for e in self._entries})

    # ── 조회 (없으면 빈 결과, 예외 아님) ─────────────────────────
    def query(self, page: Optional[int] = None, figure_no: Optional[str] = None,
              heading: Optional[str] = None, doc_id: Optional[str] = None,
              block_type: Optional[str] = None) -> List[PageEntry]:
        res = self._entries
        if doc_id is not None:
            res = [e for e in res if e.doc_id == doc_id]
        if page is not None:
            res = [e for e in res if e.page_no == page]
        if figure_no is not None:
            res = [e for e in res if e.figure_no == str(figure_no)]
        if block_type is not None:
            res = [e for e in res if e.block_type == block_type]
        if heading is not None:
            h = heading.lower()
            res = [e for e in res if e.heading and h in e.heading.lower()]
        return list(res)

    # ── 영속화 ─────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "page_index.json"), "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in self._entries], f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "PageIndex":
        fp = os.path.join(path, "page_index.json")
        if not os.path.exists(fp):
            raise FileNotFoundError(
                f"Page Index가 없습니다: {fp}. 먼저 빌드하세요(python -m src.indexing.build)."
            )
        idx = cls()
        with open(fp, encoding="utf-8") as f:
            idx._entries = [PageEntry(**d) for d in json.load(f)]
        return idx

    def __len__(self) -> int:
        return len(self._entries)
