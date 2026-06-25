"""IR (Intermediate Representation) — 파싱 단계와 DB 구축 단계 사이의 단일 계약(02).

파서가 무엇이든(Upstage/사내) 출력은 이 IR로 흡수된다. 다운스트림은 IR만 본다.
JSON 직렬화/역직렬화는 무손실이어야 한다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# block_type 최소 집합 (확장 가능)
BLOCK_TYPES = {"text", "table", "figure", "caption"}


class IRValidationError(ValueError):
    """IR 계약 위반. 침묵 통과 금지 — 명확한 에러로 거부한다."""


@dataclass
class ParsedBlock:
    """문서의 한 블록. 구조 필드(page_no/figure_no/image_path)는 Page Index(04)가 의존."""
    text: str
    page_no: int                       # 파서 기준(분할 doc 로컬) 페이지 번호
    block_type: str = "text"
    heading: Optional[str] = None
    figure_no: Optional[str] = None
    image_path: Optional[str] = None
    bbox: Optional[List[float]] = None
    chunk_id: Optional[str] = None     # 인덱스/PageIndex가 참조하는 안정 ID
    page_label: Optional[str] = None   # 문서에 인쇄된 실제 페이지 표기(예: "E2-2804"). 없으면 None

    def validate(self) -> None:
        if not isinstance(self.text, str):
            raise IRValidationError(f"block.text must be str, got {type(self.text).__name__}")
        if not isinstance(self.page_no, int):
            raise IRValidationError(f"block.page_no must be int, got {type(self.page_no).__name__}")
        if self.block_type not in BLOCK_TYPES:
            raise IRValidationError(
                f"block.block_type '{self.block_type}' not in {sorted(BLOCK_TYPES)}"
            )
        # figure는 image_path가 있어야 image_read가 동작하지만, 모든 파서가 이미지를 주진 않는다.
        # → 블록 단위 하드 요구가 아니라 03_parser 게이트에서 'figure 중 image_path 비율'로 검증.


@dataclass
class ParsedDoc:
    """파싱된 한 문서. 분할 PDF 각 조각이 별도 doc_id."""
    doc_id: str
    title: str
    blocks: List[ParsedBlock] = field(default_factory=list)
    date: Optional[str] = None

    def validate(self) -> None:
        if not self.doc_id:
            raise IRValidationError("doc.doc_id is required and non-empty")
        if not isinstance(self.title, str):
            raise IRValidationError("doc.title must be str")
        if not isinstance(self.blocks, list):
            raise IRValidationError("doc.blocks must be a list")
        for i, b in enumerate(self.blocks):
            if not isinstance(b, ParsedBlock):
                raise IRValidationError(f"doc.blocks[{i}] must be ParsedBlock")
            b.validate()

    # ── 직렬화 (무손실 왕복) ──────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParsedDoc":
        if not isinstance(d, dict):
            raise IRValidationError("from_dict expects a dict")
        missing = {"doc_id", "title"} - set(d)
        if missing:
            raise IRValidationError(f"IR dict missing required keys: {sorted(missing)}")
        blocks_raw = d.get("blocks", [])
        if not isinstance(blocks_raw, list):
            raise IRValidationError("blocks must be a list")
        blocks = []
        for i, bd in enumerate(blocks_raw):
            if not isinstance(bd, dict):
                raise IRValidationError(f"blocks[{i}] must be a dict")
            if "text" not in bd or "page_no" not in bd:
                raise IRValidationError(f"blocks[{i}] missing required keys text/page_no")
            blocks.append(ParsedBlock(**bd))
        doc = cls(doc_id=d["doc_id"], title=d["title"], blocks=blocks, date=d.get("date"))
        doc.validate()
        return doc
