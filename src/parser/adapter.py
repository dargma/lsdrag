"""교체 경계: Upstage 응답 → IR (03_parser).

사내 파서로 교체할 때 **이 파일만** 바뀐다. 다운스트림은 IR(02)만 본다.
순수 함수로 유지(HTTP 없음) → 모킹으로 회귀 고정 가능.
"""
from __future__ import annotations

import base64
import os
import re
from typing import Any, Callable, Dict, List, Optional

from src.schema import ParsedBlock, ParsedDoc, IRValidationError

# 문서의 실제 figure 라벨(예: "Figure D1-3", "Figure E2-1", "Figure 2.5") 추출용.
# Upstage element id가 아니라 캡션에 적힌 진짜 번호를 쓴다.
_FIGURE_LABEL = re.compile(r"Figure\s+([A-Z]?\d[\w.\-]*)", re.I)


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s or "").strip()


def _figure_label_in(text: str) -> Optional[str]:
    m = _FIGURE_LABEL.search(_strip_html(text or ""))
    return m.group(1).rstrip(".-") if m else None  # 끝의 문장부호 제거('2.'→'2', '2.5' 유지)

# Upstage category → IR block_type (사실 8)
_CATEGORY_BLOCK_TYPE = {
    "figure": "figure", "chart": "figure",
    "table": "table",
    "caption": "caption",
}
_HEADING_CATEGORIES = {"heading1", "heading2", "heading3", "header"}


def _block_type(category: str) -> str:
    return _CATEGORY_BLOCK_TYPE.get(category, "text")


def _text_of(content: Any) -> str:
    """content{html,markdown,text} 중 text 우선, 없으면 markdown/html."""
    if isinstance(content, dict):
        return content.get("text") or content.get("markdown") or content.get("html") or ""
    return str(content or "")


def upstage_to_ir(
    response: Dict[str, Any],
    doc_id: str,
    title: str,
    images_dir: Optional[str] = None,
    date: Optional[str] = None,
    save_image: Optional[Callable[[str, bytes], str]] = None,
) -> ParsedDoc:
    """Upstage Document Parse 응답 → ParsedDoc.

    - figure/chart 요소의 base64 이미지는 images_dir에 저장하고 image_path를 채운다.
    - save_image(filename, data)->path 를 주면 그것으로 저장(테스트 주입용). 없으면 파일시스템.
    """
    elements = response.get("elements")
    if not isinstance(elements, list):
        raise IRValidationError("Upstage response missing 'elements' list")

    blocks: List[ParsedBlock] = []
    current_heading: Optional[str] = None

    for el in elements:
        category = el.get("category", "paragraph")
        page = int(el.get("page", 1) or 1)
        text = _text_of(el.get("content"))
        bt = _block_type(category)

        if category in _HEADING_CATEGORIES:
            current_heading = text or current_heading

        image_path = None
        if bt == "figure":
            image_path = _maybe_save_image(el, doc_id, page, images_dir, save_image)

        chunk_id = f"{doc_id}:{el.get('id', len(blocks))}"
        blocks.append(ParsedBlock(
            text=text,
            page_no=page,
            block_type=bt,
            heading=current_heading,
            figure_no=None,  # 실제 문서 라벨은 아래 post-pass에서 캡션으로 채운다
            image_path=image_path,
            bbox=_coords_to_bbox(el.get("coordinates")),
            chunk_id=chunk_id,
        ))

    _assign_figure_labels(blocks)
    doc = ParsedDoc(doc_id=doc_id, title=title, blocks=blocks, date=date)
    doc.validate()
    return doc


def _assign_figure_labels(blocks: List[ParsedBlock]) -> None:
    """figure 블록에 '문서의 실제 figure 번호'를 부여(Upstage element id 사용 금지).

    근거: 같은 페이지의 caption/인접 블록 텍스트에 적힌 'Figure X'. 없으면 None으로 둔다
    (번호 없는 inline 그림 — page_index_search는 page로 조회 가능).
    """
    for i, b in enumerate(blocks):
        if b.block_type != "figure":
            continue
        label = _figure_label_in(b.text)
        if not label:
            # 같은 페이지의 caption 우선, 그다음 인접 블록을 좁게 탐색
            window = [c for c in blocks[max(0, i - 2):i + 3]
                      if c.page_no == b.page_no and c is not b]
            window.sort(key=lambda c: 0 if c.block_type == "caption" else 1)
            for c in window:
                label = _figure_label_in(c.text)
                if label:
                    break
        b.figure_no = label  # 진짜 라벨 또는 None


def _coords_to_bbox(coords: Any) -> Optional[List[float]]:
    """Upstage coordinates(상대좌표 점 리스트) → [x0,y0,x1,y1] 근사."""
    if not coords:
        return None
    try:
        xs = [float(p["x"]) for p in coords]
        ys = [float(p["y"]) for p in coords]
        return [min(xs), min(ys), max(xs), max(ys)]
    except (KeyError, TypeError, ValueError):
        return None


def _maybe_save_image(el, doc_id, page, images_dir, save_image) -> Optional[str]:
    b64 = el.get("base64_encoding") or el.get("base64")
    if not b64:
        return None  # figure인데 이미지 없음 → 상위에서 검증 시 걸린다
    fname = f"{doc_id}_p{page}_e{el.get('id', 'x')}.png"
    try:
        data = base64.b64decode(b64)
    except Exception:
        return None
    if save_image is not None:
        return save_image(fname, data)
    if not images_dir:
        return None
    os.makedirs(images_dir, exist_ok=True)
    path = os.path.join(images_dir, fname)
    with open(path, "wb") as f:
        f.write(data)
    return path
