"""교체 경계: Upstage 응답 → IR (03_parser).

사내 파서로 교체할 때 **이 파일만** 바뀐다. 다운스트림은 IR(02)만 본다.
순수 함수로 유지(HTTP 없음) → 모킹으로 회귀 고정 가능.
"""
from __future__ import annotations

import base64
import os
from typing import Any, Callable, Dict, List, Optional

from src.schema import ParsedBlock, ParsedDoc, IRValidationError

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
            figure_no=str(el.get("id")) if bt == "figure" else None,
            image_path=image_path,
            bbox=_coords_to_bbox(el.get("coordinates")),
            chunk_id=chunk_id,
        ))

    doc = ParsedDoc(doc_id=doc_id, title=title, blocks=blocks, date=date)
    doc.validate()
    return doc


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
