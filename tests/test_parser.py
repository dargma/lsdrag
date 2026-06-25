"""03_parser 회귀 테스트 — adapter(교체 경계)를 mock Upstage 응답으로 고정.

실제 API 호출 없이 응답 JSON 계약만 검증. 실행: python -m tests.test_parser
"""
import base64

from src.parser import upstage_to_ir
from src.schema import ParsedDoc

# 사실 8의 응답 형식을 모사한 한 벌
MOCK_RESPONSE = {
    "elements": [
        {"id": 0, "category": "heading1", "page": 12,
         "content": {"text": "System Control"}, "coordinates": [{"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.2}]},
        {"id": 1, "category": "paragraph", "page": 12,
         "content": {"text": "The SCTLR_EL1 register controls..."}},
        {"id": 2, "category": "table", "page": 13, "content": {"text": "field | bits"}},
        {"id": 3, "category": "figure", "page": 13, "content": {"text": "bitfield diagram"},
         "base64_encoding": base64.b64encode(b"PNGDATA").decode()},
        {"id": 4, "category": "caption", "page": 13, "content": {"text": "Figure 2. SCTLR_EL1"}},
        {"id": 5, "category": "footer", "page": 13, "content": {"text": "E2-2804"}},
    ]
}


def _convert(saved):
    return upstage_to_ir(
        MOCK_RESPONSE, doc_id="arm_part1", title="ARM Part 1",
        save_image=lambda name, data: saved.setdefault(name, data) or f"data/images/{name}",
    )


def test_categories_mapped():
    doc = _convert({})
    types = [b.block_type for b in doc.blocks]
    assert types == ["text", "text", "table", "figure", "caption", "text"], types


def test_html_only_content_is_stripped_clean():
    # Upstage가 content.html만 채우고 text/markdown 빈 경우 → 태그 제거된 clean text
    resp = {"elements": [
        {"id": 0, "category": "heading1", "page": 1,
         "content": {"html": "<br><h1 id='1' style='font-size:16px'>Exclusive Access</h1>",
                     "markdown": "", "text": ""}},
        {"id": 1, "category": "paragraph", "page": 1,
         "content": {"html": "<p id='2' data-category='paragraph'>A Store-Exclusive performs...</p>",
                     "markdown": "", "text": ""}},
    ]}
    doc = upstage_to_ir(resp, doc_id="d", title="d")
    body = doc.blocks[1]
    assert "<" not in body.text and "Store-Exclusive" in body.text, body.text
    assert body.heading == "Exclusive Access"  # 헤딩도 태그 제거 → 매칭 가능


def test_table_keeps_markdown_structure():
    resp = {"elements": [{"id": 0, "category": "table", "page": 1,
        "content": {"markdown": "| A | B |\n| --- | --- |\n| 1 | 2 |", "html": "<table>..</table>", "text": ""}}]}
    doc = upstage_to_ir(resp, doc_id="d", title="d")
    assert "| A | B |" in doc.blocks[0].text  # 표는 markdown 구조 보존


def test_page_label_from_footer():
    # P3: footer "E2-2804" → page 13 블록들이 인쇄 페이지 표기를 가진다(분할 로컬 page_no와 별개)
    doc = _convert({})
    p13 = [b for b in doc.blocks if b.page_no == 13]
    assert p13 and all(b.page_label == "E2-2804" for b in p13)
    p12 = [b for b in doc.blocks if b.page_no == 12]
    assert all(b.page_label is None for b in p12)  # footer 없는 페이지는 None


def test_heading_propagates():
    doc = _convert({})
    assert doc.blocks[1].heading == "System Control"


def test_figure_image_path_filled():
    saved = {}
    doc = _convert(saved)
    fig = [b for b in doc.blocks if b.block_type == "figure"][0]
    assert fig.image_path and saved, "figure image must be saved + path set"


def test_figure_no_is_real_caption_label_not_element_id():
    # 캡션 "Figure 2. SCTLR_EL1" → figure_no "2" (Upstage element id 3 아님)
    doc = _convert({})
    fig = [b for b in doc.blocks if b.block_type == "figure"][0]
    assert fig.figure_no == "2", f"expected real label '2', got {fig.figure_no!r}"


def test_page_and_bbox():
    doc = _convert({})
    assert doc.blocks[0].page_no == 12
    assert doc.blocks[0].bbox == [0.1, 0.1, 0.9, 0.2]


def test_doc_id_and_roundtrip():
    doc = _convert({})
    assert doc.doc_id == "arm_part1"
    assert ParsedDoc.from_dict(doc.to_dict()).to_dict() == doc.to_dict()


def test_gate_nonnull_ratio():
    # 게이트: page_no·본문 non-null 비율, figure image_path 채워짐
    doc = _convert({})
    assert all(isinstance(b.page_no, int) for b in doc.blocks)
    assert sum(1 for b in doc.blocks if b.text) / len(doc.blocks) >= 0.9
    figs = [b for b in doc.blocks if b.block_type == "figure"]
    assert all(b.image_path for b in figs)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("[03_parser] all regression tests passed")
