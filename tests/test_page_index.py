"""04_page_index 회귀 — 구조 조회 정확도 + 빈결과 graceful + 증분/영속화."""
import tempfile

from src.page_index import PageIndex
from src.schema import ParsedBlock, ParsedDoc


def _doc(doc_id="d1"):
    return ParsedDoc(doc_id=doc_id, title="t", blocks=[
        ParsedBlock(text="intro", page_no=3, block_type="text", heading="Overview", chunk_id=f"{doc_id}:0"),
        ParsedBlock(text="bitfield", page_no=3, block_type="figure", figure_no="2",
                    image_path="data/images/x.png", chunk_id=f"{doc_id}:1"),
    ])


def test_query_page_and_figure():
    idx = PageIndex(); idx.add_doc(_doc())
    hit = idx.query(page=3, figure_no="2")
    assert len(hit) == 1 and hit[0].chunk_id == "d1:1"
    assert hit[0].image_path == "data/images/x.png"  # 본문+이미지명 동반


def test_query_heading_partial():
    idx = PageIndex(); idx.add_doc(_doc())
    assert idx.query(heading="over")  # 부분 지정


def test_missing_location_empty_not_error():
    idx = PageIndex(); idx.add_doc(_doc())
    assert idx.query(page=999) == []  # 예외 아님


def test_incremental_add_remove_idempotent():
    idx = PageIndex()
    idx.add_doc(_doc("a")); idx.add_doc(_doc("b"))
    assert idx.doc_ids() == ["a", "b"]
    idx.add_doc(_doc("a"))  # 재추가=갱신, 중복 없음
    assert len(idx.query(doc_id="a")) == 2
    idx.remove_doc("a")
    assert idx.doc_ids() == ["b"] and idx.query(doc_id="a") == []


def test_persist_reload():
    idx = PageIndex(); idx.add_doc(_doc())
    with tempfile.TemporaryDirectory() as d:
        idx.save(d)
        again = PageIndex.load(d)
        assert len(again) == len(idx) and again.query(page=3, figure_no="2")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("[04_page_index] all regression tests passed")
