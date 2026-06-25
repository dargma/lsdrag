"""02_schema_IR 회귀 테스트 (작업 중 검증을 회귀로 고정 + 사용 중 가드 확인).

실행: python -m tests.test_schema   (또는 pytest tests/test_schema.py)
"""
import json

from src.schema import ParsedBlock, ParsedDoc, IRValidationError


def _sample() -> ParsedDoc:
    return ParsedDoc(
        doc_id="d1", title="t", date="2026-06-25",
        blocks=[
            ParsedBlock(text="x", page_no=1, block_type="text", chunk_id="c1"),
            ParsedBlock(text="fig", page_no=2, block_type="figure",
                        figure_no="1", image_path="data/images/a.png", chunk_id="c2"),
        ],
    )


def test_roundtrip_lossless():
    doc = _sample()
    back = ParsedDoc.from_dict(json.loads(json.dumps(doc.to_dict())))
    assert back.to_dict() == doc.to_dict()


def test_figure_keeps_image_path():
    back = ParsedDoc.from_dict(_sample().to_dict())
    assert back.blocks[1].image_path == "data/images/a.png"


def test_rejects_missing_required_keys():
    try:
        ParsedDoc.from_dict({"title": "no doc_id"})
        assert False, "should have raised"
    except IRValidationError:
        pass


def test_rejects_bad_block_type():
    try:
        ParsedBlock(text="x", page_no=1, block_type="bogus").validate()
        assert False, "should have raised"
    except IRValidationError:
        pass


def test_figure_without_image_allowed():
    # figure image_path는 권장이지 블록단위 하드요구 아님(파서가 이미지를 안 줄 수 있음).
    # 비율 검증은 03_parser 게이트 책임.
    ParsedBlock(text="x", page_no=1, block_type="figure").validate()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("[02_schema_IR] all regression tests passed")
