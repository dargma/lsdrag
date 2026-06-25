"""python -m src.schema --selftest → IR 왕복 무손실 확인 (단독 실행, 단계 분리 증명)."""
import json
import sys

from . import ParsedBlock, ParsedDoc


def selftest() -> int:
    doc = ParsedDoc(
        doc_id="arm_part1",
        title="ARM v8-A — Part 1",
        date="2026-06-25",
        blocks=[
            ParsedBlock(text="The SCTLR_EL1 register controls...", page_no=12,
                        block_type="text", heading="System Control", chunk_id="c1"),
            ParsedBlock(text="Field layout", page_no=13, block_type="table", chunk_id="c2"),
            ParsedBlock(text="Bitfield diagram", page_no=13, block_type="figure",
                        figure_no="2", image_path="data/images/arm_part1_p13_f2.png", chunk_id="c3"),
            ParsedBlock(text="Figure 2. SCTLR_EL1 bitfields", page_no=13,
                        block_type="caption", chunk_id="c4"),
        ],
    )
    s = json.dumps(doc.to_dict())
    back = ParsedDoc.from_dict(json.loads(s))
    assert back.to_dict() == doc.to_dict(), "round-trip not lossless"
    assert back.blocks[2].image_path, "figure image_path lost"
    print("[02_schema_IR] round-trip lossless OK")
    print(json.dumps(back.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(selftest())
