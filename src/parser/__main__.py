"""python -m src.parser --pdf X --out DIR  → 1개 PDF 파싱 후 IR(JSON) 저장 (단독 실행).

실제 Upstage 호출이므로 UP_TOKEN 필요. 키 없으면 명확한 에러.
"""
import argparse
import json
import os

from . import parse_pdf, upstage_to_ir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", default="data/ir")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--key-env", default="UP_TOKEN")
    args = ap.parse_args()

    doc_id = os.path.splitext(os.path.basename(args.pdf))[0]
    resp = parse_pdf(args.pdf, api_key_env=args.key_env,
                     base64_categories=["figure", "chart", "table"])
    doc = upstage_to_ir(resp, doc_id=doc_id, title=doc_id, images_dir=args.images)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{doc_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)

    figs = [b for b in doc.blocks if b.block_type == "figure"]
    print(f"[03_parser] {doc_id}: {len(doc.blocks)} blocks, "
          f"{len(figs)} figures, {sum(1 for b in figs if b.image_path)} with image_path")
    print(f"  → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
