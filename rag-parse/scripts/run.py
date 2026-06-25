"""rag-parse skill 진입점 — PDF 1개 → IR 변환·점검. 엔진 src.parser 호출(얇은 껍데기)."""
import argparse
import json
import os

from _bootstrap import load_engine


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out", default=None)
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_engine(args.config)
    from src.parser import parse_pdf, upstage_to_ir

    doc_id = os.path.splitext(os.path.basename(args.pdf))[0]
    resp = parse_pdf(args.pdf, api_key_env=cfg.get("parser.api_key_env", "UP_TOKEN"),
                     endpoint=cfg.get("parser.endpoint"), model=cfg.get("parser.model", "document-parse"),
                     base64_categories=cfg.get("parser.base64_categories"))
    doc = upstage_to_ir(resp, doc_id=doc_id, title=doc_id, images_dir=cfg.path("images_out"))

    out_dir = args.out or cfg.path("ir_out")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{doc_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)

    figs = [b for b in doc.blocks if b.block_type == "figure"]
    tables = [b for b in doc.blocks if b.block_type == "table"]
    print(f"[rag-parse] {doc_id}: {len(doc.blocks)} blocks, {len(tables)} tables, "
          f"{len(figs)} figures ({sum(1 for b in figs if b.image_path)} with image) → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
