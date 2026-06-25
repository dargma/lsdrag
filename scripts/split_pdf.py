"""ARM 매뉴얼 등 큰 PDF를 파서 소화 크기(≤100p)로 분할 (03_parser 보조).

예: python scripts/split_pdf.py --input examples/ARMv8-Reference-Manual.pdf \
        --ranges "1-60,61-120" --out examples/parts/
각 조각은 별도 파일 = 별도 doc_id.
"""
import argparse
import os
import sys


def parse_ranges(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-")
        out.append((int(a), int(b)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--ranges", required=True, help='예: "1-60,61-120"')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print("pypdf 미설치. `pip install pypdf`", file=sys.stderr)
        return 2

    if not os.path.exists(args.input):
        print(f"입력 PDF 없음: {args.input}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    reader = PdfReader(args.input)
    n = len(reader.pages)
    base = os.path.splitext(os.path.basename(args.input))[0]

    for i, (a, b) in enumerate(parse_ranges(args.ranges), 1):
        if b - a + 1 > 100:
            print(f"⚠️ part{i} {a}-{b} 는 100p 초과 — Upstage sync 한도. 더 잘게 나누세요.",
                  file=sys.stderr)
        w = PdfWriter()
        for p in range(a - 1, min(b, n)):
            w.add_page(reader.pages[p])
        out_path = os.path.join(args.out, f"{base}_part{i}.pdf")
        with open(out_path, "wb") as f:
            w.write(f)
        print(f"[split_pdf] part{i}: pages {a}-{b} → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
