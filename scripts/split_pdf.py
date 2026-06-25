"""큰 PDF를 파서 소화 크기(≤100p, <50MB)로 분할 (03_parser 보조).

P1 개선: 단순 페이지 복사는 공유 리소스(폰트 등)를 통째로 끌고 와 조각이 원본만큼 커진다.
pikepdf의 remove_unreferenced_resources()로 해당 조각이 실제 참조하는 리소스만 남겨 용량을 줄인다.
pikepdf가 없으면 pypdf로 폴백(용량 경고).

예: python scripts/split_pdf.py --input manual.pdf --ranges "1-60,61-120" --out parts/
각 조각 = 별도 파일 = 별도 doc_id.
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


def _split_pikepdf(inp, ranges, out_dir, base):
    import pikepdf
    src = pikepdf.open(inp)
    n = len(src.pages)
    for i, (a, b) in enumerate(ranges, 1):
        dst = pikepdf.Pdf.new()
        for p in range(a - 1, min(b, n)):
            dst.pages.append(src.pages[p])
        dst.remove_unreferenced_resources()  # 미사용 공유 리소스 제거 → 용량 감소
        path = os.path.join(out_dir, f"{base}_part{i}.pdf")
        dst.save(path, recompress_flate=True)
        _report(path, a, b)
    src.close()


def _split_pypdf(inp, ranges, out_dir, base):
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(inp)
    n = len(reader.pages)
    for i, (a, b) in enumerate(ranges, 1):
        w = PdfWriter()
        for p in range(a - 1, min(b, n)):
            w.add_page(reader.pages[p])
        try:
            for page in w.pages:
                page.compress_content_streams()
        except Exception:
            pass
        path = os.path.join(out_dir, f"{base}_part{i}.pdf")
        with open(path, "wb") as f:
            w.write(f)
        _report(path, a, b)


def _report(path, a, b):
    mb = os.path.getsize(path) / 1e6
    warn = "  ⚠️ >50MB (Upstage sync 한도 초과 — 더 잘게 나누세요)" if mb > 50 else ""
    if b - a + 1 > 100:
        warn += "  ⚠️ >100p (sync 한도)"
    print(f"[split_pdf] pages {a}-{b} → {path}  ({mb:.1f} MB){warn}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--ranges", required=True, help='예: "1-60,61-120"')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not os.path.exists(args.input):
        print(f"입력 PDF 없음: {args.input}", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    ranges = parse_ranges(args.ranges)
    base = os.path.splitext(os.path.basename(args.input))[0]
    try:
        import pikepdf  # noqa: F401
        _split_pikepdf(args.input, ranges, args.out, base)
    except ImportError:
        print("pikepdf 없음 → pypdf 폴백(조각이 커질 수 있음). `pip install pikepdf` 권장.",
              file=sys.stderr)
        try:
            _split_pypdf(args.input, ranges, args.out, base)
        except ImportError:
            print("pypdf도 없음. `pip install pypdf`", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
