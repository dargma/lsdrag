"""docs.py — 문서 추가/제거/목록 (09). 엔진 증분 API(05) 호출. 얇은 껍데기."""
import argparse

from _bootstrap import load_engine


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("pdf")
    r = sub.add_parser("remove"); r.add_argument("doc_id")
    sub.add_parser("list")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_engine(args.config)
    from src.indexing import build as B

    if args.cmd == "add":
        return B.cmd_add(cfg, args.pdf)
    if args.cmd == "remove":
        return B.cmd_remove(cfg, args.doc_id)
    return B.cmd_list(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
