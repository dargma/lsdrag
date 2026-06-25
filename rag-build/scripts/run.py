"""rag-build skill 진입점 — DB 빌드/증분. 엔진 src.indexing.build 호출(얇은 껍데기)."""
import argparse

from _bootstrap import load_engine


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add"); ap.add_argument("--remove"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_engine(args.config)
    from src.indexing import build as B
    if args.add:
        return B.cmd_add(cfg, args.add)
    if args.remove:
        return B.cmd_remove(cfg, args.remove)
    if args.list:
        return B.cmd_list(cfg, as_json=args.json)
    return B.cmd_build(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
