"""run.py — /rag 질의 실행 (09). 얇은 껍데기: 엔진 answer()만 호출."""
import argparse

from _bootstrap import load_engine


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_engine(args.config)
    from src.agent.runner import build_agent
    result = build_agent(cfg).run(" ".join(args.query))
    print(result.get("answer", "(no answer)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
