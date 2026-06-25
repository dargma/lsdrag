"""python -m src.agent --query "..." --config config.yaml  → 답변 (단독 실행)."""
import argparse

from src.agent.runner import answer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    result = answer(args.query, args.config)
    print(result.get("answer", "(no answer)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
