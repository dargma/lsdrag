"""uninstall.py — skill 손쉬운 삭제 (09).

기본: skill 폴더(.../skills/rag/)만 제거.
--purge-data: config의 인덱스/IR/이미지 데이터까지(확인 후, 대상 먼저 출력).
안전: 지울 경로를 먼저 보여주고, config 밖/시스템 경로는 거부. 침묵 삭제 금지.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys


def _skill_dir() -> str:
    sd = os.environ.get("CLAUDE_SKILL_DIR")
    if sd and os.path.basename(os.path.normpath(sd)) == "rag":
        return sd
    # fallback: 이 파일 기준 rag/
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _confirm(prompt: str) -> bool:
    if not sys.stdin.isatty():
        return os.environ.get("RAG_ASSUME_YES") == "1"
    return input(f"{prompt} [y/N] ").strip().lower() == "y"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge-data", action="store_true")
    ap.add_argument("--config")
    args = ap.parse_args()

    if args.purge_data:
        from _bootstrap import load_engine
        cfg = load_engine(args.config)
        targets = [cfg.path(k) for k in ("index", "page_index", "ir_out", "images_out")]
        targets = [t for t in targets if os.path.exists(t)]
        # 안전: config base_dir 안의 경로만 허용
        safe = [t for t in targets if os.path.abspath(t).startswith(os.path.abspath(cfg.base_dir))]
        rejected = [t for t in targets if t not in safe]
        print("삭제 대상 데이터:")
        for t in safe:
            print(f"  - {t}")
        for t in rejected:
            print(f"  ⚠️ 거부(엔진 밖 경로): {t}")
        if safe and _confirm("위 데이터를 삭제할까요?"):
            for t in safe:
                shutil.rmtree(t, ignore_errors=True)
            print("데이터 삭제됨.")
        else:
            print("데이터는 보존.")

    sd = _skill_dir()
    print(f"skill 폴더 삭제 대상: {sd}")
    if _confirm("skill을 삭제할까요?"):
        shutil.rmtree(sd, ignore_errors=True)
        print("✅ skill 삭제됨. (다시 설치하려면 INSTALL.md 1단계)")
    else:
        print("취소됨.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
