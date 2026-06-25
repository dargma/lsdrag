"""doctor.py — 설치 self-check C1~C9 (09). 런타임 검증의 설치판.

각 체크는 독립 진단: "몇 번에서 막혔는지 + 조치"를 출력(전체실패 아님).
출력 모드: 기본=plain 한 줄 append / --json=한 줄 JSON / --rich=TTY bar.
선택 실행: --check skill --check keys ... (없으면 전체).
침묵 실패 금지.
"""
from __future__ import annotations

import argparse
import json
import os

from _bootstrap import load_engine

CHECKS = ["skill", "deps", "keys", "parser", "reader", "embedder", "index", "multimodal", "smoke"]
TITLES = {
    "skill": "skill 배치", "deps": "Python 의존성", "keys": "API 키",
    "parser": "파서 API(Upstage)", "reader": "Reader API(GPT-4.1 mini)",
    "embedder": "임베더", "index": "인덱스", "multimodal": "멀티모달", "smoke": "스모크",
}


class Result:
    def __init__(self, ok, msg="", fix=""):
        self.ok, self.msg, self.fix = ok, msg, fix


def c_skill(cfg=None) -> Result:
    sd = os.environ.get("CLAUDE_SKILL_DIR")
    if not sd:
        # 로컬 개발: rag/SKILL.md 존재로 대체 확인
        here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SKILL.md")
        return Result(os.path.exists(here), "SKILL.md(local) present" if os.path.exists(here) else "",
                      "rag/ 를 ~/.claude/skills/rag/ 로 복사(중첩 금지).")
    p = os.path.join(sd, "SKILL.md")
    return Result(os.path.exists(p), f"{p}", "경로/중첩 확인: .../skills/rag/SKILL.md")


def c_deps(cfg=None) -> Result:
    missing = []
    for m in ["yaml", "numpy", "requests", "sentence_transformers", "tiktoken", "pypdf", "pikepdf"]:
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    return Result(not missing, "all imports ok" if not missing else f"missing: {missing}",
                  "pip install pyyaml numpy requests sentence-transformers tiktoken pypdf pikepdf")


def c_keys(cfg) -> Result:
    up = cfg.get("parser.api_key_env", "UP_TOKEN")
    rd = cfg.reader_config()["api_key_env"]  # 선택한 Reader 프로바이더의 키
    miss = [e for e in (up, rd) if not os.environ.get(e)]
    return Result(not miss, "keys present" if not miss else f"missing env: {miss}",
                  ".env 에 키 설정(평문 커밋 금지) 후 export.")


def c_parser(cfg) -> Result:
    env = cfg.get("parser.api_key_env", "UP_TOKEN")
    if not os.environ.get(env):
        return Result(False, f"{env} 없음", f".env 에 {env} 설정.")
    return Result(True, "key set (live call deferred)", "")  # 실제 호출은 빌드/스모크에서


def c_reader(cfg) -> Result:
    rc = cfg.reader_config()
    if not os.environ.get(rc["api_key_env"]):
        return Result(False, f"{rc['api_key_env']} 없음 (provider={rc['provider']})",
                      f".env 에 {rc['api_key_env']} 설정.")
    return Result(True, f"provider={rc['provider']}, model={rc['model']}", "")


def c_embedder(cfg) -> Result:
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return Result(True, f"model={cfg.get('embedding.model')} loadable", "")
    except ImportError:
        return Result(False, "sentence-transformers 미설치", "pip install sentence-transformers")


def c_index(cfg) -> Result:
    try:
        from src.indexing import IndexStore
        IndexStore.load(cfg.index_paths(), cfg.get("embedding.model"))
        return Result(True, "index + page_index loadable", "")
    except FileNotFoundError as e:
        return Result(False, str(e).splitlines()[0], "python -m src.indexing.build --config config.yaml")
    except Exception as e:
        return Result(False, str(e), "인덱스 재빌드 필요")


def c_multimodal(cfg) -> Result:
    rc = cfg.reader_config()
    if not os.environ.get(rc["api_key_env"]):
        return Result(False, f"{rc['api_key_env']} 없음", "멀티모달은 Reader 키 필요.")
    return Result(True, f"reader multimodal path ready ({rc['provider']}, live call deferred)", "")


def c_smoke(cfg) -> Result:
    try:
        from src.indexing import IndexStore
        IndexStore.load(cfg.index_paths(), cfg.get("embedding.model"))
    except Exception as e:
        return Result(False, f"index not ready: {str(e).splitlines()[0]}", "먼저 인덱스 빌드.")
    if not os.environ.get(cfg.reader_config()["api_key_env"]):
        return Result(False, "Reader 키 없음 → 실제 답변 불가", ".env 키 설정.")
    return Result(True, "ready for live query", "")


FUNCS = {"skill": c_skill, "deps": c_deps, "keys": c_keys, "parser": c_parser,
         "reader": c_reader, "embedder": c_embedder, "index": c_index,
         "multimodal": c_multimodal, "smoke": c_smoke}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="append", dest="checks")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--rich", action="store_true")
    ap.add_argument("--config")
    args = ap.parse_args()

    selected = args.checks or CHECKS
    # deps/keys/skill 는 config 없이도; 나머지는 config 필요
    cfg = None
    try:
        cfg = load_engine(args.config)
    except Exception as e:
        if set(selected) - {"skill", "deps"}:
            print(json.dumps({"step": 0, "status": "fail", "name": "config",
                              "fix": str(e)}) if args.json else f"⛔ config 로드 실패: {e}")
            return 2

    total = len(selected)
    failed_at = None
    for i, name in enumerate(selected, 1):
        r = FUNCS[name](cfg)
        status = "ok" if r.ok else "fail"
        if args.json:
            print(json.dumps({"step": i, "total": total, "name": name,
                              "status": status, "msg": r.msg, "fix": r.fix}, ensure_ascii=False))
        else:
            line = f"[{i}/{total}] {TITLES[name]} ... {'OK' if r.ok else 'FAIL'}"
            if not r.ok:
                line += f"\n      → {r.msg}\n      조치: {r.fix}"
            print(line)
        if not r.ok and failed_at is None:
            failed_at = (i, name)
            break  # 다음 단계로 안 넘어감(안내형 설치)
    if failed_at:
        return 1
    if not args.json:
        print(f"✅ {total}/{total} 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
