"""skill 스크립트 공용 부트스트랩 (09).

배포된 skill(~/.claude/skills/rag/)은 엔진(src/)을 config.yaml의 engine.root로 찾는다.
엔진(src/)과 config.yaml에만 의존한다. 그 외 개발 산출물에 의존하지 않는다.

config 탐색 순서: $RAG_CONFIG > ./config.yaml(cwd) > ${CLAUDE_SKILL_DIR}/config.yaml
"""
from __future__ import annotations

import os
import sys


def find_config() -> str:
    candidates = [
        os.environ.get("RAG_CONFIG"),
        os.path.join(os.getcwd(), "config.yaml"),
        os.path.join(os.environ.get("CLAUDE_SKILL_DIR", ""), "config.yaml"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "config.yaml 을 찾지 못했습니다.\n"
        "  해결: 엔진 디렉터리에서 실행하거나 RAG_CONFIG=/path/config.yaml 설정."
    )


def load_engine(config_path: str = None):
    """config를 읽고 engine.root(src/)를 sys.path에 올린 뒤 Config 반환."""
    config_path = config_path or find_config()
    import yaml
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    root = (data.get("engine", {}) or {}).get("root", ".")
    engine_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(config_path)), root))
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)
    from src.config import Config
    return Config.load(config_path)
