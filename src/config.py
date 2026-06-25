"""config.yaml 로더 — 경로·모델·엔드포인트의 단일 출처. 키는 env에서만.

모든 스크립트는 여기만 읽는다(경로 하드코딩 금지). engine.root 기준으로 상대경로 해석.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml

# Reader(LLM/VLM) 프로바이더 프리셋 — 설치 시 한 줄(reader.provider)로 선택.
# 둘 다 OpenAI 호환 chat/completions 엔드포인트라 동일 코드로 동작(멀티모달·tool calling 지원).
READER_PRESETS = {
    # OpenAI GPT-4.1 mini (기본)
    "openai": {"endpoint": "https://api.openai.com/v1/chat/completions",
               "model": "gpt-4.1-mini", "api_key_env": "OPENAI_API_KEY"},
    # Claude(Anthropic) — Claude Code에서 쓰는 모델을 Reader로. OpenAI 호환 엔드포인트 사용.
    "anthropic": {"endpoint": "https://api.anthropic.com/v1/chat/completions",
                  "model": "claude-sonnet-4-6", "api_key_env": "ANTHROPIC_API_KEY"},
    "claude": {"endpoint": "https://api.anthropic.com/v1/chat/completions",
               "model": "claude-sonnet-4-6", "api_key_env": "ANTHROPIC_API_KEY"},
}


class Config:
    def __init__(self, data: Dict[str, Any], base_dir: str) -> None:
        self._d = data
        self.base_dir = base_dir

    def reader_config(self) -> Dict[str, str]:
        """Reader 설정을 provider 프리셋 + 명시적 override로 해석.
        사용자는 reader.provider(openai|anthropic|claude) 한 줄만 정하면 되고,
        endpoint/model/api_key_env를 config에 적으면 그게 우선한다."""
        prov = (self.get("reader.provider") or "openai").lower()
        preset = dict(READER_PRESETS.get(prov, READER_PRESETS["openai"]))
        for k in ("endpoint", "model", "api_key_env"):
            v = self.get(f"reader.{k}")
            if v:
                preset[k] = v
        preset["provider"] = prov
        return preset

    @classmethod
    def load(cls, path: str = "config.yaml") -> "Config":
        if not os.path.exists(path):
            raise FileNotFoundError(f"config.yaml 없음: {path}. INSTALL.md 참조.")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        root = (data.get("engine", {}) or {}).get("root", ".")
        base = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(path)), root))
        return cls(data, base)

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self._d
        for k in dotted.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def path(self, name: str) -> str:
        """paths.<name> 을 engine.root 기준 절대경로로."""
        rel = self.get(f"paths.{name}")
        if rel is None:
            raise KeyError(f"config paths.{name} 미정의")
        return os.path.normpath(os.path.join(self.base_dir, rel))

    def index_paths(self) -> Dict[str, str]:
        return {
            "index": self.path("index"),
            "page_index": self.path("page_index"),
            "chunks": os.path.join(self.path("index"), "chunks.json"),
        }

    def require_key(self, dotted_env: str) -> str:
        """config의 *.api_key_env 가 가리키는 환경변수 값을 강제 로드."""
        env_name = self.get(dotted_env)
        if not env_name:
            raise KeyError(f"config {dotted_env} 미정의")
        val = os.environ.get(env_name)
        if not val:
            raise RuntimeError(
                f"환경변수 '{env_name}' 가 없습니다. .env 에 설정하세요(평문 커밋 금지)."
            )
        return val
