"""질의 파이프라인 진입점 (07). 인덱스 로드 → 5 tool registry → BaseAgent.run.

- 이미지 추적: vendor base.py의 AgentContext 참조를 ImageAgentContext로 교체(loop·vendor 불변).
- Reader = GPT-4.1 mini(멀티모달), config의 reader 블록. image_read의 VLM도 이 Reader.
"""
from __future__ import annotations

import base64
import os
from typing import Any, Callable, Dict, Optional

import src.agent  # noqa: F401  (vendor path 부트스트랩)
import arag.agent.base as _agent_base
from arag.agent.base import BaseAgent
from arag.core.llm import LLMClient

from src.agent.context import ImageAgentContext
from src.config import Config
from src.indexing import IndexStore
from src.retrieval import default_tools

# loop·vendor를 건드리지 않고 이미지 추적 컨텍스트를 주입(서브클래스 우선).
_agent_base.AgentContext = ImageAgentContext

SYSTEM_PROMPT = (
    "You are a precise technical document assistant. Use the tools to ground every answer. "
    "For 'which page/figure' questions use page_index_search. "
    "Only call image_read when text is insufficient (e.g. bitfield diagrams). Cite page numbers."
)


def make_reader_vlm(cfg: Config) -> Callable[[str], str]:
    """image_path → Reader(멀티모달)로 이미지 설명 텍스트 (별도 VLM 백엔드 없음, 사실 8)."""
    endpoint = cfg.get("reader.endpoint", "https://api.openai.com/v1/chat/completions")
    model = cfg.get("reader.model", "gpt-4.1-mini")
    key = cfg.require_key("reader.api_key_env")

    def _read(image_path: str) -> str:
        import requests
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe this technical figure precisely "
                                         "(bitfields, labels, values). Be concise."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
        }
        r = requests.post(endpoint, headers={"Authorization": f"Bearer {key}"},
                          json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    return _read


def build_agent(cfg: Config, reader_fn: Optional[Callable[[str], str]] = None) -> BaseAgent:
    paths = cfg.index_paths()
    model_name = cfg.get("embedding.model", "sentence-transformers/all-MiniLM-L6-v2")
    store = IndexStore.load(paths, model_name)  # 빈/없는 인덱스면 명확한 에러(가드)
    reader_fn = reader_fn or make_reader_vlm(cfg)
    tools = default_tools(paths, model_name, reader_fn, store.page_index, store.chunks)
    llm = LLMClient(
        model=cfg.get("reader.model", "gpt-4.1-mini"),
        api_key=cfg.require_key("reader.api_key_env"),
        base_url=cfg.get("reader.endpoint", "https://api.openai.com/v1").replace(
            "/chat/completions", ""),
    )
    from src.retrieval import build_registry
    return BaseAgent(
        llm_client=llm, tools=build_registry(tools), system_prompt=SYSTEM_PROMPT,
        max_loops=cfg.get("agent.max_loops", 10),
        max_token_budget=cfg.get("agent.max_token_budget", 128000),
    )


def answer(query: str, config_path: str = "config.yaml") -> Dict[str, Any]:
    cfg = Config.load(config_path)
    return build_agent(cfg).run(query)
