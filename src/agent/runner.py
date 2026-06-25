"""질의 파이프라인 진입점 (07). 인덱스 로드 → 5 tool registry → BaseAgent.run.

- 이미지 추적: vendor base.py의 AgentContext 참조를 ImageAgentContext로 교체(loop·vendor 불변).
- Reader = OpenAI(GPT-4.1 mini) 또는 Claude(Anthropic) — 설치 시 config.reader.provider로 선택. image_read VLM도 동일.
"""
from __future__ import annotations

import base64
import os
from typing import Any, Callable, Dict, Optional

import arag.agent.base as _agent_base
from arag.agent.base import BaseAgent
from arag.core.llm import LLMClient

from src.agent.context import ImageAgentContext
from src.config import Config
from src.indexing import IndexStore
from src.retrieval import default_tools

# loop·vendor를 건드리지 않고 이미지 추적 컨텍스트를 주입(서브클래스 우선).
_agent_base.AgentContext = ImageAgentContext


def _reader_key(rc: Dict[str, str]) -> str:
    """Reader 프로바이더가 요구하는 환경변수에서 키를 읽는다(평문 금지). 없으면 명확한 에러."""
    name = rc["api_key_env"]
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Reader({rc['provider']}) 키가 없습니다. 환경변수 '{name}' 를 .env에 설정하세요."
        )
    return val

SYSTEM_PROMPT = (
    "You are a precise technical-document assistant for any hardware manual. Ground every answer in the tools.\n"
    "- Be token-efficient: make the FEWEST tool calls needed, and answer as soon as you have enough. "
    "Do not repeat a search that already returned nothing — change the term or tool instead.\n"
    "- 'Which page/figure' → page_index_search. Cite the PRINTED page label exactly as shown in results "
    "(formats vary by manual, e.g. 'E2-2804', '12-3', '45'); never present a chunk id / element number as a page.\n"
    "- Figures are often not captioned 'diagram'. Find one by its TOPIC (semantic_search/keyword_search) or a "
    "nearby heading; a figure entry appears with its page label and image file. To actually READ it, call "
    "image_read on that image file — don't claim to have read a figure otherwise.\n"
    "- If the document doesn't contain the answer, say so plainly; never fabricate."
)


def make_reader_vlm(cfg: Config) -> Callable[[str], str]:
    """image_path → Reader(멀티모달)로 이미지 설명 텍스트 (별도 VLM 백엔드 없음, 사실 8).
    Reader provider(openai|anthropic/claude)는 config.reader_config()가 해석 — 둘 다 OpenAI 호환."""
    rc = cfg.reader_config()
    endpoint, model, key = rc["endpoint"], rc["model"], _reader_key(rc)

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
    rc = cfg.reader_config()
    if rc["provider"] == "claude_code":
        # Claude Code 내장 모델(claude CLI) — 외부 Reader API·키 불필요.
        from src.agent.claude_cli import ClaudeCliLLMClient, claude_cli_reader
        reader_fn = reader_fn or claude_cli_reader()
        llm = ClaudeCliLLMClient()
    else:
        reader_fn = reader_fn or make_reader_vlm(cfg)
        llm = LLMClient(model=rc["model"], api_key=_reader_key(rc),
                        base_url=rc["endpoint"].replace("/chat/completions", ""))
    tools = default_tools(paths, model_name, reader_fn, store.page_index, store.chunks)
    from src.retrieval import build_registry
    return BaseAgent(
        llm_client=llm, tools=build_registry(tools), system_prompt=SYSTEM_PROMPT,
        max_loops=cfg.get("agent.max_loops", 10),
        max_token_budget=cfg.get("agent.max_token_budget", 128000),
    )


def answer(query: str, config_path: str = "config.yaml") -> Dict[str, Any]:
    cfg = Config.load(config_path)
    return build_agent(cfg).run(query)
