"""Claude Code 내장 모델(`claude` CLI, 헤드리스)을 Reader로 쓰는 어댑터.

API 키 없이 지금 Claude Code에서 도는 모델을 그대로 쓴다(provider=claude_code).
- image_read(VLM): `claude -p`가 이미지 파일을 읽어 설명.
- agent 루프 LLM: A-RAG의 `LLMClient.chat(messages, tools)` 계약을 흉내내, 매 스텝
  `claude -p`에 "다음 한 단계"를 JSON으로 물어 tool 호출/최종답을 결정한다.

전제: 로컬에 `claude` CLI 존재(Claude Code 설치). 외부 Reader API·키 불필요.
"""
from __future__ import annotations

import json
import re
import subprocess
from typing import Any, Dict, List, Optional

_JSON = re.compile(r"\{.*\}", re.S)


def _run_claude(prompt: str, timeout: int = 200) -> str:
    p = subprocess.run(["claude", "-p", prompt], capture_output=True, text=True,
                       timeout=timeout, stdin=subprocess.DEVNULL)
    if p.returncode != 0:
        raise RuntimeError(f"claude CLI 실패: {p.stderr.strip()[:200]}")
    return p.stdout.strip()


def _extract_json(text: str) -> Dict[str, Any]:
    m = _JSON.search(text or "")
    if not m:
        return {"final": text}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"final": text}


def claude_cli_reader(image_prompt: str = None):
    """image_read용 reader_fn: image_path → 설명 텍스트 (claude CLI가 이미지 읽음)."""
    instr = image_prompt or ("Read the image file and describe this technical figure precisely "
                             "(states/fields/labels/transitions). If it is NOT a real figure "
                             "(e.g. a header banner), say so plainly. Be concise.")

    def _read(image_path: str) -> str:
        return _run_claude(f"Read the image at {image_path}. {instr}")
    return _read


class ClaudeCliLLMClient:
    """A-RAG BaseAgent가 기대하는 chat() 계약을 claude CLI로 구현."""

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or "claude-code-builtin"

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        schemas = [t.get("function", t) for t in (tools or [])]
        # 도구별 '이름(인자…)' 한 줄 요약 — Claude가 정확한 인자명을 쓰게.
        hints = []
        for s in schemas:
            params = list(((s.get("parameters") or {}).get("properties") or {}).keys())
            hints.append(f"  - {s.get('name')}({', '.join(params)})")
        tool_hint = "\n".join(hints)
        convo = []
        for m in messages:
            role, content = m.get("role"), m.get("content")
            if role == "tool":
                convo.append(f"[TOOL RESULT]\n{str(content)[:1500]}")
            elif m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    convo.append(f"[YOU CALLED] {tc['function']['name']}({tc['function']['arguments']})")
            elif content:
                convo.append(f"[{str(role).upper()}] {content}")
        prompt = (
            "You drive a retrieval agent over a technical manual. Decide the SINGLE next step.\n"
            "Tools and their EXACT argument names (use only these):\n" + tool_hint + "\n\n"
            "Rules:\n"
            "- Topic/subject search → semantic_search(query=...) or keyword_search(keywords=[...]).\n"
            "- page_index_search takes ONLY page/figure_no/heading/doc_id (NO 'query'). Use it for "
            "location and to surface figure images.\n"
            "- Copy chunk_ids and image file paths EXACTLY as they appear in tool results.\n"
            "- To read a figure, call image_read(image_path=<exact path from results>).\n"
            "- As soon as you can answer (you have the content and the printed page label), FINISH. "
            "Do not keep searching. Never invent a page label.\n\n"
            "Conversation so far:\n" + "\n".join(convo)[:8000] + "\n\n"
            "Reply with EXACTLY ONE JSON object and nothing else:\n"
            '  call a tool -> {\"tool\": \"<name>\", \"args\": { ... }}\n'
            '  finish      -> {\"final\": \"<full grounded answer citing the printed page label>\"}'
        )
        obj = _extract_json(_run_claude(prompt))
        if obj.get("tool"):
            return {"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "claude_cli", "function": {
                    "name": obj["tool"], "arguments": json.dumps(obj.get("args", {}))}}]},
                "cost": 0.0}
        return {"message": {"role": "assistant", "content": obj.get("final", "")}, "cost": 0.0}
