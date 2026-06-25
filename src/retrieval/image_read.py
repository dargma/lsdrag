"""image_read — 신규 tool (06). 이미지 → VLM 해석 → 텍스트 설명 (조건부 + 중복 방지).

사실 4: AgentContext 이미지 추적으로 VLM 중복 호출 차단(07이 read_image_ids 확장 제공).
사실 8: 별도 VLM 백엔드 없음 — Reader(GPT-4.1 mini, 멀티모달)가 이미지를 직접 처리.
구현 (b): 이미지를 텍스트 설명화해 반환 → 기존 (str,dict) 계약 유지, loop 변경 최소화.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Tuple

import src.agent  # noqa: F401  (vendor path 부트스트랩)
from arag.tools.base import BaseTool

# reader_fn(image_path) -> 설명 텍스트. 실제는 Reader(VLM) 호출, 테스트는 mock 주입.
ReaderFn = Callable[[str], str]


class ImageReadTool(BaseTool):
    def __init__(self, reader_fn: ReaderFn) -> None:
        self._reader = reader_fn

    @property
    def name(self) -> str:
        return "image_read"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "image_read",
                "description": (
                    "Interpret a figure/diagram image and return a TEXT description. "
                    "Use ONLY when text context is insufficient (e.g. bitfield diagrams). "
                    "Provide image_path from page_index_search results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"image_path": {"type": "string"}},
                    "required": ["image_path"],
                },
            },
        }

    def execute(self, context: "Any", image_path: str = "", **_: Any) -> Tuple[str, Dict[str, Any]]:
        if not image_path:
            return "image_read: no image_path provided.", {"tool": self.name, "error": "no_path"}
        # 중복 방지 (사실 4 패턴). context에 없으면 graceful.
        if _is_image_read(context, image_path):
            return (f"(Image '{os.path.basename(image_path)}' already interpreted earlier.)",
                    {"tool": self.name, "cached": True})
        if not os.path.exists(image_path):
            return (f"image_read: file not found ({image_path}). Proceeding with text only.",
                    {"tool": self.name, "error": "missing_file"})
        try:
            desc = self._reader(image_path)
        except Exception as e:  # VLM 실패 → 폴백 (사용 중 가드)
            return (f"image_read: VLM failed ({e}). Proceeding with text only.",
                    {"tool": self.name, "error": "vlm_failed"})
        _mark_image_read(context, image_path)
        _safe_log(context, self.name, len(desc or ""), {"tool": self.name, "image": image_path})
        return desc or "(empty description)", {"tool": self.name, "image": image_path}


def _is_image_read(context, path) -> bool:
    fn = getattr(context, "is_image_read", None)
    return bool(fn(path)) if callable(fn) else False


def _mark_image_read(context, path) -> None:
    fn = getattr(context, "mark_image_read", None)
    if callable(fn):
        fn(path)


def _safe_log(context, name, tokens, meta):
    if hasattr(context, "add_retrieval_log"):
        try:
            context.add_retrieval_log(name, tokens, meta)
        except Exception:
            pass
