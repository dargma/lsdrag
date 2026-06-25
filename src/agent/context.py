"""AgentContext 확장 — 이미지 추적 (07). 유일한 코어 확장(사실 4).

vendor를 직접 고치지 않는다: AgentContext를 서브클래스로 확장하고,
runner가 vendor base.py의 모듈 참조(AgentContext)를 이 클래스로 교체(monkeypatch).
청크 추적 패턴(read_chunk_ids/mark/is)을 이미지에 복제.
"""
from __future__ import annotations

import src.agent  # noqa: F401  (vendor path 부트스트랩)
from arag.core.context import AgentContext


class ImageAgentContext(AgentContext):
    def __init__(self) -> None:
        super().__init__()
        self.read_image_ids = set()

    def mark_image_read(self, image_id: str) -> None:
        self.read_image_ids.add(str(image_id))

    def is_image_read(self, image_id: str) -> bool:
        return str(image_id) in self.read_image_ids
