"""검색 단계 tool 5종 (06). vendor 3종 + 신규 2종을 ToolRegistry에 등록."""
from typing import Any, Callable, List

from arag.tools.registry import ToolRegistry

from .image_read import ImageReadTool
from .page_index_search import PageIndexSearchTool

__all__ = ["PageIndexSearchTool", "ImageReadTool", "build_registry", "default_tools"]


def build_registry(tools: List[Any]) -> ToolRegistry:
    """tool 리스트를 ToolRegistry에 등록(사실 2: register만, 코어 수정 없음)."""
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def default_tools(paths: dict, model_name: str, reader_fn: Callable[[str], str],
                  page_index, chunks: dict) -> List[Any]:
    """프로덕션 5종: vendor(keyword/semantic/read_chunk) + 신규(page_index_search/image_read).

    vendor 도구는 디스크 산출물(chunks.json, sentence_index.pkl)에 의존하므로
    빌드된 인덱스가 있을 때만 구성된다.
    """
    from arag.tools.keyword_search import KeywordSearchTool
    from arag.tools.read_chunk import ReadChunkTool
    from arag.tools.semantic_search import SemanticSearchTool

    chunks_file = paths["chunks"]
    index_dir = paths["index"]
    return [
        KeywordSearchTool(chunks_file=chunks_file),
        SemanticSearchTool(chunks_file=chunks_file, index_dir=index_dir, model_name=model_name),
        ReadChunkTool(chunks_file=chunks_file),
        PageIndexSearchTool(page_index=page_index, chunks=chunks),
        ImageReadTool(reader_fn=reader_fn),
    ]
