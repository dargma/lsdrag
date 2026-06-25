"""파서 단계 (03). client=HTTP 경계, adapter=교체 경계(응답→IR)."""
from .adapter import upstage_to_ir
from .client import ParserAPIError, ParserConfigError, parse_pdf

__all__ = ["upstage_to_ir", "parse_pdf", "ParserConfigError", "ParserAPIError"]
