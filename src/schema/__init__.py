"""IR 스키마 — 단계 1↔2 계약 (02_schema_IR)."""
from .ir import ParsedBlock, ParsedDoc, IRValidationError, BLOCK_TYPES

__all__ = ["ParsedBlock", "ParsedDoc", "IRValidationError", "BLOCK_TYPES"]
