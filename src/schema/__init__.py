"""IR 스키마 — 단계 1↔2 계약 (02_schema_IR)."""
from .ir import BLOCK_TYPES, IRValidationError, ParsedBlock, ParsedDoc

__all__ = ["ParsedBlock", "ParsedDoc", "IRValidationError", "BLOCK_TYPES"]
