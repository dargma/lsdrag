"""Upstage Document Parse 클라이언트 (03_parser).

HTTP 경계만 담당. 응답 JSON → IR 변환은 adapter.py(교체 경계).
키는 env(UP_TOKEN)에서만 읽는다 — 평문 금지.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

DEFAULT_ENDPOINT = "https://api.upstage.ai/v1/document-digitization"
DEFAULT_MODEL = "document-parse"


class ParserConfigError(RuntimeError):
    """키 없음 등 설정 오류 — 사람 말 에러로 드러낸다(침묵 실패 금지)."""


class ParserAPIError(RuntimeError):
    """Upstage 호출 실패."""


def _require_key(api_key_env: str) -> str:
    key = os.environ.get(api_key_env)
    if not key:
        raise ParserConfigError(
            f"파서 키가 없습니다. 환경변수 '{api_key_env}' 를 설정하세요.\n"
            f"  방법: .env 에 {api_key_env}=... 기입 후 셸에 export. (키는 평문 커밋 금지)"
        )
    return key


def parse_pdf(
    pdf_path: str,
    api_key_env: str = "UP_TOKEN",
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    base64_categories: Optional[List[str]] = None,
    output_formats: Optional[List[str]] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    """PDF 1개를 Upstage로 파싱해 원시 응답(dict)을 반환. (adapter가 IR로 변환)

    output_formats: content에 채울 표현. 기본 ["text","markdown","html"] —
    text를 요청해야 content.text가 채워져 HTML 오염을 피한다(없으면 adapter가 태그 제거).
    """
    key = _require_key(api_key_env)
    if not os.path.exists(pdf_path):
        raise ParserConfigError(f"입력 PDF가 없습니다: {pdf_path}")
    try:
        import requests
    except ImportError as e:
        raise ParserConfigError("requests 미설치. `pip install requests`") from e

    data = {"model": model}
    data["output_formats"] = str(output_formats or ["text", "markdown", "html"])
    if base64_categories:
        # Upstage는 리스트를 반복 필드로 받는다
        data["base64_encoding"] = str(base64_categories)
    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {key}"},
                data=data,
                files={"document": f},
                timeout=timeout,
            )
    except Exception as e:
        raise ParserAPIError(f"Upstage 호출 실패({pdf_path}): {e}. 네트워크·엔드포인트 확인.") from e

    if resp.status_code != 200:
        raise ParserAPIError(
            f"Upstage {resp.status_code} for {pdf_path}: {resp.text[:300]}\n"
            f"  확인: 키 유효성, 페이지 수(sync 100p 한도), 파일 크기(50MB)."
        )
    return resp.json()
