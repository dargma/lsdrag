"""LSD-RAG 엔진 패키지.

vendored A-RAG는 내부에서 `from arag... import` (절대 import)를 쓴다. 그 패키지 루트
(src/agent/vendor)를 여기서 sys.path에 올린다 — `src`의 어떤 하위 모듈을 import하든
이 __init__가 먼저 실행되므로, 모듈별 import 순서에 의존하지 않는다(린터 정렬에 안전).
"""
import os
import sys

_VENDOR = os.path.join(os.path.dirname(__file__), "agent", "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
