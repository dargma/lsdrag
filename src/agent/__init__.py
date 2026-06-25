"""Agent 단계 (07). vendored A-RAG를 import 가능하게 path 부트스트랩.

vendor 코드는 `from arag... import` (절대 import)를 쓰므로, 그 패키지 루트
(src/agent/vendor)를 sys.path에 올린다. vendor를 직접 수정하지 않기 위함.
"""
import os
import sys

_VENDOR = os.path.join(os.path.dirname(__file__), "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
