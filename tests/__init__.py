"""테스트 패키지. arag(vendor)를 직접 import하는 테스트가 있어, 패키지 로드 시점에
vendor 경로와 repo 루트를 sys.path에 올린다(테스트 본문 import보다 먼저 실행됨).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENDOR = os.path.join(_ROOT, "src", "agent", "vendor")
for _p in (_ROOT, _VENDOR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
