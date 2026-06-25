"""다중 페르소나 병렬 수용 테스트 (10). 개발 전용(배포 skill에 미포함).

각 페르소나 = 독립 평가자(병렬). 구조화 리포트 → synthesizer 종합 판정.
여기 평가자는 결정적 정적 검사판(실제 운영에선 LLM 에이전트로 교체 가능).
한 페르소나가 죽어도 나머지는 진행(부분 결과).
"""
from __future__ import annotations

import concurrent.futures
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Report:
    persona: str
    verdict: str            # pass | warn | fail | died
    findings: List[str] = field(default_factory=list)
    evidence: str = ""
    fix_suggestion: str = ""


def _read(path: str) -> str:
    p = os.path.join(ROOT, path)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


# ── 페르소나 평가자 ──────────────────────────────────────────────
def installer() -> Report:
    txt = _read("INSTALL.md")
    steps = re.findall(r"^##\s*\d+단계", txt, re.M)
    has_checks = txt.count("doctor.py")
    ok = len(steps) >= 4 and has_checks >= 3
    return Report("installer", "pass" if ok else "warn",
                  [f"{len(steps)} numbered steps", f"{has_checks} doctor checkpoints"],
                  "INSTALL.md", "" if ok else "각 단계에 검증(doctor) 한 줄을 보강.")


def verifier() -> Report:
    test_dir = os.path.join(ROOT, "tests")
    tests = [f for f in os.listdir(test_dir) if f.startswith("test_")]
    ok = len(tests) >= 8
    return Report("verifier", "pass" if ok else "warn",
                  [f"{len(tests)} test modules"], "tests/",
                  "" if ok else "모듈별 회귀 테스트 보강.")


def user() -> Report:
    g = _read("tests/e2e/golden.py")
    types = set(re.findall(r'"(text|structure|image)"', g))
    ok = {"text", "structure", "image"} <= types
    return Report("user", "pass" if ok else "fail",
                  [f"golden types: {sorted(types)}"], "tests/e2e/golden.py",
                  "" if ok else "3유형 골든 질의 모두 포함.")


def user2() -> Report:
    """설치~사용 전반 직관성 (1급 축)."""
    install, readme, skill = _read("INSTALL.md"), _read("README.md"), _read("rag/SKILL.md")
    signals = {
        "quick_start": "Quick Start" in readme,
        "one_line_use": "/rag" in readme or "run.py" in skill,
        "human_errors": "조치" in install or "조치" in _read("rag/scripts/doctor.py"),
        "single_config": "config.yaml" in install,
    }
    score = sum(signals.values()) / len(signals)
    verdict = "pass" if score >= 0.75 else ("warn" if score >= 0.5 else "fail")
    return Report("user2", verdict,
                  [f"intuitiveness {score:.2f}", *(k for k, v in signals.items() if not v and 'missing:'+k)],
                  "INSTALL/README/SKILL", "" if score >= 0.75 else "막힘 지점·사람말 에러 보강.")


def leader() -> Report:
    """HW 설계 가치: 구조 질의 + 이미지 해석이 실제 있는가."""
    has_pi = os.path.exists(os.path.join(ROOT, "src/retrieval/page_index_search.py"))
    has_img = os.path.exists(os.path.join(ROOT, "src/retrieval/image_read.py"))
    ok = has_pi and has_img
    return Report("leader", "pass" if ok else "fail",
                  ["page_index_search " + ("✓" if has_pi else "✗"),
                   "image_read " + ("✓" if has_img else "✗"),
                   "가치: 레지스터 표/비트필드 figure 질의를 그래프 없이 커버"],
                  "src/retrieval/", "" if ok else "구조/이미지 tool 누락.")


def designer() -> Report:
    r = _read("README.md")
    signals = {"tagline": "<p align" in r or "agentic RAG" in r,
               "badges": "img.shields.io" in r,
               "sections": all(s in r for s in ("What is this?", "Usage", "How it works")),
               "diagram": "파싱" in r and "검색" in r}
    score = sum(signals.values()) / len(signals)
    return Report("designer", "pass" if score >= 0.75 else "warn",
                  [f"readability {score:.2f}", *(f"missing:{k}" for k, v in signals.items() if not v)],
                  "README.md", "" if score >= 0.75 else "README 섹션/배지/다이어그램 보강.")


PERSONAS: Dict[str, Callable[[], Report]] = {
    "installer": installer, "verifier": verifier, "user": user,
    "user2": user2, "leader": leader, "designer": designer,
}


def run_personas(names: List[str] = None) -> List[Report]:
    names = names or list(PERSONAS)
    reports: List[Report] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(names)) as ex:
        futs = {ex.submit(PERSONAS[n]): n for n in names}
        for fut in concurrent.futures.as_completed(futs):
            n = futs[fut]
            try:
                reports.append(fut.result())
            except Exception as e:  # 한 페르소나 사망 → 나머지 진행
                reports.append(Report(n, "died", [str(e)], "", "페르소나 평가자 수리 필요."))
    return reports


def synthesize(reports: List[Report]) -> dict:
    order = {"fail": 0, "died": 1, "warn": 2, "pass": 3}
    worst = min(reports, key=lambda r: order.get(r.verdict, 0)).verdict
    decision = {"fail": "거부", "died": "조건부", "warn": "조건부", "pass": "수용"}[worst]
    intuitiveness = next((r for r in reports if r.persona == "user2"), None)
    actions = [{"persona": r.persona, "fix": r.fix_suggestion}
               for r in reports if r.verdict in ("fail", "warn", "died") and r.fix_suggestion]
    return {
        "decision": decision,
        "intuitiveness": intuitiveness.findings[0] if intuitiveness else "n/a",  # 1급 축 필수
        "verdicts": {r.persona: r.verdict for r in reports},
        "actions": actions,
    }
