"""10_persona 게이트 — 6 페르소나 병렬 완주 + 종합 판정(직관성 포함) + 사망 격리."""
from tests.acceptance import personas as P


def test_six_personas_run_parallel():
    reports = P.run_personas()
    assert len(reports) == 6
    assert {r.persona for r in reports} == set(P.PERSONAS)


def test_synthesizer_includes_intuitiveness_and_decision():
    s = P.synthesize(P.run_personas())
    assert s["decision"] in ("수용", "조건부", "거부")
    assert "intuitiveness" in s and s["intuitiveness"] != "n/a"  # 1급 축 필수
    assert isinstance(s["actions"], list)


def test_dead_persona_isolated():
    # 한 페르소나가 죽어도 나머지 진행 + 종합은 근거 유지
    orig = P.PERSONAS["leader"]
    P.PERSONAS["leader"] = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        reports = P.run_personas()
        verds = {r.persona: r.verdict for r in reports}
        assert verds["leader"] == "died"
        assert verds["installer"] in ("pass", "warn")  # 나머지 진행
        s = P.synthesize(reports)
        assert s["decision"] in ("거부", "조건부", "수용")
    finally:
        P.PERSONAS["leader"] = orig


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    rep = P.run_personas()
    import json
    print(json.dumps(P.synthesize(rep), ensure_ascii=False, indent=2))
    print("[10_persona] 6 병렬 + 종합판정 + 사망격리 통과")
