"""09_skill 게이트 — doctor 진단력(G1) + 정상 부분(G2) + 패키징 분리(G7).

doctor를 서브프로세스로 돌려 의도적 결함에서 정확한 단계에 멈추는지 검증.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCTOR = os.path.join(ROOT, "rag", "scripts", "doctor.py")


def _run(args, env_overrides, config_path=None):
    env = dict(os.environ)
    env.update(env_overrides)
    env["RAG_CONFIG"] = config_path or os.path.join(ROOT, "config.yaml")
    return subprocess.run([sys.executable, DOCTOR] + args,
                          cwd=ROOT, env=env, capture_output=True, text=True)


def test_g1_keys_check_fails_when_key_removed():
    # provider=openai 에서 OPENAI_API_KEY 제거 → keys 체크 fail. engine.root="." 유지 위해 ROOT에 임시 config.
    p = os.path.join(ROOT, ".test_openai_config.yaml")
    base = open(os.path.join(ROOT, "config.yaml"), encoding="utf-8").read()
    open(p, "w", encoding="utf-8").write(base.replace("provider: claude_code", "provider: openai"))
    try:
        out = _run(["--check", "keys", "--json"], {"OPENAI_API_KEY": ""}, p)
    finally:
        os.remove(p)
    lines = [l for l in out.stdout.strip().splitlines() if l.startswith("{")]
    rec = json.loads(lines[-1])
    assert rec["name"] == "keys" and rec["status"] == "fail"
    assert "OPENAI_API_KEY" in (rec["msg"] + rec.get("fix", ""))


def test_g1_index_check_fails_without_build():
    out = _run(["--check", "index", "--json"], {})
    lines = [l for l in out.stdout.strip().splitlines() if l.startswith("{")]
    rec = json.loads(lines[-1])
    # 인덱스 빌드 전이므로 fail + 빌드 안내
    assert rec["name"] == "index" and rec["status"] == "fail"
    assert "build" in rec["fix"]


def test_g2_deps_and_skill_pass():
    out = _run(["--check", "skill", "--check", "deps", "--json"], {})
    recs = [json.loads(l) for l in out.stdout.strip().splitlines() if l.startswith("{")]
    by = {r["name"]: r["status"] for r in recs}
    assert by.get("deps") == "ok"


def test_g7_packaging_no_dev_files():
    # 배포 대상 스킬 패밀리에 개발 전용 파일이 섞이지 않았는지(대원칙 5)
    forbidden = {"CLAUDE.md", "PROGRESS.md", "docs", "_MASTER.md", "tests", "examples", "src"}
    bad = ("CLAUDE.md", "PROGRESS.md", "_MASTER", "open('docs", 'open("docs', "/docs/0")
    for skill in ("rag", "rag-build", "rag-parse"):
        sdir = os.path.join(ROOT, skill)
        assert os.path.exists(os.path.join(sdir, "SKILL.md")), f"{skill}/SKILL.md missing"
        present = set(os.listdir(sdir))
        assert not (forbidden & present), f"dev files leaked into {skill}/: {forbidden & present}"
        for fn in os.listdir(os.path.join(sdir, "scripts")):
            if fn.endswith(".py"):
                txt = open(os.path.join(sdir, "scripts", fn), encoding="utf-8").read()
                hit = [b for b in bad if b in txt]
                assert not hit, f"{skill}/{fn} references dev artifact: {hit}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("[09_skill] doctor 진단력 + 정상부분 + 패키징분리 통과")
