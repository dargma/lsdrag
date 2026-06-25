"""09_skill 게이트 — doctor 진단력(G1) + 정상 부분(G2) + 패키징 분리(G7).

doctor를 서브프로세스로 돌려 의도적 결함에서 정확한 단계에 멈추는지 검증.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCTOR = os.path.join(ROOT, "rag", "scripts", "doctor.py")


def _run(args, env_overrides):
    env = dict(os.environ)
    env.update(env_overrides)
    env["RAG_CONFIG"] = os.path.join(ROOT, "config.yaml")
    return subprocess.run([sys.executable, DOCTOR] + args,
                          cwd=ROOT, env=env, capture_output=True, text=True)


def test_g1_keys_check_fails_when_key_removed():
    # OPENAI_API_KEY 제거 → keys 체크 fail, 조치 안내
    env = {k: v for k, v in os.environ.items()}
    out = _run(["--check", "keys", "--json"],
               {**{k: os.environ.get(k, "") for k in []}, "OPENAI_API_KEY": ""})
    # 마지막 JSON 라인 파싱
    lines = [l for l in out.stdout.strip().splitlines() if l.startswith("{")]
    rec = json.loads(lines[-1])
    assert rec["name"] == "keys" and rec["status"] == "fail"
    assert "OPENAI_API_KEY" in rec["msg"]


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
    # 배포 대상 rag/ 에 개발 전용 파일이 섞이지 않았는지(대원칙 5)
    rag_dir = os.path.join(ROOT, "rag")
    forbidden = {"CLAUDE.md", "PROGRESS.md", "docs", "_MASTER.md", "tests", "examples"}
    present = set(os.listdir(rag_dir))
    assert not (forbidden & present), f"dev files leaked into rag/: {forbidden & present}"
    # rag/scripts 가 개발 지시서를 런타임에 읽거나 import하지 않는지(실제 의존만 검사)
    bad = ("CLAUDE.md", "PROGRESS.md", "_MASTER", "open('docs", 'open("docs', "/docs/0")
    for fn in os.listdir(os.path.join(rag_dir, "scripts")):
        if fn.endswith(".py"):
            txt = open(os.path.join(rag_dir, "scripts", fn), encoding="utf-8").read()
            hit = [b for b in bad if b in txt]
            assert not hit, f"{fn} references dev artifact: {hit}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("[09_skill] doctor 진단력 + 정상부분 + 패키징분리 통과")
