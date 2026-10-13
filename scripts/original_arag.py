"""Dispatch to the unchanged, revision-pinned upstream text-only A-RAG scripts.

Arguments following -- are passed verbatim to the selected upstream script.
This path intentionally bypasses every lsdrag prompt, tool, fusion and image hook.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

REVISION = "a44de6b2216bf6791979c4b6ac4ae106212fa1a6"

p = argparse.ArgumentParser()
p.add_argument("--upstream", type=Path, required=True)
p.add_argument("--operation", choices=["build_index", "batch_runner"], required=True)
p.add_argument("arguments", nargs=argparse.REMAINDER)
a = p.parse_args()
root = a.upstream.resolve()
if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != REVISION:
    raise ValueError("Wrong upstream revision")
if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip():
    raise ValueError("Upstream tracked files have modifications")
arguments = a.arguments[1:] if a.arguments[:1] == ["--"] else a.arguments
environment = dict(os.environ)
environment.setdefault("HF_HOME", str(Path(environment.get("LSDRAG_WORK_ROOT", "/content/local_fast/lsdrag-work")) / "hf"))
environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
raise SystemExit(subprocess.call([sys.executable, str(root / "scripts" / (a.operation + ".py")), *arguments], env=environment))
