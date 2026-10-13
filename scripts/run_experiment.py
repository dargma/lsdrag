"""Run a fixed question manifest without loading answers or evidence metadata."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

from lsdrag.agent import Agent, AgentConfig
from lsdrag.embeddings import Embedder
from lsdrag.io import dump_json, read_jsonl, sha256
from lsdrag.retrieval import Retrieval, VectorIndex


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["B0", "V1", "V2", "C1"], required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--indexes", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8102/v1")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-loops", type=int, default=15)
    a = p.parse_args()
    questions = read_jsonl(a.questions)
    if any(set(q) != {"qid", "question"} for q in questions):
        raise ValueError("Agent question manifest must contain only qid and question")
    if len({q["qid"] for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")
    if a.limit:
        questions = questions[:a.limit]
    channels = ["gemma_text"] if a.arm == "C1" else ["qwen_text"]
    if a.arm in {"V1", "V2"}:
        channels.append("gemma_image")
    config = AgentConfig(base_url=a.base_url, max_loops=a.max_loops)
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / ("visual.txt" if a.arm == "V2" else "baseline.txt")
    identity = {"arm": a.arm, "questions_sha256": sha256(a.questions), "question_ids": [q["qid"] for q in questions], "corpus_sha256": sha256(a.corpus / "manifest.json"), "index_manifests": {c: sha256(a.indexes / c / "manifest.json") for c in channels}, "prompt_sha256": sha256(prompt_path), "config": asdict(config), "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit code before running a tracked experiment")
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = a.output / "run_manifest.json"
    if manifest.exists():
        if not a.resume or json.loads(manifest.read_text()) != identity:
            raise ValueError("Existing run has different identity or resume not requested")
    else:
        dump_json(manifest, identity)
    pending = []
    for q in questions:
        result_path = a.output / (q["qid"] + ".json")
        if a.resume and result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous["status"] != "failed":
                continue
            # Preserve infrastructure failures before retrying the same question.
            archive = a.output / "failed_attempts"
            archive.mkdir(exist_ok=True)
            result_path.rename(archive / (q["qid"] + f".{time.time_ns()}.json"))
        pending.append(q)
    if not pending:
        print("All requested questions already recorded")
        return
    indexes = {c: VectorIndex(a.indexes / c, a.corpus / ("figures.jsonl" if c.endswith("image") else "chunks.jsonl")) for c in channels}
    embedders = {channel: Embedder(channel, a.cache_dir) for channel in sorted({c.split("_")[0] for c in channels})}
    retrieval = Retrieval(a.corpus, indexes, embedders, a.arm)
    agent = Agent(retrieval, prompt_path.read_text(), config)
    start = time.monotonic()
    for i, q in enumerate(pending):
        result = agent.run(q["qid"], q["question"])
        dump_json(a.output / (q["qid"] + ".json"), result)
        print(json.dumps({"completed": i + 1, "remaining": len(pending) - i - 1, "qid": q["qid"], "status": result["status"], "answers": result["answers"], "loops": result["loops"], "images": result["unique_delivered_images"], "seconds": result["seconds"], "run_seconds": time.monotonic() - start}), flush=True)
        if result["status"] == "failed":
            raise RuntimeError("Infrastructure failure recorded; inspect before continuing")


if __name__ == "__main__":
    main()
