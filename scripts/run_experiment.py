"""Run a fixed question manifest without loading answers or evidence metadata."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

from lsdrag.agent import Agent, AgentConfig
from lsdrag.embeddings import Embedder
from lsdrag.io import dump_json, read_jsonl, sha256
from lsdrag.retrieval import Retrieval, VectorIndex
from lsdrag.tool_registry import action_names, synchronize_prompt, tool_contract_sha256, tool_schemas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["B0", "V1", "V2", "C1", "G1"], required=True)
    p.add_argument("--benchmark", choices=["m3docvqa", "memgallery", "multihoprag"], default="m3docvqa")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--indexes", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8102/v1")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-loops", type=int, default=8)
    p.add_argument("--question-image-policy", choices=["text-only", "original"], default="text-only")
    p.add_argument("--image-delivery-policy", choices=["agent-read", "search-linked"], default="search-linked")
    p.add_argument("--tool-profile", choices=["legacy", "multimodal_full"], default="legacy")
    p.add_argument("--reader-mode", choices=["generated", "direct-head"], default="generated")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--head", type=Path)
    a = p.parse_args()
    if a.reader_mode == "direct-head" and (a.tool_profile != "multimodal_full" or a.arm != "G1" or not a.head or not a.model_path):
        p.error("direct-head requires --arm G1 --tool-profile multimodal_full --model-path MODEL --head NEW_SIX_ACTION_CHECKPOINT")
    schemas = tool_schemas(a.arm, a.image_delivery_policy, a.tool_profile)
    questions = read_jsonl(a.questions)
    allowed = {"qid", "question"} if a.benchmark != "memgallery" else {"qid", "question", "scenario", "task_type", "question_image_paths"}
    if any(set(q) != allowed for q in questions):
        raise ValueError("Question manifest must contain only the approved public input fields")
    if len({q["qid"] for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")
    if a.limit:
        questions = questions[:a.limit]
    if a.benchmark == "memgallery" and a.question_image_policy == "text-only":
        questions = [{**q, "question_image_paths": []} for q in questions]
    channels = ["gemma_text"] if a.arm in {"C1", "G1"} else ["qwen_text"]
    if a.arm in {"V1", "V2", "G1"}:
        channels.append("gemma_image")
    config = AgentConfig(base_url=a.base_url, max_loops=a.max_loops, image_delivery_policy=a.image_delivery_policy, tool_profile=a.tool_profile)
    prompt_name = "visual.txt" if a.arm == "V2" else ("search_linked.txt" if a.image_delivery_policy == "search-linked" else "baseline.txt")
    if a.tool_profile == "multimodal_full":
        prompt_name = "search_linked.txt" if a.image_delivery_policy == "search-linked" else "visual.txt"
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / prompt_name
    effective_prompt = synchronize_prompt(prompt_path.read_text(), schemas, a.image_delivery_policy) if a.tool_profile == "multimodal_full" else prompt_path.read_text()
    question_image_hashes = {path: sha256(path) for q in questions for path in q.get("question_image_paths", [])}
    identity = {"arm": a.arm, "question_image_policy": a.question_image_policy, "iteration_budget_semantics": "A-RAG agent iterations; forced final answer outside iteration budget; no independent Reader-call limit", "benchmark": a.benchmark, "questions_sha256": sha256(a.questions), "question_ids": [q["qid"] for q in questions], "question_image_hashes": question_image_hashes, "corpus_sha256": sha256(a.corpus / "manifest.json"), "index_manifests": {c: sha256(a.indexes / c / "manifest.json") for c in channels}, "prompt_sha256": sha256(prompt_path), "config": asdict(config), "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    identity.update(reader_mode=a.reader_mode, tool_contract_sha256=tool_contract_sha256(schemas),
                    actions=list(action_names(schemas)), effective_prompt_sha256=hashlib.sha256(effective_prompt.encode()).hexdigest(),
                    head_sha256=sha256(a.head) if a.head else None, model_path=str(a.model_path.resolve()) if a.model_path else None)
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
    reader = None
    if a.reader_mode == "direct-head":
        from lsdrag.direct_reader import DirectHFReader
        reader = DirectHFReader(str(a.model_path), config, a.head, schemas)
    agent = Agent(retrieval, prompt_path.read_text(), config, reader)
    documents = read_jsonl(a.corpus / "documents.jsonl") if a.benchmark == "memgallery" else []
    start = time.monotonic()
    for i, q in enumerate(pending):
        images = []
        if a.benchmark == "memgallery":
            scope = [d["document_id"] for d in documents if d["scenario"] == q["scenario"]]
            retrieval = Retrieval(a.corpus, indexes, embedders, a.arm, document_scope=scope)
            prompt = prompt_path.read_text() + "\nYou are answering a question about one user's stored conversations. The searchable corpus includes all recorded sessions for this scenario. Dates, session IDs and dialogue IDs are preserved. Use adjacent chunk IDs to inspect surrounding turns. Do not confuse the question image with retrieved memory evidence.\n"
            if q["task_type"] == "VS":
                prompt += "For this visual search task, return the source image_id(s), such as D2:IMG_003, as separate answers sorted in ascending order. Use the Image ID in the read text, not the internal figure ID.\n"
            agent = Agent(retrieval, prompt, config, reader)
            images = [{"image_path": path, "image_sha256": question_image_hashes[path], "figure_ids": ["question:" + q["qid"]]} for path in q["question_image_paths"]]
        result = agent.run(q["qid"], q["question"], images)
        if a.benchmark == "memgallery":
            result.update(scenario=q["scenario"], task_type=q["task_type"])
        dump_json(a.output / (q["qid"] + ".json"), result)
        print(json.dumps({"completed": i + 1, "remaining": len(pending) - i - 1, "qid": q["qid"], "status": result["status"], "answers": result["answers"], "loops": result["loops"], "images": result["unique_delivered_images"], "seconds": result["seconds"], "run_seconds": time.monotonic() - start}), flush=True)
        if result["status"] == "failed":
            raise RuntimeError("Infrastructure failure recorded; inspect before continuing")


if __name__ == "__main__":
    main()
