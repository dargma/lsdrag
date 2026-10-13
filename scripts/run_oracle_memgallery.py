"""Separate Reader diagnostic using annotated clue turns, never gold answers."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import tiktoken

from lsdrag.agent import AgentConfig, Reader, image_message, parse_answer
from lsdrag.io import dump_json, read_jsonl, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit code before oracle execution")
    if a.output.exists():
        raise ValueError("Use a new output directory; preserve previous diagnostics")
    questions = read_jsonl(a.questions)
    gold = {r["qid"]: r for r in read_jsonl(a.gold)}
    chunks = {c["chunk_id"]: c for c in read_jsonl(a.corpus / "chunks.jsonl")}
    figures = {f["figure_id"]: f for f in read_jsonl(a.corpus / "figures.jsonl")}
    config = AgentConfig()
    reader = Reader(config)
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    prompt = "Answer the question using the supplied conversation evidence and images. Do not guess unsupported visual details. Return JSON with answers (a list of short strings), citations (a list of chunk IDs), and reasoning (a short explanation). Leave answers empty if the evidence is insufficient. For a visual search question, return source Image IDs as separate answers in ascending order."
    identity = {"arm": "Oracle", "benchmark": "memgallery", "questions_sha256": sha256(a.questions), "gold_sha256": sha256(a.gold), "question_ids": [q["qid"] for q in questions], "corpus_sha256": sha256(a.corpus / "manifest.json"), "config": asdict(config), "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "uses_gold_clues": True, "uses_gold_answers_in_reader_input": False, "diagnostic_only": True}
    dump_json(a.output / "run_manifest.json", identity)
    for q in questions:
        start = time.monotonic()
        clue_ids = [q["scenario"] + ":" + turn for turn in gold[q["qid"]]["clue"]]
        evidence = [chunks[key] for key in clue_ids]
        if any(c["scenario"] != q["scenario"] for c in evidence):
            raise ValueError("Cross-scenario oracle clue")
        texts = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in evidence]
        evidence_by_hash = {}
        for c in evidence:
            for fid in c["figure_ids"]:
                figure = figures[fid]
                item = evidence_by_hash.setdefault(figure["image_sha256"], {"image_path": figure["image_path"], "image_sha256": figure["image_sha256"], "figure_ids": []})
                item["figure_ids"].append(fid)
        evidence_images = list(evidence_by_hash.values())
        query_images = [{"image_path": path, "image_sha256": sha256(path), "figure_ids": ["question:" + q["qid"]]} for path in q["question_image_paths"]]
        images = query_images + evidence_images
        if len(images) > config.max_images:
            raise ValueError("Oracle evidence exceeds shared image budget; do not silently truncate")
        text = "Question: " + q["question"] + "\nTask type: " + q["task_type"] + "\nConversation evidence:\n" + json.dumps(texts, ensure_ascii=False)
        if len(tokenizer.encode(text + prompt)) * 1.5 + len(images) * (config.image_soft_tokens + 20) + config.max_output_tokens + 2000 > config.context_limit:
            raise ValueError("Oracle evidence exceeds shared context budget")
        user = image_message(images) if images else {"role": "user", "content": text}
        if images:
            user["content"][0]["text"] = text + "\nImages labeled question are part of the query; other labels identify the supplied evidence figures."
        choice, usage = reader.chat([{"role": "system", "content": prompt}, user], None)
        answer = choice["message"].get("content") or ""
        result = {"qid": q["qid"], "question": q["question"], "scenario": q["scenario"], "task_type": q["task_type"], "arm": "Oracle", "status": "truncated" if choice.get("finish_reason") == "length" else "ok", "answer_raw": answer, **parse_answer(answer), "loops": 1, "seconds": time.monotonic() - start, "search_tokens": 0, "evidence_tokens": len(tokenizer.encode(json.dumps(texts))), "unique_delivered_images": len(evidence_images), "question_image_count": len(query_images), "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0), "read_chunk_ids": clue_ids, "opened_figure_ids": [f for item in evidence_images for f in item["figure_ids"]], "delivered_figure_ids": [f for item in evidence_images for f in item["figure_ids"]], "trajectory": [], "config": asdict(config), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "diagnostic_only": True}
        dump_json(a.output / (q["qid"] + ".json"), result)
        print(json.dumps({"qid": q["qid"], "status": result["status"], "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
