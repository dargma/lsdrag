"""Freeze VS/VR selection, fetch referenced images, and build common memory IR."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random

import requests

from lsdrag.io import dump_json, sha256, write_jsonl
from lsdrag.memgallery import convert_scenario, image_reference, save_corpus, select_questions

REVISION = "af912daba984e896e253016b7c7e334ef92c2a6f"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261013)
    a = p.parse_args()
    source = json.loads((a.root / "source_manifest.json").read_text())
    if source["revision"] != REVISION:
        raise ValueError("Unexpected source revision")
    files = sorted((a.root / "data/dialog").glob("*.json"))
    data = {path.stem: json.loads(path.read_text()) for path in files}
    scenarios = sorted(data)
    random.Random(a.seed).shuffle(scenarios)
    dev_scenarios = set(scenarios[:2])
    rows = []
    for scenario, item in sorted(data.items()):
        turns = {t["round"] for s in item["multi_session_dialogues"] for t in s["dialogues"]}
        for index, qa in enumerate(item["human-annotated QAs"]):
            if qa["point"] not in {"VS", "VR"}:
                continue
            if not qa.get("clue") or not set(qa["clue"]) <= turns:
                raise ValueError("Invalid gold clue references")
            rows.append({**qa, "qid": f"{scenario}:qa{index:04d}", "scenario": scenario, "source_qa_index": index})
    selections = {}
    for split, count in (("development", 5), ("evaluation", 25)):
        subset = []
        for offset, category in enumerate(("VS", "VR")):
            candidates = [r for r in rows if r["point"] == category and ((r["scenario"] in dev_scenarios) == (split == "development"))]
            subset.extend(select_questions(candidates, count, a.seed + offset + (10 if split == "evaluation" else 0)))
        selections[split] = sorted(subset, key=lambda r: r["qid"])
    selected_scenarios = {r["scenario"] for subset in selections.values() for r in subset}
    for split, subset in selections.items():
        write_jsonl(a.output / split / "gold.jsonl", subset)
        inputs = []
        for row in subset:
            images = [str((a.root / image_reference(row["question_image"])).resolve())] if row.get("question_image") else []
            inputs.append({"qid": row["qid"], "question": row["question"], "scenario": row["scenario"], "task_type": row["point"], "question_image_paths": images})
        write_jsonl(a.output / split / "questions.jsonl", inputs)
    requested = set()
    for scenario in selected_scenarios:
        for session in data[scenario]["multi_session_dialogues"]:
            for turn in session["dialogues"]:
                requested.update(image_reference(reference) for reference in turn.get("input_image", []))
    for subset in selections.values():
        requested.update(image_reference(q["question_image"]) for q in subset if q.get("question_image"))
    available = set(source["available_files"])
    if any(str(path) not in available for path in requested):
        raise ValueError("An image reference is absent from the pinned dataset")
    def fetch(relative):
        target = a.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            response = requests.get(f"https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery/resolve/{REVISION}/{relative.as_posix()}", timeout=90)
            response.raise_for_status()
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(response.content)
            temporary.replace(target)
        return {"path": str(relative), "bytes": target.stat().st_size, "sha256": sha256(target)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        downloads = list(pool.map(fetch, sorted(requested)))
    write_jsonl(a.output / "image_manifest.jsonl", downloads)
    chunks, figures, docs = [], [], []
    for scenario in sorted(selected_scenarios):
        c, f, d = convert_scenario(scenario, data[scenario], a.root)
        chunks.extend(c); figures.extend(f); docs.extend(d)
    corpus = save_corpus(a.output / "corpus", chunks, figures, docs, REVISION)
    summary = {"source_revision": REVISION, "seed": a.seed, "eligible_counts": dict(Counter(r["point"] for r in rows)), "development_scenarios": sorted(dev_scenarios), "scenario_overlap": 0, "selection_policy": "two seeded scenario holdouts for development; category quotas then proportional scenario/question-image strata; no model outcomes", "splits": {split: {"n": len(subset), "by_task": dict(Counter(q["point"] for q in subset)), "with_question_image": sum(bool(q.get("question_image")) for q in subset), "question_ids": [q["qid"] for q in subset], "questions_sha256": sha256(a.output / split / "questions.jsonl"), "gold_sha256": sha256(a.output / split / "gold.jsonl")} for split, subset in selections.items()}, "images": len(downloads), "image_bytes": sum(d["bytes"] for d in downloads), "corpus": corpus, "status": "selected_and_collected_before_model_evaluation"}
    dump_json(a.summary, summary)
    print(json.dumps({"splits": {k: len(v) for k, v in selections.items()}, "images": len(downloads), "corpus": corpus}))


if __name__ == "__main__":
    main()
