"""Download official MMQA metadata and freeze image-evidence question splits.

Only the private evaluation manifest contains gold answers/evidence. The agent
must consume questions.jsonl, never gold.jsonl or the selection metadata.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import urllib.request


BASE = "https://raw.githubusercontent.com/allenai/multimodalqa/master/dataset/"
FILES = ("MMQA_dev", "MMQA_images", "MMQA_texts", "MMQA_tables")


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def strata_key(q):
    modalities = sorted({c["doc_part"] for c in q["supporting_context"]})
    return "+".join(modalities) + ":" + q["metadata"]["type"]


def sample_stratified(rows, count, seed):
    """Proportional largest-remainder sampling, independent of answers/scores."""
    if not 0 <= count <= len(rows):
        raise ValueError(f"Cannot sample {count} from {len(rows)} candidates")
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["qid"]):
        groups[strata_key(row)].append(row)
    quotas = {k: count * len(v) / len(rows) for k, v in groups.items()}
    sizes = {k: int(v) for k, v in quotas.items()}
    for key in sorted(groups, key=lambda k: (-(quotas[k] - sizes[k]), k))[: count - sum(sizes.values())]:
        sizes[key] += 1
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        rng.shuffle(groups[key])
        selected.extend(groups[key][:sizes[key]])
    return sorted(selected, key=lambda r: r["qid"])


def build_splits(data_dir, output_dir, summary_path, seed=20261013):
    questions = read_jsonl(data_dir / "MMQA_dev.jsonl")
    if len({q["qid"] for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")
    candidates = [q for q in questions if any(c["doc_part"] == "image" for c in q["supporting_context"])]
    development = sample_stratified(candidates, 50, seed)
    dev_ids = {q["qid"] for q in development}
    evaluation = sample_stratified([q for q in candidates if q["qid"] not in dev_ids], 200, seed + 1)
    source_docs = {}
    for name in FILES[1:]:
        for row in read_jsonl(data_dir / f"{name}.jsonl"):
            if row["id"] in source_docs and source_docs[row["id"]]["url"] != row["url"]:
                raise ValueError(f"Ambiguous source document: {row['id']}")
            source_docs[row["id"]] = row
    all_doc_ids = sorted({c["doc_id"] for q in questions for c in q["supporting_context"]})
    missing = [i for i in all_doc_ids if i not in source_docs]
    if missing:
        raise ValueError(f"Missing source metadata for {len(missing)} documents")
    def docs_for(qs):
        return {c["doc_id"] for q in qs for c in q["supporting_context"]}
    development_docs = docs_for(development)
    evaluation_docs = docs_for(evaluation)
    distractors = sorted(set(all_doc_ids) - development_docs - evaluation_docs)
    random.Random(seed + 2).shuffle(distractors)
    dev_corpus_ids = sorted(development_docs | set(distractors[:100]))
    for name, subset in [("development", development), ("evaluation", evaluation)]:
        write_jsonl(output_dir / name / "gold.jsonl", subset)
        write_jsonl(output_dir / name / "questions.jsonl", [{"qid": q["qid"], "question": q["question"]} for q in subset])
    write_jsonl(output_dir / "corpus_full.jsonl", [{"document_id": i, "url": source_docs[i]["url"], "title": source_docs[i]["title"]} for i in all_doc_ids])
    write_jsonl(output_dir / "corpus_development.jsonl", [{"document_id": i, "url": source_docs[i]["url"], "title": source_docs[i]["title"]} for i in dev_corpus_ids])
    # Source image paths are original MMQA references, not inferred PDF crops.
    image_docs = {c["doc_id"] for q in development + evaluation for c in q["supporting_context"] if c["doc_part"] == "image"}
    write_jsonl(output_dir / "gold_image_sources.jsonl", [source_docs[i] for i in sorted(image_docs)])
    summary = {
        "seed": seed, "total_dev_questions": len(questions), "eligible_image_questions": len(candidates),
        "criterion": "supporting_context contains doc_part=image",
        "allocation": "proportional largest remainder by evidence modality combination and metadata.type",
        "development_count": len(development), "evaluation_count": len(evaluation),
        "full_corpus_document_ids": len(all_doc_ids),
        "full_corpus_unique_urls": len({source_docs[i]["url"] for i in all_doc_ids}),
        "development_corpus_document_ids": len(dev_corpus_ids),
        "question_id_overlap": len(dev_ids & {q["qid"] for q in evaluation}),
        "support_document_overlap": len(development_docs & evaluation_docs),
        "evaluation_questions_sharing_development_documents": sum(bool(docs_for([q]) & development_docs) for q in evaluation),
        "candidate_strata": dict(Counter(strata_key(q) for q in candidates)),
        "development_strata": dict(Counter(strata_key(q) for q in development)),
        "evaluation_strata": dict(Counter(strata_key(q) for q in evaluation)),
        "selection_status": "selected_before_model_evaluation; collection_QC_pending",
        "development_ids": [q["qid"] for q in development],
        "evaluation_ids": [q["qid"] for q in evaluation],
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data_dir.glob("MMQA_*.jsonl"))},
        "output_sha256": {str(p.relative_to(output_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output_dir.rglob("*.jsonl"))},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = args.data_dir / f"{name}.jsonl"
        if not path.exists():
            with urllib.request.urlopen(BASE + name + ".jsonl.gz", timeout=120) as response:
                content = gzip.decompress(response.read())
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
    summary = build_splits(args.data_dir, args.output_dir, args.summary)
    print(json.dumps({k: summary[k] for k in ["eligible_image_questions", "development_count", "evaluation_count", "full_corpus_document_ids", "support_document_overlap"]}))


if __name__ == "__main__":
    main()
