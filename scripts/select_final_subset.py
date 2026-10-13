"""Apply the approved final-size reduction without changing development data."""
import argparse
from collections import Counter
from pathlib import Path

from prepare_metadata import sample_stratified, strata_key
from lsdrag.io import dump_json, read_jsonl, sha256, write_jsonl

p = argparse.ArgumentParser()
p.add_argument("--candidates", type=Path, required=True)
p.add_argument("--development", type=Path, required=True)
p.add_argument("--corpus", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--summary", type=Path, required=True)
p.add_argument("--count", type=int, default=50)
p.add_argument("--seed", type=int, default=20261014)
a = p.parse_args()
if a.output.exists():
    raise ValueError("Refusing to overwrite an existing final selection")
candidates, development = read_jsonl(a.candidates), read_jsonl(a.development)
selected = sample_stratified(candidates, a.count, a.seed)
corpus = {r["document_id"]: r for r in read_jsonl(a.corpus)}
dev_ids = {r["qid"] for r in development}
dev_urls = {corpus[c["doc_id"]]["url"] for r in development for c in r["supporting_context"]}
gold_ids = {c["doc_id"] for r in selected for c in r["supporting_context"]}
assert not dev_ids.intersection(r["qid"] for r in selected)
assert not dev_urls.intersection(corpus[i]["url"] for i in gold_ids)
write_jsonl(a.output / "gold.jsonl", selected)
write_jsonl(a.output / "questions.jsonl", [{"qid": r["qid"], "question": r["question"]} for r in selected])
write_jsonl(a.output / "support_documents_for_collection_only.jsonl", [corpus[i] for i in sorted(gold_ids)])
dump_json(a.summary, {"count": len(selected), "seed": a.seed, "candidate_count": len(candidates), "source": "Previously selected 200 candidates, sampled without model outcomes", "candidate_sha256": sha256(a.candidates), "development_sha256": sha256(a.development), "question_ids": [r["qid"] for r in selected], "strata": dict(Counter(strata_key(r) for r in selected)), "support_document_ids": len(gold_ids), "support_unique_urls": len({corpus[i]["url"] for i in gold_ids}), "question_overlap": 0, "support_url_overlap": 0, "questions_sha256": sha256(a.output / "questions.jsonl"), "gold_sha256": sha256(a.output / "gold.jsonl"), "status": "selected_before_final_evaluation; collection_QC_pending"})
print(f"Selected {len(selected)} final questions; development and prior candidate manifests preserved")
