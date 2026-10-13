"""Freeze a resource-limited corpus with every selected support URL and fixed distractors."""
import argparse
from collections import defaultdict
from pathlib import Path
import random
import shutil

from lsdrag.io import dump_json, read_jsonl, sha256, write_jsonl

p = argparse.ArgumentParser()
p.add_argument("--full", type=Path, required=True)
p.add_argument("--existing-documents", type=Path, required=True)
p.add_argument("--support", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--summary", type=Path, required=True)
p.add_argument("--count", type=int, default=288)
p.add_argument("--seed", type=int, default=20261015)
a = p.parse_args()
if a.output.exists():
    raise ValueError("Refusing to overwrite a corpus selection")
all_rows = read_jsonl(a.full)
by_url = defaultdict(list)
for row in all_rows:
    by_url[row["url"]].append(row)
existing = {d["url"] for d in read_jsonl(a.existing_documents)}
support = {d["url"] for d in read_jsonl(a.support)}
selected = existing | support
if len(selected) > a.count:
    raise ValueError("Requested corpus size cannot contain existing and support documents")
candidates = sorted(set(by_url) - selected)
random.Random(a.seed).shuffle(candidates)
selected.update(candidates[:a.count - len(selected)])
if len(selected) != a.count:
    raise ValueError("Insufficient corpus candidates")
write_jsonl(a.output / "corpus.jsonl", [r for r in all_rows if r["url"] in selected])
new_urls = selected - existing
new_rows = [min(by_url[url], key=lambda r: r["document_id"]) for url in sorted(new_urls)]
write_jsonl(a.output / "render_new.jsonl", new_rows)
write_jsonl(a.output / "new_aliases.jsonl", [r for r in all_rows if r["url"] in new_urls])
dump_json(a.summary, {"seed": a.seed, "unique_urls": len(selected), "support_urls": len(support), "distractor_urls": len(selected - support), "reused_development_urls": len(existing), "new_urls_to_render": len(new_urls), "source_unique_urls": len(by_url), "policy": "all 83 selected support URLs + frozen development corpus as distractors + seeded random extra URLs; no model outcome selection", "limitation": "reduced retrieval corpus; not directly comparable to full-open-domain M3DocVQA", "resource_reason": "Full-corpus float32 text indexes extrapolate to about 36 GiB from development; current local free disk is under 4 GiB. Reuse development index shards to avoid copying vectors.", "local_free_bytes_at_selection": shutil.disk_usage(a.output.parent).free, "full_manifest_sha256": sha256(a.full), "support_manifest_sha256": sha256(a.support), "corpus_sha256": sha256(a.output / "corpus.jsonl"), "new_render_sha256": sha256(a.output / "render_new.jsonl")})
print(f"Frozen {len(selected)} URLs: {len(support)} support, {len(selected-support)} distractors, {len(new_urls)} new downloads")
