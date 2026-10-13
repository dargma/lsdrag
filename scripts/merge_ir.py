"""Merge disjoint immutable IR shards without changing chunk or figure identity."""
import argparse
from collections import Counter
import json
from pathlib import Path

from lsdrag.io import dump_json, read_jsonl, sha256, write_jsonl

p = argparse.ArgumentParser()
p.add_argument("--part", type=Path, action="append", required=True)
p.add_argument("--corpus-manifest", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise ValueError("Refusing to overwrite frozen IR")
manifests = [json.loads((part / "manifest.json").read_text()) for part in a.part]
policies = ("max_words", "page_index_base", "table_policy", "page_boundary_policy", "figure_description_policy")
for key in policies:
    if any(m[key] != manifests[0][key] for m in manifests):
        raise ValueError(f"IR policies differ: {key}")
merged = {}
for name, identity in (("chunks", "chunk_id"), ("figures", "figure_id"), ("documents", "document_id")):
    rows = [r for part in a.part for r in read_jsonl(part / (name + ".jsonl"))]
    if len({r[identity] for r in rows}) != len(rows):
        raise ValueError(f"Overlapping {name}")
    merged[name] = rows
docs, chunks, figures = merged["documents"], merged["chunks"], merged["figures"]
if len({d["url"] for d in docs}) != len(docs):
    raise ValueError("Overlapping document URLs")
wanted = {d["url"] for d in read_jsonl(a.corpus_manifest)}
available = {d["url"] for d in docs}
if available - wanted:
    raise ValueError("IR contains documents outside frozen corpus")
a.output.mkdir(parents=True)
for name, rows in merged.items():
    write_jsonl(a.output / (name + ".jsonl"), rows)
manifest = {key: manifests[0][key] for key in policies}
manifest.update(documents=len(docs), document_aliases=sum(len(d["document_ids"]) for d in docs),
    pages=sum(d["pages"] for d in docs), chunks=len(chunks), figures=len(figures),
    chunk_types=dict(Counter(c["kind"] for c in chunks)),
    figures_without_caption=sum(not f["has_caption"] for f in figures),
    figures_without_any_description=sum(not f["has_description"] for f in figures),
    chunks_sha256=sha256(a.output / "chunks.jsonl"), figures_sha256=sha256(a.output / "figures.jsonl"),
    corpus_manifest_sha256=sha256(a.corpus_manifest), missing_urls=sorted(wanted - available),
    parts=[{"directory": str(part.resolve()), "manifest_sha256": sha256(part / "manifest.json")} for part in a.part])
dump_json(a.output / "manifest.json", manifest)
print(json.dumps(manifest))
