"""Reference immutable local index shards without duplicating vector files."""
import argparse
import json
from pathlib import Path

from lsdrag.io import dump_json, read_jsonl, sha256
from lsdrag.retrieval import VectorIndex

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--part", nargs=2, action="append", metavar=("INDEX", "SOURCE"), required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise ValueError("Refusing to overwrite a composite index")
parts = [{"directory": str(Path(d).resolve()), "source": str(Path(s).resolve()),
          "manifest_sha256": sha256(Path(d) / "manifest.json")} for d, s in a.part]
manifests = [json.loads((Path(s["directory"]) / "manifest.json").read_text()) for s in parts]
keys = ("channel", "modality", "model", "revision", "dimension", "dtype", "normalization", "query_instruction")
manifest = {key: manifests[0][key] for key in keys}
manifest.update(format="composite-v1", parts=parts, source_sha256=sha256(a.source), rows=sum(m["rows"] for m in manifests))
a.output.mkdir(parents=True)
dump_json(a.output / "manifest.json", manifest)
try:
    index = VectorIndex(a.output, a.source)
except Exception:
    (a.output / "manifest.json").unlink()
    a.output.rmdir()
    raise
print(json.dumps({"rows": len(index.rows), "dimension": index.manifest["dimension"], "parts": len(parts)}))
