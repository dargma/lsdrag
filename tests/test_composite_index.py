import json

import numpy as np
import pytest

from lsdrag.io import dump_json, sha256, write_jsonl
from lsdrag.retrieval import VectorIndex


def test_composite_global_ranking_and_source_integrity(tmp_path):
    common = dict(channel="gemma", modality="text", model="test", revision="fixed", dimension=2,
                  dtype="float32", normalization="L2", query_instruction="query")
    sources, parts = [], []
    for number, vector in enumerate(([0., 1.], [1., 0.])):
        directory = tmp_path / str(number)
        directory.mkdir()
        source = directory / "chunks.jsonl"
        rows = [{"chunk_id": str(number), "text": "x"}]
        sources.extend(rows)
        write_jsonl(source, rows)
        write_jsonl(directory / "rows.jsonl", rows)
        np.save(directory / "vectors.npy", np.array([vector], dtype=np.float32))
        dump_json(directory / "manifest.json", dict(common, rows=1, source_sha256=sha256(source),
                  rows_sha256=sha256(directory / "rows.jsonl"), vectors_sha256=sha256(directory / "vectors.npy")))
        parts.append(dict(directory=str(directory), source=str(source), manifest_sha256=sha256(directory / "manifest.json")))
    source = tmp_path / "all.jsonl"
    write_jsonl(source, sources)
    combined = tmp_path / "combined"
    combined.mkdir()
    manifest = dict(common, format="composite-v1", parts=parts, source_sha256=sha256(source), rows=2)
    dump_json(combined / "manifest.json", manifest)
    index = VectorIndex(combined, source)
    assert index.search([1., 0.], 1) == [("1", 1.)]
    assert index.search([1., 0.], 1, allowed={"0"}, row_limit=1) == [("0", 0.)]
    # Even if the parent source hash is updated, a changed corpus is not a valid partition.
    sources[0]["text"] = "changed"
    write_jsonl(source, sources)
    manifest["source_sha256"] = sha256(source)
    dump_json(combined / "manifest.json", manifest)
    with pytest.raises(ValueError, match="partition"):
        VectorIndex(combined, source)
