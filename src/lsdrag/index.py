from pathlib import Path
import re

import numpy as np
from PIL import Image

from .embeddings import Embedder, MODELS, GEMMA_QUERY_PREFIX, QWEN_INSTRUCTION
from .io import dump_json, read_jsonl, sha256, write_jsonl


def sentence_units(chunks):
    """A-RAG sentence-level retrieval, aggregated to chunk IDs at query time."""
    for chunk in chunks:
        text = chunk["text"]
        units = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        for number, sentence in enumerate(units):
            yield {"chunk_id": chunk["chunk_id"], "sentence_id": number, "text": sentence, "title": chunk["title"]}


def build_index(corpus, output, channel, modality, cache_dir, batch_size=16, device="cuda"):
    corpus, output = Path(corpus), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if modality == "image" and channel != "gemma":
        raise ValueError("Image indexes require EmbeddingGemma")
    source = corpus / ("figures.jsonl" if modality == "image" else "chunks.jsonl")
    rows = read_jsonl(source) if modality == "image" else list(sentence_units(read_jsonl(source)))
    if not rows:
        raise ValueError("Cannot index an empty corpus")
    embedder = Embedder(channel, cache_dir, device=device, batch_size=batch_size)
    target = output / "vectors.npy"
    temporary = output / "vectors.partial.npy"
    vectors = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(rows), MODELS[channel][2]))
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        if modality == "image":
            inputs = []
            try:
                for row in batch:
                    with Image.open(row["image_path"]) as image:
                        inputs.append(image.convert("RGB"))
                values = embedder.encode(inputs, "image")
            finally:
                for image in inputs:
                    image.close()
        else:
            values = embedder.encode([r["text"] for r in batch], "document", titles=[r["title"] for r in batch])
        vectors[start:start + len(batch)] = values
        if start % (batch_size * 20) == 0:
            print(f"{channel}/{modality}: {start + len(batch)}/{len(rows)}", flush=True)
    vectors.flush()
    del vectors
    temporary.replace(target)
    write_jsonl(output / "rows.jsonl", rows)
    manifest = {"channel": channel, "modality": modality, "model": MODELS[channel][0], "revision": MODELS[channel][1], "dimension": MODELS[channel][2], "rows": len(rows), "dtype": "float32", "normalization": "L2 after float32 conversion", "source_sha256": sha256(source), "vectors_sha256": sha256(target), "rows_sha256": sha256(output / "rows.jsonl"), "query_instruction": QWEN_INSTRUCTION if channel == "qwen" else GEMMA_QUERY_PREFIX}
    dump_json(output / "manifest.json", manifest)
    return manifest
