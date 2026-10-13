"""Progressively disclosed A-RAG retrieval; channel scores never mix."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .embeddings import normalized_float32
from .io import read_jsonl, sha256


def reciprocal_rank_fusion(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for rank, identity in enumerate(ranking, 1):
            if identity not in seen:
                scores[identity] += 1.0 / (k + rank)
                seen.add(identity)
    return sorted(scores, key=lambda identity: (-scores[identity], identity))


class VectorIndex:
    def __init__(self, directory, source):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if self.manifest["source_sha256"] != sha256(source):
            raise ValueError("Index does not match corpus")
        for name in ("rows.jsonl", "vectors.npy"):
            key = "rows_sha256" if name == "rows.jsonl" else "vectors_sha256"
            if sha256(directory / name) != self.manifest[key]:
                raise ValueError(f"Index integrity failure: {name}")
        self.rows = read_jsonl(directory / "rows.jsonl")
        self.vectors = np.load(directory / "vectors.npy", mmap_mode="r")
        if self.vectors.shape != (len(self.rows), self.manifest["dimension"]):
            raise ValueError("Index shape mismatch")

    def search(self, query_vector, limit, identity="chunk_id", allowed=None):
        query = normalized_float32(np.asarray(query_vector).reshape(1, -1))[0]
        scores = self.vectors @ query
        # Stable order makes tied retrieval deterministic across experimental arms.
        order = np.argsort(-scores, kind="stable")
        output, seen = [], set()
        for index in order:
            row = self.rows[int(index)]
            key = row[identity]
            if key in seen or (allowed is not None and key not in allowed):
                continue
            output.append((key, float(scores[index])))
            seen.add(key)
            if len(output) >= limit:
                break
        return output


class Retrieval:
    def __init__(self, corpus, indexes, embedders, arm, candidate_count=60):
        if arm not in {"B0", "V1", "V2", "C1"}:
            raise ValueError(arm)
        self.arm = arm
        self.chunks = {r["chunk_id"]: r for r in read_jsonl(Path(corpus) / "chunks.jsonl")}
        self.figures = {r["figure_id"]: r for r in read_jsonl(Path(corpus) / "figures.jsonl")}
        self.indexes, self.embedders = indexes, embedders
        self.candidate_count = candidate_count
        self.query_cache = {}

    def query_vector(self, channel, query):
        key = (channel, query)
        if key not in self.query_cache:
            self.query_cache[key] = self.embedders[channel].encode([query], "query")[0]
        return self.query_cache[key]

    def snippet(self, chunk_id):
        c = self.chunks[chunk_id]
        return {"chunk_id": chunk_id, "document_id": c["document_id"], "page_index": c["page_index"], "kind": c["kind"], "snippet": c["text"][:300]}

    def keyword_search(self, keywords, top_k=20):
        if isinstance(keywords, str):
            keywords = [keywords]
        if not keywords or any(not isinstance(k, str) or not k.strip() for k in keywords):
            raise ValueError("keywords must be nonempty strings")
        terms = [k.lower() for k in keywords]
        scored = [(sum(c["text"].lower().count(t) * len(t) for t in terms), key) for key, c in self.chunks.items()]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [self.snippet(key) for score, key in scored[:max(1, min(20, top_k))] if score > 0]

    def semantic_search(self, query, top_k=5):
        top_k = max(1, min(20, top_k))
        channel = "gemma" if self.arm == "C1" else "qwen"
        count = self.candidate_count if self.arm in {"V1", "V2"} else top_k
        text = self.indexes[channel + "_text"].search(self.query_vector(channel, query), count)
        identities = [key for key, _ in text]
        if self.arm in {"V1", "V2"}:
            images = self.indexes["gemma_image"].search(self.query_vector("gemma", query), count)
            identities = reciprocal_rank_fusion([identities, [key for key, _ in images]])
        return [self.snippet(key) for key in identities[:top_k]]

    def image_search(self, query, document_ids=None, top_k=5):
        if self.arm != "V2":
            raise ValueError("image_search is only exposed in V2")
        allowed = None
        if document_ids:
            wanted = set(document_ids)
            allowed = {key for key, f in self.figures.items() if wanted.intersection(f["document_ids"])}
        results = self.indexes["gemma_image"].search(self.query_vector("gemma", query), max(1, min(20, top_k)), "figure_id", allowed)
        return [{**self.snippet(self.figures[key]["chunk_id"]), "figure_id": key} for key, _ in results]
