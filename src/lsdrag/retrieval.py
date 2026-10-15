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
        if self.manifest.get("format") == "composite-v1":
            sources, parts = [], []
            for spec in self.manifest["parts"]:
                part_dir, part_source = Path(spec["directory"]), Path(spec["source"])
                if sha256(part_dir / "manifest.json") != spec["manifest_sha256"]:
                    raise ValueError("Composite part manifest changed")
                part = VectorIndex(part_dir, part_source)
                for key in ("channel", "modality", "model", "revision", "dimension", "dtype", "normalization", "query_instruction"):
                    if part.manifest[key] != self.manifest[key]:
                        raise ValueError(f"Incompatible composite part: {key}")
                sources.extend(read_jsonl(part_source))
                parts.append(part)
            if sources != read_jsonl(source):
                raise ValueError("Composite sources must partition the corpus exactly in order")
            identity = "figure_id" if self.manifest["modality"] == "image" else "chunk_id"
            if len({r[identity] for r in sources}) != len(sources):
                raise ValueError("Duplicate composite source identities")
            self.rows = [r for part in parts for r in part.rows]
            self.vectors = np.concatenate([part.vectors for part in parts], axis=0)
            if len(self.rows) != self.manifest["rows"]:
                raise ValueError("Composite row count mismatch")
            return
        for name in ("rows.jsonl", "vectors.npy"):
            key = "rows_sha256" if name == "rows.jsonl" else "vectors_sha256"
            if sha256(directory / name) != self.manifest[key]:
                raise ValueError(f"Index integrity failure: {name}")
        self.rows = read_jsonl(directory / "rows.jsonl")
        self.vectors = np.load(directory / "vectors.npy", mmap_mode="r")
        if self.vectors.shape != (len(self.rows), self.manifest["dimension"]):
            raise ValueError("Index shape mismatch")

    def search(self, query_vector, limit, identity="chunk_id", allowed=None, row_limit=None):
        query = normalized_float32(np.asarray(query_vector).reshape(1, -1))[0]
        scores = self.vectors @ query
        # Stable order makes tied retrieval deterministic across experimental arms.
        order = np.argsort(-scores, kind="stable")
        if allowed is not None:
            order = [index for index in order if self.rows[int(index)][identity] in allowed]
        if row_limit is not None:
            order = order[:row_limit]
        output, seen = [], set()
        self.last_matches = {}
        for index in order:
            row = self.rows[int(index)]
            key = row[identity]
            if key in seen or (allowed is not None and key not in allowed):
                continue
            output.append((key, float(scores[index])))
            self.last_matches[key] = row.get("text")
            seen.add(key)
            if len(output) >= limit:
                break
        return output


class Retrieval:
    def __init__(self, corpus, indexes, embedders, arm, candidate_count=60, document_scope=None):
        if arm not in {"B0", "V1", "V2", "C1", "G1"}:
            raise ValueError(arm)
        self.arm = arm
        self.chunks = {r["chunk_id"]: r for r in read_jsonl(Path(corpus) / "chunks.jsonl")}
        self.figures = {r["figure_id"]: r for r in read_jsonl(Path(corpus) / "figures.jsonl")}
        self.scoped = document_scope is not None
        if self.scoped:
            scope = set(document_scope)
            self.chunks = {k: c for k, c in self.chunks.items() if scope.intersection(c["document_ids"])}
            self.figures = {k: f for k, f in self.figures.items() if f["chunk_id"] in self.chunks}
            if not self.chunks:
                raise ValueError("Empty question-specific corpus scope")
        self.indexes, self.embedders = indexes, embedders
        self.candidate_count = candidate_count
        self.query_cache = {}

    def query_vector(self, channel, query):
        key = (channel, query)
        if key not in self.query_cache:
            self.query_cache[key] = self.embedders[channel].encode([query], "query")[0]
        return self.query_cache[key]

    def snippet(self, chunk_id, matched=None):
        c = self.chunks[chunk_id]
        return {"chunk_id": chunk_id, "document_id": c["document_id"], "page_index": c["page_index"], "kind": c["kind"], "snippet": (matched or c["text"])[:300]}

    def keyword_search(self, keywords, top_k=5):
        if isinstance(keywords, str):
            keywords = [keywords]
        if not keywords or any(not isinstance(k, str) or not k.strip() for k in keywords):
            raise ValueError("keywords must be nonempty strings")
        terms = [k.lower() for k in keywords]
        scored = [(sum(c["text"].lower().count(t) * len(t) for t in terms), key) for key, c in self.chunks.items()]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        import re
        return [self.snippet(key, " ... ".join(s for s in re.split(r"[.!?\n]+", self.chunks[key]["text"]) if any(t in s.lower() for t in terms))) for score, key in scored[:max(1, min(20, top_k))] if score > 0]

    def semantic_search(self, query, top_k=5):
        top_k = max(1, min(20, top_k))
        channel = "gemma" if self.arm in {"C1", "G1"} else "qwen"
        # Use the same candidate budget across arms. G1 uses Google for
        # both text and image rankings; legacy V1/V2 retain Qwen text.
        count = self.candidate_count
        allowed_chunks = set(self.chunks) if self.scoped else None
        text = self.indexes[channel + "_text"].search(self.query_vector(channel, query), count, allowed=allowed_chunks, row_limit=count * 3)
        identities = [key for key, _ in text]
        if self.arm in {"V1", "V2", "G1"}:
            images = self.indexes["gemma_image"].search(self.query_vector("gemma", query), count, allowed=allowed_chunks)
            identities = reciprocal_rank_fusion([identities, [key for key, _ in images]])
        matches = self.indexes[channel + "_text"].last_matches
        return [self.snippet(key, matches.get(key)) for key in identities[:top_k]]

    def image_search(self, query, document_ids=None, top_k=5):
        if self.arm not in {"G1", "V2"}:
            raise ValueError("image_search requires a visual G1 or V2 runtime")
        allowed = set(self.figures) if self.scoped else None
        if document_ids:
            wanted = set(document_ids)
            allowed = {key for key, f in self.figures.items() if wanted.intersection(f["document_ids"])}
        results = self.indexes["gemma_image"].search(self.query_vector("gemma", query), max(1, min(20, top_k)), "figure_id", allowed)
        return [{**self.snippet(self.figures[key]["chunk_id"]), "figure_id": key} for key, _ in results]
