"""DB 구축 — IR → A-RAG 호환 인덱스 + Page Index + 매니페스트 (05).

A-RAG 도구가 읽는 형식과 호환(검증된 사실):
- chunks.json          : List[{'id': chunk_id, 'text': str}]   (read_chunk/keyword_search/semantic)
- index/sentence_index.pkl : {'sentences','embeddings','sentence_to_chunk','chunks'}  (semantic_search)
- page_index/page_index.json : 구조 인덱스 (04)
- manifest.json        : doc_id → chunk_ids/image_paths  (증분 remove용)

임베더는 주입형(embed_fn). 실제는 A-RAG 임베더(sentence-transformers), 테스트는 가짜 주입.
"""
from __future__ import annotations

import json
import os
import pickle
import re
from typing import Callable, Dict, List, Optional

import numpy as np

from src.page_index import PageIndex
from src.schema import ParsedDoc

EmbedFn = Callable[[List[str]], "np.ndarray"]  # texts -> (N, dim) normalized


def default_embedder(model_name: str) -> EmbedFn:
    """A-RAG와 동일 임베더(sentence-transformers). query/index 벡터공간 일치."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    def _embed(texts: List[str]) -> "np.ndarray":
        return np.asarray(model.encode(texts, normalize_embeddings=True))
    return _embed


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class IndexStore:
    """chunks + 벡터 인덱스 + Page Index + manifest. add/remove는 doc_id 단위(증분)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.chunks: Dict[str, str] = {}                 # chunk_id -> text
        self.page_index = PageIndex()
        self.manifest: Dict[str, Dict] = {}              # doc_id -> {chunk_ids, image_paths}
        # 문장 레벨
        self.sentences: List[str] = []
        self.sentence_to_chunk: List[str] = []
        self.embeddings: Optional["np.ndarray"] = None

    # ── 증분 ───────────────────────────────────────────────────
    def add_doc(self, doc: ParsedDoc, embed_fn: EmbedFn) -> None:
        if doc.doc_id in self.manifest:
            self.remove_doc(doc.doc_id)  # 멱등: 갱신
        chunk_ids, image_paths, new_sents, new_links = [], [], [], []
        for b in doc.blocks:
            cid = b.chunk_id or f"{doc.doc_id}:{len(self.chunks)}"
            if not b.text:
                continue
            self.chunks[cid] = b.text
            chunk_ids.append(cid)
            if b.image_path:
                image_paths.append(b.image_path)
            for s in split_sentences(b.text):
                new_sents.append(s)
                new_links.append(cid)
        self.page_index.add_doc(doc)
        self.manifest[doc.doc_id] = {"chunk_ids": chunk_ids, "image_paths": image_paths}

        if new_sents:
            new_emb = embed_fn(new_sents)
            self.sentences.extend(new_sents)
            self.sentence_to_chunk.extend(new_links)
            self.embeddings = new_emb if self.embeddings is None else np.vstack([self.embeddings, new_emb])

    def remove_doc(self, doc_id: str) -> int:
        if doc_id not in self.manifest:
            return 0
        cids = set(self.manifest[doc_id]["chunk_ids"])
        for cid in cids:
            self.chunks.pop(cid, None)
        # 문장/임베딩에서 해당 chunk 제거
        keep = [i for i, c in enumerate(self.sentence_to_chunk) if c not in cids]
        self.sentences = [self.sentences[i] for i in keep]
        self.sentence_to_chunk = [self.sentence_to_chunk[i] for i in keep]
        self.embeddings = self.embeddings[keep] if self.embeddings is not None and keep else (
            None if not keep else self.embeddings)
        if not keep:
            self.embeddings = None
        self.page_index.remove_doc(doc_id)
        del self.manifest[doc_id]
        return len(cids)

    def doc_ids(self) -> List[str]:
        return sorted(self.manifest)

    def status(self) -> Dict[str, Any]:
        """문서 관리 현황 — doc별 chunk·figure·table·page 범위 + 전체 합계."""
        docs = {}
        for did in self.doc_ids():
            ents = self.page_index.query(doc_id=did)
            pages = sorted({e.page_no for e in ents})
            docs[did] = {
                "chunks": len(self.manifest[did]["chunk_ids"]),
                "figures": sum(1 for e in ents if e.block_type == "figure"),
                "tables": sum(1 for e in ents if e.block_type == "table"),
                "images": len(self.manifest[did].get("image_paths", [])),
                "pages": f"{pages[0]}–{pages[-1]}" if pages else "-",
            }
        return {
            "docs": docs,
            "total_docs": len(docs),
            "total_chunks": len(self.chunks),
            "total_figures": sum(d["figures"] for d in docs.values()),
        }

    # ── 영속화 (A-RAG 호환 산출물) ──────────────────────────────
    def save(self, paths: Dict[str, str]) -> None:
        os.makedirs(paths["index"], exist_ok=True)
        os.makedirs(os.path.dirname(paths["chunks"]) or ".", exist_ok=True)
        with open(paths["chunks"], "w", encoding="utf-8") as f:
            json.dump([{"id": cid, "text": t} for cid, t in self.chunks.items()], f, ensure_ascii=False)
        with open(os.path.join(paths["index"], "sentence_index.pkl"), "wb") as f:
            pickle.dump({
                "sentences": self.sentences,
                "embeddings": self.embeddings if self.embeddings is not None else np.zeros((0, 1)),
                "sentence_to_chunk": self.sentence_to_chunk,
                "chunks": {cid: {"text": t} for cid, t in self.chunks.items()},
            }, f)
        self.page_index.save(paths["page_index"])
        with open(os.path.join(paths["index"], "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False)

    @classmethod
    def load(cls, paths: Dict[str, str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> "IndexStore":
        pkl = os.path.join(paths["index"], "sentence_index.pkl")
        if not os.path.exists(pkl):
            raise FileNotFoundError(
                f"인덱스가 없습니다: {pkl}. 먼저 빌드하세요(python -m src.indexing.build)."
            )
        store = cls(model_name)
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        store.sentences = data["sentences"]
        store.embeddings = data["embeddings"] if len(data["sentences"]) else None
        store.sentence_to_chunk = data["sentence_to_chunk"]
        store.chunks = {cid: c["text"] for cid, c in data["chunks"].items()}
        store.page_index = PageIndex.load(paths["page_index"])
        mpath = os.path.join(paths["index"], "manifest.json")
        if os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as f:
                store.manifest = json.load(f)
        return store
