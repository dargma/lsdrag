"""05_indexing 회귀 — 가짜 임베더로 오프라인. 자기조회·재로드·증분·A-RAG 호환 산출물."""
import json
import os
import pickle
import tempfile

import numpy as np

from src.indexing import IndexStore
from src.schema import ParsedBlock, ParsedDoc

DIM = 8


def fake_embed(texts):
    """결정적 가짜 임베더: 단어 해시 → 정규화 벡터."""
    out = np.zeros((len(texts), DIM), dtype=float)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, hash(w) % DIM] += 1.0
        n = np.linalg.norm(out[i]) or 1.0
        out[i] /= n
    return out


def _doc(doc_id="d1"):
    return ParsedDoc(doc_id=doc_id, title="t", blocks=[
        ParsedBlock(text="alpha beta gamma. delta epsilon.", page_no=1, chunk_id=f"{doc_id}:0"),
        ParsedBlock(text="zeta eta theta.", page_no=2, block_type="figure",
                    figure_no="1", image_path="data/images/x.png", chunk_id=f"{doc_id}:1"),
    ])


def _paths(d):
    return {"index": os.path.join(d, "index"),
            "page_index": os.path.join(d, "page_index"),
            "chunks": os.path.join(d, "index", "chunks.json")}


def test_self_retrieval_top1():
    store = IndexStore(); store.add_doc(_doc(), fake_embed)
    q = fake_embed(["alpha beta gamma"])[0]
    sims = store.embeddings @ q
    top = store.sentence_to_chunk[int(np.argmax(sims))]
    assert top == "d1:0"


def test_arag_compatible_artifacts():
    store = IndexStore(); store.add_doc(_doc(), fake_embed)
    with tempfile.TemporaryDirectory() as d:
        store.save(_paths(d))
        # chunks.json = [{'id','text'}]
        chunks = json.load(open(_paths(d)["chunks"], encoding="utf-8"))
        assert chunks and {"id", "text"} <= set(chunks[0])
        # sentence_index.pkl 키
        data = pickle.load(open(os.path.join(_paths(d)["index"], "sentence_index.pkl"), "rb"))
        assert {"sentences", "embeddings", "sentence_to_chunk", "chunks"} <= set(data)
        assert isinstance(data["chunks"], dict) and "text" in next(iter(data["chunks"].values()))


def test_reload_roundtrip():
    store = IndexStore(); store.add_doc(_doc(), fake_embed)
    with tempfile.TemporaryDirectory() as d:
        store.save(_paths(d))
        again = IndexStore.load(_paths(d))
        assert again.doc_ids() == store.doc_ids()
        assert len(again.sentences) == len(store.sentences)
        assert again.page_index.query(page=2, figure_no="1")


def test_incremental_add_remove():
    store = IndexStore()
    store.add_doc(_doc("a"), fake_embed); store.add_doc(_doc("b"), fake_embed)
    assert store.doc_ids() == ["a", "b"]
    n_sent = len(store.sentences)
    store.remove_doc("a")
    assert store.doc_ids() == ["b"]
    assert all(not c.startswith("a:") for c in store.sentence_to_chunk)
    assert len(store.sentences) < n_sent
    assert store.embeddings.shape[0] == len(store.sentences)


def test_remove_then_readd_idempotent():
    store = IndexStore()
    store.add_doc(_doc("a"), fake_embed)
    store.add_doc(_doc("a"), fake_embed)  # 재추가=갱신
    assert len(store.manifest["a"]["chunk_ids"]) == 2
    assert store.embeddings.shape[0] == len(store.sentences)


def test_status_view():
    store = IndexStore()
    store.add_doc(_doc("a"), fake_embed)
    st = store.status()
    assert st["total_docs"] == 1 and st["total_chunks"] == 2
    assert st["docs"]["a"]["figures"] == 1 and st["docs"]["a"]["tables"] == 0
    assert st["docs"]["a"]["pages"] != "-"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("[05_indexing] all regression tests passed")
