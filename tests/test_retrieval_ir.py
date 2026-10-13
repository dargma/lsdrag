import numpy as np
import pytest
from PIL import Image

from lsdrag.ir import convert_document, split_table
from lsdrag.retrieval import VectorIndex, reciprocal_rank_fusion


def test_rrf_ranks_only_and_does_not_duplicate_votes():
    assert reciprocal_rank_fusion([["a", "a"], ["b"]]) == ["a", "b"]
    assert reciprocal_rank_fusion([["a", "b"], ["b", "c"]])[0] == "b"


def test_sentence_scores_aggregate_and_filter_figures():
    index = VectorIndex.__new__(VectorIndex)
    index.rows = [{"chunk_id": "a", "figure_id": "x"}, {"chunk_id": "a", "figure_id": "y"}, {"chunk_id": "b", "figure_id": "z"}]
    index.vectors = np.array([[1., 0.], [.9, .1], [.8, .2]])
    assert [key for key, _ in index.search([1., 0.], 2)] == ["a", "b"]
    assert index.search([1., 0.], 2, "figure_id", {"z"}) == [("z", .8)]
    assert index.search([1., 0.], 2, "figure_id", {"z"}, row_limit=1) == [("z", .8)]


def test_saved_asset_reference_and_visual_footnote(tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "actual.png")
    block = {"type": "image", "content": "", "image_source": "actual.png", "footnotes": [{"content": "A red bridge."}]}
    source = {"pages": [{"page_idx": 2, "blocks": [block]}]}
    meta = {"document_id": "d", "document_ids": ["d"], "title": "Bridge"}
    chunks, figures = convert_document(source, meta, tmp_path)
    assert "A red bridge." in chunks[0]["text"]
    assert figures[0]["has_description"] and not figures[0]["has_caption"]
    assert figures[0]["page_index"] == 2
    block["image_source"] = "missing.png"
    with pytest.raises(ValueError, match="Missing"):
        convert_document(source, meta, tmp_path)


def test_table_rows_keep_headers():
    table = "| Name | Value |\n| --- | --- |\n| First | 1 |\n| Second | 2 |"
    parts = split_table(table, max_words=16)
    assert len(parts) == 2
    assert all(p.startswith("| Name | Value |\n| --- | --- |") for p in parts)
    assert "| Second | 2 |" in parts[1]


def test_google_multimodal_retrieves_images_without_qwen():
    from lsdrag.retrieval import Retrieval
    from types import SimpleNamespace
    class FakeIndex:
        last_matches = {}
        def __init__(self, ranking): self.ranking = ranking
        def search(self, *args, **kwargs): return [(key, 1.0) for key in self.ranking]
    retrieval = Retrieval.__new__(Retrieval)
    retrieval.arm = "G1"
    retrieval.scoped = False
    retrieval.candidate_count = 60
    retrieval.query_cache = {}
    retrieval.chunks = {key: {"chunk_id": key, "document_id": "d", "page_index": 0, "kind": "image_caption", "text": key} for key in ["caption", "visual"]}
    retrieval.indexes = {"gemma_text": FakeIndex(["caption", "visual"]), "gemma_image": FakeIndex(["visual"])}
    retrieval.embedders = {"gemma": SimpleNamespace(encode=lambda *args: np.array([[1., 0.]]))}
    assert retrieval.semantic_search("captionless object", top_k=1)[0]["chunk_id"] == "visual"
    assert list(retrieval.embedders) == ["gemma"]
