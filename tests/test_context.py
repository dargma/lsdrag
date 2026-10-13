from types import SimpleNamespace

import pytest

from lsdrag.context import EvidenceContext


def context():
    chunks = {key: {"document_id": "d", "page_index": 0, "kind": "image_caption", "text": key, "figure_ids": [key]} for key in ("a", "b", "c")}
    figures = {key: {"chunk_id": key, "image_sha256": "same" if key in ("a", "b") else "other", "image_path": key + ".jpg"} for key in chunks}
    return EvidenceContext(SimpleNamespace(chunks=chunks, figures=figures, arm="V2"), max_images=1)


def test_only_successful_reader_transport_counts_as_delivery():
    c = context()
    c.read(["a", "b"])
    assert not c.delivered_figures
    assert len(c.pending_images) == 1
    assert c.read(["a"])[0]["status"] == "already_read"
    c.mark_images_accepted(list(c.pending_images.values()))
    assert c.delivered_figures == {"a", "b"}
    assert c.read(["c"])[0]["figures"][0]["status"] == "image_budget_exhausted"
    assert c.summary()["unique_delivered_images"] == 1


def test_invalid_batch_does_not_partially_mutate_evidence():
    c = context()
    with pytest.raises(ValueError):
        c.read(["a", "missing"])
    assert not c.read_chunks
    assert not c.pending_images
