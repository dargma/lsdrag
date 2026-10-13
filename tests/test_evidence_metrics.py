from lsdrag.evidence_metrics import visual_evidence_scores


def test_found_opened_delivered_are_separate():
    chunks = {"c": {"document_id": "d", "page_index": 0, "figure_ids": ["f"]}}
    figures = {"f": {"document_id": "d", "page_index": 0}}
    review = {"original": {"status": "verified_same_visual_content", "accepted_figure_ids": ["f"]}}
    gold = {"supporting_context": [{"doc_id": "original", "doc_part": "image"}]}
    prediction = {"read_chunk_ids": ["c"], "opened_figure_ids": ["f"], "delivered_figure_ids": [], "trajectory": [{"tool_name": "semantic_search", "result": [{"chunk_id": "c"}]}]}
    scores = visual_evidence_scores(prediction, gold, chunks, figures, review)
    assert scores["figure_candidate_recall"] == 1
    assert scores["figure_open_recall"] == 1
    assert scores["image_evidence_page_recall"] == 1
    assert scores["figure_delivery_recall"] == 0


def test_unmapped_is_unknown_not_a_retrieval_failure():
    prediction = {"read_chunk_ids": [], "opened_figure_ids": [], "delivered_figure_ids": [], "trajectory": []}
    gold = {"supporting_context": [{"doc_id": "original", "doc_part": "image"}]}
    scores = visual_evidence_scores(prediction, gold, {}, {}, {})
    assert scores["figure_delivery_recall"] is None
    assert scores["mapped_gold_image_count"] == 0
    assert not scores["visual_mapping_complete"]
