import pytest

from lsdrag.metrics import contain_all, document_read_scores, paired_comparison, cluster_paired_bootstrap


def test_list_containment_requires_all_answers():
    assert contain_all("The red and blue badges", ["red", "blue"]) == 1
    assert contain_all("red", ["red", "blue"]) == 0
    assert contain_all("anything", [""]) == 0


def test_document_aliases_and_actual_reads():
    chunks = {"c": {"document_ids": ["original", "alias"]}}
    gold = {"supporting_context": [{"doc_id": "alias", "doc_part": "image"}], "metadata": {"type": "ImageQ"}}
    assert document_read_scores({"read_chunk_ids": ["c"]}, gold, chunks)["document_recall"] == 1
    assert document_read_scores({"read_chunk_ids": []}, gold, chunks)["document_recall"] == 0
    result = document_read_scores({"read_chunk_ids": [], "trajectory": [{"tool_name": "semantic_search", "result": [{"chunk_id": "c"}]}]}, gold, chunks)
    assert result["document_candidate_recall"] == 1 and result["document_recall"] == 0


def test_pairing_and_statistics():
    a = {"q1": {"em": 0}, "q2": {"em": 1}}
    b = {"q2": {"em": 0}, "q1": {"em": 1}}
    result = paired_comparison(a, b, "em")
    assert result["difference"] == 0
    assert result["gains"] == result["losses"] == 1
    assert result["mcnemar_exact_pvalue"] == 1
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(a, {"q3": {"em": 1}}, "em")


def test_cluster_bootstrap_keeps_related_questions_together():
    a = {"q1": {"em": 0, "scenario": "a"}, "q2": {"em": 0, "scenario": "a"}, "q3": {"em": 1, "scenario": "b"}}
    b = {"q1": {"em": 1, "scenario": "a"}, "q2": {"em": 1, "scenario": "a"}, "q3": {"em": 0, "scenario": "b"}}
    result = cluster_paired_bootstrap(a, b, "em")
    assert result["clusters"] == 2 and result["questions"] == 3
    assert result["difference"] == pytest.approx(1 / 3)
    assert result["cluster_bootstrap_95ci"] == [-1, 1]
