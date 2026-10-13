from PIL import Image
import pytest

from lsdrag.memgallery import convert_scenario, image_reference, select_questions
from lsdrag.io import write_jsonl
from lsdrag.retrieval import Retrieval
from lsdrag.context import EvidenceContext


def test_selection_ignores_answers_and_input_order():
    rows = [{"qid": str(i), "scenario": "a" if i < 10 else "b", "answer": str(i)} for i in range(20)]
    first = select_questions(rows, 5, 123)
    changed = [{**r, "answer": "changed"} for r in reversed(rows)]
    assert [r["qid"] for r in first] == [r["qid"] for r in select_questions(changed, 5, 123)]


def test_preserve_all_images_and_adjacent_turns(tmp_path):
    folder = tmp_path / "data/image/topic"
    folder.mkdir(parents=True)
    for name in ("first.png", "second.png"):
        Image.new("RGB", (10, 10)).save(folder / name)
    data = {"character_profile": {"name": "Person"}, "multi_session_dialogues": [{"session_id": "D1", "date": "2024-01-01", "dialogues": [{"round": "D1:1", "user": "Compare these", "assistant": "Two shapes", "image_id": ["D1:IMG_1", "D1:IMG_2"], "input_image": ["../image/topic/first.png", "../image/topic/second.png"], "image_caption": ["First", "Second"]}, {"round": "D1:2", "user": "Thanks", "assistant": "Welcome"}]}]}
    chunks, figures, documents = convert_scenario("topic", data, tmp_path)
    assert len(figures) == 2
    assert chunks[0]["next_chunk_id"] == chunks[1]["chunk_id"]
    assert "D1:IMG_2" in chunks[0]["text"]
    assert len(documents) == 1


def test_image_references_cannot_escape_local_copy():
    with pytest.raises(ValueError):
        image_reference("../image/../../outside.png")


def test_question_scope_excludes_other_scenarios(tmp_path):
    chunks = [{"chunk_id": key, "document_id": key, "document_ids": [key], "text": text, "page_index": 0, "kind": "text", "figure_ids": []} for key, text in (("a", "private apple"), ("b", "public orange"))]
    write_jsonl(tmp_path / "chunks.jsonl", chunks)
    write_jsonl(tmp_path / "figures.jsonl", [])
    retrieval = Retrieval(tmp_path, {}, {}, "B0", document_scope=["b"])
    assert not retrieval.keyword_search(["apple"])
    assert retrieval.keyword_search(["orange"])[0]["chunk_id"] == "b"
    with pytest.raises(ValueError, match="Unknown chunk"):
        EvidenceContext(retrieval).read(["a"])
