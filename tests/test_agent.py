import hashlib
from types import SimpleNamespace

from PIL import Image

from lsdrag.agent import Agent, AgentConfig, parse_answer, tool_schemas


def test_baseline_arms_have_identical_tool_schemas():
    assert tool_schemas("B0") == tool_schemas("V1") == tool_schemas("C1")
    assert len(tool_schemas("V2")) == 5


def test_transport_and_delivery_after_tool_results(tmp_path):
    path = tmp_path / "figure.png"
    Image.new("RGB", (8, 8), "red").save(path)
    chunk = {"document_id": "d", "page_index": 0, "kind": "image_caption", "text": "A circle", "figure_ids": ["f"]}
    figure = {"image_path": str(path), "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    retrieval = SimpleNamespace(arm="B0", chunks={"c": chunk}, figures={"f": figure}, keyword_search=lambda **_: [], semantic_search=lambda **_: [])

    class FakeReader:
        count = 0
        def chat(self, messages, tools):
            self.count += 1
            if self.count == 1:
                return {"message": {"content": None, "tool_calls": [{"id": "call1", "type": "function", "function": {"name": "read_chunk", "arguments": '{"chunk_ids":["c"]}'}}]}, "finish_reason": "tool_calls"}, {}
            assert [m["role"] for m in messages][-3:] == ["assistant", "tool", "user"]
            assert isinstance(messages[-2]["content"], str)
            assert messages[-1]["content"][-1]["type"] == "image_url"
            return {"message": {"content": '{"answers":["red"],"citations":["c"]}'}, "finish_reason": "stop"}, {"prompt_tokens": 100}

    result = Agent(retrieval, "test", AgentConfig(max_loops=3), FakeReader()).run("qid", "color?")
    assert result["answers"] == ["red"]
    assert result["delivered_figure_ids"] == ["f"]
    assert result["loops"] == 2 and result["status"] == "ok"
    assert "base64" not in str(result)


def test_invalid_answer_is_preserved():
    assert parse_answer("not json")["answers"] == ["not json"]
    assert not parse_answer("not json")["answer_format_valid"]
    recovered = parse_answer('I cannot find it.\n```json\n{"answers":[],"citations":[]}\n```')
    assert recovered["answers"] == []
    assert recovered["answer_json_recovered"]
    assert not parse_answer('{"answers":"invalid"}')["answer_format_valid"]


def test_v2_verification_stays_within_common_loop_budget():
    retrieval = SimpleNamespace(arm="V2", chunks={}, figures={}, keyword_search=lambda **_: [], semantic_search=lambda **_: [], image_search=lambda **_: [])
    class FakeReader:
        count = 0
        def chat(self, messages, tools):
            self.count += 1
            if self.count == 2:
                assert tools is None  # Common max-loop finalization, not extra budget.
            return {"message": {"content": '{"answers":[],"citations":[]}'}, "finish_reason": "stop"}, {}
    reader = FakeReader()
    result = Agent(retrieval, "test", AgentConfig(max_loops=1), reader).run("q", "test?")
    assert result["verification_requested"]
    assert result["stop_reason"] == "max_loops"
    assert result["loops"] == 2


def test_question_image_is_not_counted_as_retrieved_evidence(tmp_path):
    path = tmp_path / "query.png"
    Image.new("RGB", (8, 8)).save(path)
    question_image = {"image_path": str(path), "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "figure_ids": ["question:q"]}
    retrieval = SimpleNamespace(arm="B0", chunks={}, figures={}, keyword_search=lambda **_: [], semantic_search=lambda **_: [])
    class FakeReader:
        def chat(self, messages, tools):
            assert messages[1]["content"][-1]["type"] == "image_url"
            return {"message": {"content": '{"answers":[],"citations":[]}'}, "finish_reason": "stop"}, {}
    result = Agent(retrieval, "test", reader=FakeReader()).run("q", "test?", [question_image])
    assert result["question_image_count"] == 1
    assert result["unique_delivered_images"] == 0
