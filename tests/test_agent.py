import hashlib
import pytest
from types import SimpleNamespace

from PIL import Image

from lsdrag.agent import Agent, AgentConfig, parse_answer, tool_schemas


@pytest.mark.parametrize("policy", ["agent-read", "search-linked"])
def test_baseline_arms_have_identical_tool_schemas(policy):
    assert tool_schemas("B0", policy) == tool_schemas("V1", policy) == tool_schemas("C1", policy) == tool_schemas("G1", policy)
    assert len(tool_schemas("V2", policy)) == 5


@pytest.mark.parametrize("arm", ["B0", "C1", "G1"])
def test_transport_and_delivery_after_tool_results(tmp_path, arm):
    path = tmp_path / "figure.png"
    Image.new("RGB", (8, 8), "red").save(path)
    chunk = {"document_id": "d", "page_index": 0, "kind": "image_caption", "text": "A circle", "figure_ids": ["f"]}
    figure = {"image_path": str(path), "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    retrieval = SimpleNamespace(arm=arm, chunks={"c": chunk}, figures={"f": figure}, keyword_search=lambda **_: [], semantic_search=lambda **_: [])

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
    assert result["loops"] == 1
    assert result["reader_call_count"] == 2


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


def test_eight_agent_iterations_allow_separate_final_reader_call():
    retrieval = SimpleNamespace(arm="G1", chunks={}, figures={}, keyword_search=lambda **_: [], semantic_search=lambda **_: [])
    class RepeatingReader:
        count = 0
        def chat(self, messages, tools):
            self.count += 1
            if self.count == 9:
                assert tools is None
                return {"message": {"content": '{"answers":[],"citations":[]}'}, "finish_reason": "stop"}, {}
            assert tools is not None
            return {"message": {"content": None, "tool_calls": [{"id": str(self.count), "type": "function", "function": {"name": "semantic_search", "arguments": '{"query":"test"}'}}]}, "finish_reason": "tool_calls"}, {}
    reader = RepeatingReader()
    result = Agent(retrieval, "test", AgentConfig(max_loops=8), reader).run("q", "test?")
    assert reader.count == result["reader_call_count"] == 9
    assert result["loops"] == 8
    assert result["stop_reason"] == "max_loops"
    assert len(result["trajectory"]) == 8


@pytest.mark.parametrize("arm", ["B0", "C1", "G1"])
@pytest.mark.parametrize("search_tool", ["keyword_search", "semantic_search"])
@pytest.mark.parametrize("policy", ["agent-read", "search-linked"])
def test_search_image_reaches_final_reader_without_a_read_tool(tmp_path, arm, search_tool, policy):
    path = tmp_path / "synthetic.png"
    Image.new("RGB", (8, 8), "red").save(path)
    chunk = {"document_id": "d", "page_index": 0, "kind": "image_caption",
             "text": "No appearance caption", "figure_ids": ["f"]}
    figure = {"image_path": str(path), "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    results = [{"chunk_id": "c", "kind": "image_caption", "snippet": "No appearance caption"}]
    retrieval = SimpleNamespace(arm=arm, chunks={"c": chunk}, figures={"f": figure},
                                keyword_search=lambda **_: results, semantic_search=lambda **_: results)

    class FakeReader:
        count = 0
        def chat(self, messages, tools):
            self.count += 1
            if self.count == 1:
                args = '{"query":"object"}' if search_tool == "semantic_search" else '{"keywords":["object"]}'
                return {"message": {"content": None, "tool_calls": [{"id": "search", "type": "function",
                        "function": {"name": search_tool, "arguments": args}}]}, "finish_reason": "tool_calls"}, {}
            assert tools is None  # One agent iteration, separate forced final response.
            has_image = any(isinstance(m.get("content"), list) and any(
                b.get("type") == "image_url" for b in m["content"]) for m in messages)
            assert has_image == (policy == "search-linked")
            return {"message": {"content": '{"answers":[],"citations":[]}'}, "finish_reason": "stop"}, {}

    result = Agent(retrieval, "test", AgentConfig(max_loops=1, image_delivery_policy=policy), FakeReader()).run("synthetic", "object?")
    assert result["loops"] == 1 and result["reader_call_count"] == 2
    assert [x["tool_name"] for x in result["trajectory"]] == [search_tool]
    assert result["delivered_figure_ids"] == (["f"] if policy == "search-linked" else [])
    assert result["read_chunk_ids"] == (["c"] if policy == "search-linked" else [])
    assert result["trajectory"][0]["result"][0]["chunk_id"] == "c"
