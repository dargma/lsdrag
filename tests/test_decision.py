import hashlib
import json

import pytest

from lsdrag.decision import action_label, label_gate, trajectory_states


def test_single_action_and_multi_call_are_distinct():
    call = {"function": {"name": "read_chunk"}}
    assert action_label({"content": "answer"}) == "FINAL"
    assert action_label({"tool_calls": [call]}) == "read_chunk"
    assert action_label({"tool_calls": [call, call]}) is None


def test_history_excludes_current_and_future_targets():
    tool = {"id": "a", "function": {"name": "keyword_search", "arguments": '{"keywords":["public"]}'}}
    prediction = {"qid": "train", "question": "public question", "prompt_sha256": hashlib.sha256(b"prompt").hexdigest(), "reader_calls": [{"loop": 1, "message": {"tool_calls": [tool]}}, {"loop": 2, "message": {"content": "secret_target"}}], "trajectory": [{"loop": 1, "tool_name": "keyword_search", "result": [{"snippet": "evidence"}]}]}
    states = list(trajectory_states(prediction, "prompt", {}))
    assert len(states) == 2
    assert "public question" in json.dumps(states[0]["messages"])
    assert "secret_target" not in json.dumps(states)
    assert "evidence" not in json.dumps(states[0]["messages"])
    assert "evidence" in json.dumps(states[1]["messages"])
    assert states[1]["action"] == "FINAL"
    with pytest.raises(ValueError, match="Prompt hash"):
        list(trajectory_states(prediction, "changed", {}))


def test_gate_counts_questions_not_repeated_turns_or_nulls():
    rows = [{"qid": str(i), "actions": ["FINAL"] * 50} for i in range(30)]
    assert not label_gate(rows)["passed"]
    rows = [{"qid": str(i), "actions": ["FINAL", "read_chunk", "semantic_search", "keyword_search"]} for i in range(30)]
    assert label_gate(rows)["passed"]
    rows[0]["answerable"] = False
    assert not label_gate(rows)["passed"]
