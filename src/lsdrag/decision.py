"""Decision labels reconstructed from public Agent history; no gold in model input."""
import copy
import json
from pathlib import Path

from .agent import image_message

ACTIONS = ("keyword_search", "semantic_search", "read_chunk", "FINAL")


def action_label(message):
    calls = message.get("tool_calls") or []
    if not calls:
        return "FINAL"
    if len(calls) != 1:
        return None  # Preserve original multi-call behavior; unsupported by this head.
    name = calls[0]["function"]["name"]
    return name if name in ACTIONS[:-1] else None


def trajectory_states(prediction, prompt, figures):
    """Yield exact pre-call history, before appending that call's target output.

    Requires original search-linked single-run records. Reject ambiguous forced
    final/V2 histories rather than silently fabricating training contexts.
    """
    import hashlib
    if hashlib.sha256(prompt.encode()).hexdigest() != prediction["prompt_sha256"]:
        raise ValueError("Prompt hash mismatch")
    if prediction.get("question_image_count", 0) or prediction.get("verification_requested"):
        raise ValueError("Unsupported question-image or verification history")
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": prediction["question"]}]
    trajectories = prediction["trajectory"]
    for call in prediction["reader_calls"]:
        if prediction.get("stop_reason") and call is prediction["reader_calls"][-1]:
            messages.append({"role": "user", "content": "You have reached the limit. Do not call more tools. Return your final JSON answer using the evidence already available; leave answers empty if insufficient."})
        ids = call.get("new_figure_ids", [])
        if ids:
            # Context batches may group duplicate file hashes under multiple IDs.
            grouped = {}
            for fid in ids:
                figure = figures[fid]
                key = figure["image_sha256"]
                grouped.setdefault(key, {"image_path": figure["image_path"], "image_sha256": key, "figure_ids": []})["figure_ids"].append(fid)
            messages.append(image_message(list(grouped.values()), prediction["question"]))
            # Actual Agent appends images before the forced-final instruction.
            if prediction.get("stop_reason") and call is prediction["reader_calls"][-1]:
                messages[-2], messages[-1] = messages[-1], messages[-2]
        target = call["message"]
        yield {"qid": prediction["qid"], "loop": call["loop"], "messages": copy.deepcopy(messages), "action": action_label(target), "forced_final": bool(prediction.get("stop_reason") and call is prediction["reader_calls"][-1])}
        assistant = {"role": "assistant", "content": target.get("content")}
        calls = target.get("tool_calls") or []
        if calls:
            assistant["tool_calls"] = calls
        messages.append(assistant)
        results = [t for t in trajectories if t["loop"] == call["loop"]]
        if len(results) != len(calls):
            raise ValueError("Missing tool result in history")
        for tc, result in zip(calls, results):
            if tc["function"]["name"] != result["tool_name"]:
                raise ValueError("Tool/result order mismatch")
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result["result"], ensure_ascii=False)})


def label_gate(records, minimum_questions=30, minimum_per_action=10):
    answerable = [r for r in records if r.get("answerable", True)]
    counts = {a: len({r["qid"] for r in answerable if a in r["actions"]}) for a in ACTIONS}
    n = len({r["qid"] for r in answerable})
    return {"answerable_successful_questions": n, "distinct_questions_per_action": counts, "passed": n >= minimum_questions and all(v >= minimum_per_action for v in counts.values())}
