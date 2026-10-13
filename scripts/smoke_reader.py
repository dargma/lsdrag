"""Check native tool calling and both image delivery message layouts."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw
import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8102/v1")
    p.add_argument("--model", default="gemma-reader")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    image = Image.new("RGB", (384, 384), "white")
    ImageDraw.Draw(image).ellipse((40, 40, 344, 344), fill="red")
    stream = BytesIO()
    image.save(stream, format="PNG")
    media = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()}}
    tool = {"type": "function", "function": {"name": "figure_read", "description": "Open the requested figure. Its appearance is unknown until this tool returns an image.", "parameters": {"type": "object", "properties": {"figure_id": {"type": "string"}}, "required": ["figure_id"]}}}
    history = [{"role": "system", "content": "Use the tool to inspect the requested figure before answering. Do not guess its appearance."}, {"role": "user", "content": "Open figure f1 and report the color and shape of the object shown. Answer in a short phrase."}]
    def call(messages, **kwargs):
        payload = {"model": args.model, "messages": messages, "temperature": 0, "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False}, **kwargs}
        start = time.monotonic()
        r = requests.post(args.base_url + "/chat/completions", json=payload, timeout=180)
        body = r.json()
        return r.status_code, body, time.monotonic() - start
    status, initial, seconds = call(history, tools=[tool], tool_choice="auto")
    report = {"purpose": "runtime smoke only; synthetic figure", "model": args.model, "tool_call": {"http_status": status, "response": initial, "seconds": seconds}}
    if status != 200:
        raise RuntimeError(json.dumps(report))
    message = initial["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if not calls:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError("Model did not call the image tool")
    call_id = calls[0]["id"]
    assistant = {"role": "assistant", "content": message.get("content"), "tool_calls": calls}
    blocks = [{"type": "text", "text": "Figure f1 is displayed below. Describe only the depicted object's color and shape."}, media]
    attempts = {
        "image_in_tool": history + [assistant, {"role": "tool", "tool_call_id": call_id, "content": blocks}],
        "image_after_tool_as_user": history + [assistant, {"role": "tool", "tool_call_id": call_id, "content": "Figure f1 loaded. Its image follows in the next message."}, {"role": "user", "content": blocks}],
    }
    for name, messages in attempts.items():
        # Run the server with --mm-processor-cache-gb 0. A rejected tool-role
        # image otherwise leaves a frontend-only cache entry in vLLM 0.20.
        status, response, seconds = call(messages)
        text = response.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        report[name] = {"http_status": status, "response": response, "seconds": seconds, "recognized_color_and_shape": status == 200 and "red" in text.lower() and "circle" in text.lower()}
    report["selected_transport"] = next((name for name in attempts if report[name]["recognized_color_and_shape"]), None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k in {"model", "selected_transport"}}))
    if not report["selected_transport"]:
        raise RuntimeError("Neither image transport passed the visual check")


if __name__ == "__main__":
    main()
