"""Single-request HF Gemma Reader with shared prefill/KV for an optional head."""
import base64
import copy
import io
import json
import re
import time
from jsonschema import ValidationError

from .decision import ACTIONS


def processor_messages(messages):
    from PIL import Image
    converted = copy.deepcopy(messages)
    for message in converted:
        for call in message.get("tool_calls") or []:
            if isinstance(call["function"].get("arguments"), str):
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        if isinstance(message.get("content"), str):
            message["content"] = [{"type": "text", "text": message["content"]}]
        if isinstance(message.get("content"), list):
            for block in message["content"]:
                if block.get("type") == "image_url":
                    url = block["image_url"]["url"]
                    if not url.startswith("data:image/"):
                        raise ValueError("Only audited inline images are supported")
                    block.clear()
                    block.update(type="image", image=Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))
    return converted


def parse_native(text):
    # Match the existing vLLM parser and use its typed argument parser, preserving
    # arrays/integers and multiple tool calls. Dependency is pinned in runtime.
    from vllm.tool_parsers.gemma4_tool_parser import _parse_gemma4_args
    pattern = re.compile(r"<\|tool_call>call:([\w\-.]+)\{(.*?)\}<tool_call\|>", re.S)
    matches = list(pattern.finditer(text))
    calls = [{"id": f"call_{i}", "type": "function", "function": {"name": m[1], "arguments": json.dumps(_parse_gemma4_args(m[2]), ensure_ascii=False)}} for i, m in enumerate(matches)]
    content = pattern.sub("", text)
    for token in ("<turn|>", "<|endoftext|>", "<eos>", "<|tool_response>"):
        content = content.replace(token, "")
    if "<|tool_call>" in content:
        raise ValueError("Incomplete native tool call")
    return {"role": "assistant", "content": content.strip() or None, **({"tool_calls": calls} if calls else {})}


class HFReader:
    def __init__(self, model_path, config, head_path=None):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch, self.config = torch, config
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", local_files_only=True).eval()
        self.model.requires_grad_(False)
        self.head = None
        if head_path:
            checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
            if tuple(checkpoint["actions"]) != ACTIONS:
                raise ValueError("Wrong head class order")
            self.head = torch.nn.Linear(checkpoint["weight"].shape[1], len(ACTIONS)).cuda().float()
            self.head.load_state_dict({k: checkpoint[k] for k in ("weight", "bias")})
            self.head.eval().requires_grad_(False)
        self.feature = None
        self.hook = self.model.model.language_model.norm.register_forward_hook(self._capture)
        eos = self.model.generation_config.eos_token_id
        self.stop_ids = set(eos if isinstance(eos, list) else [eos])
        self.tool_token = self.processor.tokenizer.convert_tokens_to_ids("<|tool_call>")

    def _capture(self, module, args, output):
        self.feature = output[:, -1, :].detach().clone()

    def _sync(self):
        self.torch.cuda.synchronize()

    def prepare(self, messages, tools):
        inputs = self.processor.apply_chat_template(processor_messages(messages), tools=tools or None, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt", processor_kwargs={"max_soft_tokens": self.config.image_soft_tokens})
        return inputs.to(self.model.device)

    def prefill(self, messages, tools):
        inputs = self.prepare(messages, tools)
        if inputs["input_ids"].shape[1] + self.config.max_output_tokens > self.config.context_limit:
            raise ValueError("Actual HF context exceeds fixed context budget")
        self._sync()
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model(**inputs, use_cache=True, logits_to_keep=1)
        self._sync()
        return inputs, output, self.feature, time.perf_counter() - start

    def chat(self, messages, tools):
        torch = self.torch
        start = time.perf_counter()
        inputs, output, feature, prefill_seconds = self.prefill(messages, tools)
        head_seconds, forced_tokens, action, probabilities = 0., [], None, None
        mask = inputs["attention_mask"]
        prefix_seconds = 0.
        with torch.inference_mode():
            if self.head is not None and tools:
                self._sync()
                tick = time.perf_counter()
                probs = self.head(feature.float()).softmax(-1)[0]
                action = ACTIONS[int(probs.argmax())]
                probabilities = probs.cpu().tolist()
                self._sync()
                head_seconds = time.perf_counter() - tick
                prefix = '{"answers":' if action == "FINAL" else f"<|tool_call>call:{action}{{"
                forced_tokens = self.processor.tokenizer.encode(prefix, add_special_tokens=False)
                ids = torch.tensor([forced_tokens], device=self.model.device)
                mask = torch.cat([mask, torch.ones_like(ids)], -1)
                self._sync()
                tick = time.perf_counter()
                output = self.model(input_ids=ids, attention_mask=mask, past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
                self._sync()
                prefix_seconds = time.perf_counter() - tick
            generated = []
            self._sync()
            decode_start = time.perf_counter()
            for _ in range(self.config.max_output_tokens - len(forced_tokens)):
                logits = output.logits[:, -1, :].float()
                if action == "FINAL":
                    logits[:, self.tool_token] = -torch.inf
                token = logits.argmax(-1, keepdim=True)
                value = int(token.item())
                generated.append(value)
                if value in self.stop_ids:
                    break
                mask = torch.cat([mask, torch.ones_like(token)], -1)
                output = self.model(input_ids=token, attention_mask=mask, past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
            self._sync()
            decode_seconds = time.perf_counter() - decode_start
        raw = self.processor.tokenizer.decode(forced_tokens + generated, skip_special_tokens=False)
        usage = {"prompt_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": len(generated) + len(forced_tokens), "generated_tokens": len(generated), "forced_prefix_tokens": len(forced_tokens), "prefill_seconds": prefill_seconds, "head_seconds": head_seconds, "prefix_seconds": prefix_seconds, "decode_seconds": decode_seconds, "reader_wall_seconds": time.perf_counter() - start, "selected_action": action, "action_probabilities": probabilities, "kv_reused": True, "raw_native": raw}
        try:
            message = parse_native(raw)
            if action == "FINAL":
                from .agent import parse_answer
                if message.get("tool_calls") or not parse_answer(message.get("content") or "")["answer_format_valid"]:
                    raise ValueError("FINAL did not generate the required answer JSON")
            if action and action != "FINAL":
                names = [c["function"]["name"] for c in message.get("tool_calls", [])]
                if names != [action]:
                    raise ValueError("Selected action and generated arguments disagree")
                from jsonschema import validate
                schema = next(t["function"]["parameters"] for t in tools if t["function"]["name"] == action)
                validate(json.loads(message["tool_calls"][0]["function"]["arguments"]), schema)
        except (ValueError, ValidationError) as error:
            if self.head is None:
                # Original generation can have malformed outputs, scored normally.
                message = {"role": "assistant", "content": raw}
            else:
                # Structural fallback stays inside the same solve. Include all
                # discarded work and the additional prefill/generation in totals.
                del output, inputs, mask
                saved_head, self.head = self.head, None
                try:
                    choice, fallback = self.chat(messages, tools)
                finally:
                    self.head = saved_head
                for key in ("prompt_tokens", "completion_tokens", "generated_tokens", "prefill_seconds", "prefix_seconds", "decode_seconds"):
                    usage[key] += fallback[key]
                usage.update(fallback_count=1, fallback_reason=str(error), fallback_usage=fallback, reader_wall_seconds=time.perf_counter() - start)
                return choice, usage
        return {"message": message, "finish_reason": "stop" if generated and generated[-1] in self.stop_ids else "length"}, usage
