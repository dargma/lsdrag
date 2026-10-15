"""Full multimodal backbone readout with no full-vocabulary projection.

This retains all vision, text layers, PLE, positional processing and attention.
It is not early exit or a cached-state controller. Selected-row logits are
mathematically equivalent to full projection, but GPU GEMM rounding may differ;
real-backbone feature/action equivalence is a required separate probe.
"""
import time

from .hf_reader import processor_messages


def fast_encode_state(backend, messages, selected_token_logits=False):
    """Return final normalized state, optionally A/B/C probabilities, no decoding.

    Uses precisely the backend's chat template, image soft-token setting and
    one-token context reservation. Timing includes CPU transfer of features.
    """
    torch = backend.torch
    start = time.perf_counter()
    inputs = backend.processor.apply_chat_template(
        processor_messages(messages), add_generation_prompt=True,
        enable_thinking=False, tokenize=True, return_dict=True,
        return_tensors='pt', processor_kwargs={'max_soft_tokens': backend.image_soft_tokens})
    inputs = inputs.to(backend.model.device)
    backend._sync()
    preprocess_seconds = time.perf_counter()-start
    prompt_tokens = int(inputs['input_ids'].shape[1])
    if prompt_tokens+1 > backend.max_context:
        raise ValueError('Actual multimodal context exceeds fixed budget')
    tick = time.perf_counter()
    with torch.inference_mode():
        output = backend.model.model(**inputs, use_cache=True, return_dict=True)
        last = output.last_hidden_state[:, -1, :]
        feature = last[0].float().cpu().numpy().copy()
        probabilities = None
        if selected_token_logits:
            head = backend.model.get_output_embeddings()
            indices = torch.tensor(backend.label_ids, device=head.weight.device, dtype=torch.long)
            weight = head.weight.index_select(0, indices)
            bias = getattr(head, 'bias', None)
            if bias is not None:
                bias = bias.index_select(0, indices)
            logits = torch.nn.functional.linear(last, weight, bias)
            softcap = backend.model.config.get_text_config().final_logit_softcapping
            if softcap is not None:
                logits = logits / softcap
                logits = torch.tanh(logits)
                logits = logits * softcap
            probabilities = logits[0].float().softmax(-1).cpu().tolist()
    backend._sync()
    usage = {'wall_seconds': time.perf_counter()-start,
             'preprocess_seconds': preprocess_seconds,
             'prefill_seconds': time.perf_counter()-tick,
             'decode_seconds': 0., 'prompt_tokens': prompt_tokens,
             'generated_tokens': int(selected_token_logits), 'forced_prefix_tokens': 0,
             'vocab_projection': '3rows' if selected_token_logits else 'none'}
    return {'feature': feature, 'probs': probabilities,
            'action_probs': probabilities, 'usage': usage}
