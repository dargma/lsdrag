"""LoRA + GRPO for native A-RAG (LLM picks the tool) and head A-RAG (head picks, LLM writes arguments).

Rollouts sample every policy token at temperature 1 under the common Qwen3.5 protocol (Qwen35Reader).
Each decision stores its full token sequence so training recomputes token log-probs (and, for the head,
the pack hidden) with one teacher-forced forward through the LoRA model. Recorded log-probs are the
unmasked full-vocabulary log-softmax; grammar fallback only changes which token is sampled.
"""
import copy
import json
import time
import torch
from jsonschema import validate
from .decision_policy import distribution, pack_hidden
from .decision_qwen35_decode import tool_boundary
from .text_integrated import ACTIONS, options_for, argument_schema, advance_json_parser, observed_ids
from .text_integrated_aligned import syntax_prefix
from .text_decision_benchmark import TOOLS, FINAL_SCHEMA, canonical

LORA_TARGETS = r'.*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'


def attach_lora(model, rank=16, alpha=32):
    from peft import LoraConfig, inject_adapter_in_model
    inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0., target_modules=LORA_TARGETS), model)
    params = [p for n, p in model.named_parameters() if 'lora_' in n]
    for p in params:  # fp32 master weights; peft casts activations to the adapter dtype
        p.data = p.data.float()
        p.requires_grad_(True)
    return params


def lora_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if 'lora_' in n}


def load_lora(model, path):
    state = torch.load(path, weights_only=True, map_location='cpu')
    named = dict(model.named_parameters())
    assert set(state) == {n for n in named if 'lora_' in n}
    with torch.no_grad():
        for n, v in state.items():
            named[n].copy_(v)


def _sample(logits, allowed, temperature, generator):
    z = logits if allowed is None else logits[allowed]
    p = torch.softmax(z/temperature, -1)
    i = int(torch.multinomial(p, 1, generator=generator))
    return i if allowed is None else allowed[i]


def _feed(reader, pending, tokens):
    t = torch.tensor([tokens], device=reader.model.device)
    pending['mask'] = torch.cat([pending['mask'], torch.ones_like(t)], -1)
    with torch.inference_mode():
        out = reader.model.model(input_ids=t, attention_mask=pending['mask'], past_key_values=pending['past'], use_cache=True, return_dict=True)
    pending['past'] = out.past_key_values
    pending['last_hidden'] = out.last_hidden_state[:, -1].detach().clone()
    pending['ids'] = pending['ids']+tokens


def sample_native_prefix(reader, history, temperature, generator):
    """Sample the tool name through the legal-prefix trie; trie choice tokens are policy tokens."""
    legal = [a for a in ACTIONS if a != 'read_chunk' or observed_ids(history)]
    prefixes = {a: ('{"answer":', '{"answer":') if a == 'FINAL' else syntax_prefix(a) for a in legal}
    seqs = {a: reader.tokenizer.encode(v[0], add_special_tokens=False) for a, v in prefixes.items()}
    pending, emitted, policy = reader.pending, [], []
    while True:
        active = {a: s for a, s in seqs.items() if s[:len(emitted)] == emitted}
        done = [a for a, s in active.items() if len(s) == len(emitted)]
        if done:
            return done[0], prefixes[done[0]][1], emitted, policy
        nxt = sorted({s[len(emitted)] for s in active.values()})
        if len(nxt) == 1:
            tid = nxt[0]
        else:
            with torch.inference_mode():
                logits = reader.model.lm_head(pending['last_hidden'])[0].float()
            tid = _sample(logits, nxt, temperature, generator)
            policy.append((len(emitted), tid, float(torch.log_softmax(logits, -1)[tid])))
        emitted.append(tid)
        _feed(reader, pending, [tid])


def sample_payload(reader, action, history, forced_ids, argprefix, temperature, generator, budget):
    """Sampling twin of decode_cached: forced syntax prefix, then grammar-checked sampled JSON."""
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer
    pending = reader.pending
    schema = FINAL_SCHEMA if action == 'FINAL' else argument_schema(action, history)
    initial = advance_json_parser(JsonSchemaParser(schema, reader.json_config), argprefix)
    parser, generated, logps, decoded, complete, error = initial, [], [], '', False, None
    if forced_ids:
        _feed(reader, pending, forced_ids)
    seq = list(pending['ids'])
    mask, past = pending['mask'], pending['past']
    with torch.inference_mode():
        logits = reader.model.lm_head(pending['last_hidden'])[0].float()
        for _ in range(budget):
            tid = _sample(logits, None, temperature, generator)
            proposed = reader.tokenizer.decode(generated+[tid], skip_special_tokens=True)
            if tool_boundary(argprefix+proposed, action, schema) is not None:
                generated.append(tid); logps.append(float(torch.log_softmax(logits, -1)[tid])); decoded = proposed; complete = True
                break
            candidate = advance_json_parser(initial, proposed)
            if tid in reader.stop_ids or candidate is None or len(proposed) <= len(decoded):
                allowed = TokenEnforcer(reader.grammar_data, parser).get_allowed_tokens(seq).allowed_tokens
                tid = _sample(logits, allowed, temperature, generator)
                proposed = reader.tokenizer.decode(generated+[tid], skip_special_tokens=True)
                candidate = advance_json_parser(initial, proposed)
                if candidate is None:
                    error = 'grammar token mismatch'
                    break
            generated.append(tid); logps.append(float(torch.log_softmax(logits, -1)[tid])); seq.append(tid)
            decoded, parser = proposed, candidate
            try:
                json.loads(argprefix+decoded); complete = True
                break
            except json.JSONDecodeError:
                pass
            token = torch.tensor([[tid]], device=reader.model.device)
            mask = torch.cat([mask, torch.ones_like(token)], -1)
            out = reader.model(input_ids=token, attention_mask=mask, past_key_values=past, use_cache=True, logits_to_keep=1)
            past, logits = out.past_key_values, out.logits[0, -1].float()
    reader.pending = None
    text = argprefix+decoded
    value = None
    if error is None:
        try:
            if not complete:
                raise ValueError('truncated')
            b = tool_boundary(text, action, schema)
            value = b[0] if b is not None else json.loads(text)
            validate(value, schema)
        except Exception as e:
            error, value = f'{type(e).__name__}: {e}', None
    return value, generated, logps, error


def sample_episode(reader, corpus, prompt, question, head, temperature, generator, max_calls=8, budget=512):
    """head=None -> native A-RAG. Returns episode summary and per-decision training records."""
    messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': question}]
    history, records, errors, final, prompt_tokens = [], [], [], None, 0
    start = time.perf_counter()
    while True:
        forced = len(history) >= max_calls
        pack, usage = reader.capture_pack(messages, history, float('inf'))
        prompt_tokens += usage['prompt_tokens']
        prompt_ids = list(reader.pending['ids'])
        rec = dict(prompt_len=len(prompt_ids), qs=usage['question_span'], obs=usage['observation_span'], forced=forced)
        legal = torch.tensor([[any(o['action'] == a for o in options_for(question, history)) for a in ACTIONS]], device=pack.device)
        if forced:
            legal[:] = False; legal[:, 3] = True
        policy = []
        if head is None and not forced:
            name, argprefix, emitted, policy = sample_native_prefix(reader, history, temperature, generator)
            forced_ids = []
            prefix_ids = emitted
        else:
            if head is not None and not forced:
                with torch.no_grad():
                    dist = distribution(head(pack.unsqueeze(0), legal)[0], legal)
                    action = int(torch.multinomial(dist.probs[0], 1, generator=generator))
                    rec.update(head_action=action, head_old_logp=float(dist.log_prob(torch.tensor([action], device=pack.device))),
                               legal=legal[0].cpu())
            else:
                action = 3
            name = ACTIONS[action]
            prefix, argprefix = syntax_prefix(name)
            forced_ids = reader.tokenizer.encode(prefix, add_special_tokens=False)
            prefix_ids = forced_ids
        value, generated, logps, error = sample_payload(reader, name, history, forced_ids, argprefix, temperature, generator, budget)
        prompt_tokens += len(prefix_ids)  # forced/trie prefix tokens are prefilled, as in rollout()
        seq = prompt_ids+prefix_ids+generated
        base = len(prompt_ids)
        pos = [base+i for i, _, _ in policy]+[base+len(prefix_ids)+i for i in range(len(generated))]
        old = [lp for _, _, lp in policy]+logps
        rec.update(seq=seq, positions=pos, old_logps=old, action=name)
        records.append(rec)
        if error:
            errors.append(error)
            break
        if name == 'FINAL':
            final = value
            break
        result = corpus.call(name, value)
        history.append({'tool_name': name, 'arguments': value, 'result': result})
        tc = {'id': f'call_{len(history)-1}', 'type': 'function', 'function': {'name': name, 'arguments': canonical(value)}}
        messages += [{'role': 'assistant', 'content': None, 'tool_calls': [tc]},
                     {'role': 'tool', 'tool_call_id': tc['id'], 'content': canonical(result)}]
    reader.pending = None
    episode = dict(answer=final['answer'] if final else '', errors=errors, trajectory=history, tool_calls=len(history),
                   prompt_tokens=prompt_tokens, generated_tokens=sum(len(r['positions']) for r in records),
                   decisions=len(records), actions=[r['action'] for r in records], seconds=time.perf_counter()-start)
    return episode, records


def reward(em, f1, prompt_tokens, error):
    return em+.25*f1-.2*prompt_tokens/20000-.5*error


def decision_logps(model, head, rec, device):
    """Teacher-forced forward with grad; returns (token logps, head logp or None)."""
    last = max(rec['positions'])+1 if rec['positions'] else rec['prompt_len']
    ids = torch.tensor([rec['seq'][:last]], device=device)
    h = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state[0]
    tok = None
    if rec['positions']:
        pos = torch.tensor(rec['positions'], device=device)
        logits = model.lm_head(h[pos-1]).float()
        tok = torch.log_softmax(logits, -1).gather(1, torch.tensor([rec['seq'][p] for p in rec['positions']], device=device)[:, None])[:, 0]
    hl = None
    if head is not None and 'head_action' in rec:
        pack = pack_hidden(h[:rec['prompt_len']], rec['qs'], rec['obs'])
        legal = rec['legal'].to(device)[None]
        hl = distribution(head(pack[None], legal)[0], legal).log_prob(torch.tensor([rec['head_action']], device=device))[0]
    return tok, hl
