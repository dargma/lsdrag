"""Sampling reader and teacher-forced replay for full fine-tuning GRPO (native vs head ARAG).

Both arms record, per decision step, one segment: the context the step saw plus every token appended
after it, each tagged forced (syntax supplied by the runtime), branch (native tool-name choice among the
legal trie branches), free (sampled payload token) or fallback (grammar replaced the sample; not trained).
FINAL starts with the forced prefix '{"answer":' in both arms. temperature 0 is greedy and reproduces
the original native_fast / head paths except for that FINAL prefix in head mode.
"""
import json
import time
from types import SimpleNamespace

import torch
from jsonschema import validate

from .decision_policy_recovery import RecoveryPolicyReader
from .text_decision_benchmark import FINAL_SCHEMA
from .text_integrated import ACTIONS, argument_schema, advance_json_parser, observed_ids
from .text_integrated_aligned import syntax_prefix

FINAL_PREFIX = '{"answer":'


def prefixes_for(action):
    return (FINAL_PREFIX, FINAL_PREFIX) if action == 'FINAL' else syntax_prefix(action)


class SampledReader(RecoveryPolicyReader):
    v22 = False
    pointer_head = None
    head_temperature = 1.
    temperature = 0.
    decision_temperature = None   # native tool-name branches; None = same as temperature
    answer_max_chars = None       # grammar cap on the FINAL answer string (token constraint)

    def payload_schema(self, action, history):
        schema = FINAL_SCHEMA if action == 'FINAL' else argument_schema(action, history)
        if self.v22 and action == 'read_chunk':
            from .chunk_pointer import unread_candidates
            ids = [cid for cid, _ in unread_candidates(history)]
            if not ids:
                raise ValueError('read_chunk has no unread candidate')
            schema['properties']['chunk_ids']['items']['enum'] = ids
            schema['properties']['chunk_ids']['maxItems'] = min(8, len(ids))
            schema['properties']['chunk_ids']['uniqueItems'] = True
        return schema

    def _choose(self, logits, allowed=None, temperature=None):
        """Return (token, log-prob at T (1 if greedy), entropy) over allowed ids (or full vocab)."""
        T = self.temperature if temperature is None else temperature
        x = logits if allowed is None else logits[allowed]
        logp = (x/(T or 1.)).log_softmax(-1)
        i = int(torch.multinomial(logp.exp(), 1)) if T else int(x.argmax())
        ent = float(-(logp.exp()*logp).sum())
        return (i if allowed is None else allowed[i]), float(logp[i]), ent

    def _feed(self, state, tokens):
        t = torch.tensor([tokens], device=self.model.device)
        state['mask'] = torch.cat([state['mask'], torch.ones_like(t)], -1)
        out = self.model(input_ids=t, attention_mask=state['mask'], past_key_values=state['past'], use_cache=True, logits_to_keep=1)
        state['past'] = out.past_key_values; state['logits'] = out.logits[0, -1].float()

    def _payload(self, state, action, trajectory, argprefix, segment, deadline, budget):
        from lmformatenforcer import JsonSchemaParser, TokenEnforcer
        schema = self.payload_schema(action, trajectory)
        if action == 'FINAL' and self.answer_max_chars:
            schema = json.loads(json.dumps(schema)); schema['properties']['answer']['maxLength'] = self.answer_max_chars
        initial = advance_json_parser(JsonSchemaParser(schema, self.json_config), argprefix); parser = initial
        sequence = segment['context_ids']+segment['tokens']; generated = []; decoded = ''; complete = False; error = None
        for _ in range(budget):
            if time.perf_counter() >= deadline:
                raise TimeoutError('payload deadline')
            tid, lp, _ = self._choose(state['logits']); kind = 'free'
            proposed = self.tokenizer.decode(generated+[tid], skip_special_tokens=True); candidate = advance_json_parser(initial, proposed)
            if tid in self.stop_ids or candidate is None or len(proposed) <= len(decoded):
                allowed = TokenEnforcer(self.grammar_data, parser).get_allowed_tokens(sequence).allowed_tokens
                tid, lp, _ = self._choose(state['logits'], allowed); kind = 'fallback'
                proposed = self.tokenizer.decode(generated+[tid], skip_special_tokens=True); candidate = advance_json_parser(initial, proposed)
                if candidate is None:
                    error = 'ValueError: Grammar token mismatch'; break
            generated.append(tid); sequence.append(tid); decoded = proposed; parser = candidate
            segment['tokens'].append(tid); segment['kinds'].append(kind); segment['old_logp'].append(lp)
            try:
                json.loads(argprefix+decoded); complete = True; break
            except json.JSONDecodeError:
                pass
            self._feed(state, [tid])
        text = argprefix+decoded
        usage = {'prompt_tokens': 0, 'generated_tokens': len(generated), 'completion_tokens': len(generated), 'raw_json': text,
                 'token_ids': generated, 'full_prefill_count': 0, 'cache_reused': True, 'temperature': self.temperature,
                 'fallback_tokens': sum(k == 'fallback' for k in segment['kinds']), 'segment': segment}
        if error:
            return None, usage, error
        try:
            if not complete:
                raise ValueError('Truncated generation')
            value = json.loads(text); validate(value, schema); return value, usage, None
        except Exception as e:
            return None, usage, type(e).__name__+': '+str(e)

    def continue_cached(self, action, trajectory, deadline=float('inf')):
        """Head arm: the head's action is executed by prefilling its syntax prefix; the LLM writes only the payload."""
        pending = self.pending; self.pending = None
        if pending is None:
            raise RuntimeError('Missing decision cache')
        start = time.perf_counter(); prefix, argprefix = prefixes_for(action)
        ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        if len(pending['ids'])+len(ids)+self.config.max_output_tokens > self.config.context_limit:
            raise ValueError('Context limit')
        seg = {'context_ids': list(pending['ids']), 'tokens': list(ids), 'kinds': ['forced']*len(ids), 'old_logp': [0.]*len(ids), 'choices': {}}
        state = {'mask': pending['mask'], 'past': pending['past']}
        with torch.inference_mode():
            self._feed(state, ids)
            value, usage, error = self._payload(state, action, trajectory, argprefix, seg, deadline, self.config.max_output_tokens-len(ids))
        usage.update(prompt_tokens=len(ids), forced_prefix_tokens=len(ids), prefix_token_ids=ids, seconds=time.perf_counter()-start)
        return value, usage, error

    def fast_native(self, history, deadline):
        """Native arm: the LLM chooses the tool name among legal trie branches (sampled at temperature), then the payload."""
        if self.v22:
            from .chunk_pointer import unread_candidates
            candidates = unread_candidates(history)
        else:
            candidates = observed_ids(history)
        legal = [a for a in ACTIONS if a != 'read_chunk' or candidates]
        prefixes = {a: prefixes_for(a) for a in legal}
        sequences = {a: self.tokenizer.encode(v[0], add_special_tokens=False) for a, v in prefixes.items()}
        pending = self.pending; self.pending = None; start = time.perf_counter()
        dt = self.temperature if self.decision_temperature is None or not self.temperature else self.decision_temperature
        seg = {'context_ids': list(pending['ids']), 'tokens': [], 'kinds': [], 'old_logp': [], 'choices': {}, 'decision_temperature': dt}
        state = {'mask': pending['mask'], 'past': pending['past'], 'logits': self.model.lm_head(pending['last_hidden'])[0].float()}
        emitted = []; forced = 0; entropies = []
        with torch.inference_mode():
            while True:
                if time.perf_counter() >= deadline:
                    raise TimeoutError('native deadline')
                active = {a: s for a, s in sequences.items() if s[:len(emitted)] == emitted}
                done = [a for a, s in active.items() if len(s) == len(emitted)]
                if done:
                    assert len(done) == 1; action = done[0]; break
                remaining = list(active.values()); common = []
                for j in range(len(emitted), min(map(len, remaining))):
                    if len({s[j] for s in remaining}) != 1:
                        break
                    common.append(remaining[0][j])
                if common:
                    new = common; kinds = ['forced']*len(common); lps = [0.]*len(common); forced += len(common)
                else:
                    choices = sorted({s[len(emitted)] for s in remaining})
                    tid, lp, ent = self._choose(state['logits'], choices, dt); entropies.append(ent)
                    seg['choices'][len(seg['tokens'])] = choices; new = [tid]; kinds = ['branch']; lps = [lp]
                seg['tokens'] += new; seg['kinds'] += kinds; seg['old_logp'] += lps; emitted += new
                self._feed(state, new)
            value, usage, error = self._payload(state, action, history, prefixes[action][1], seg, deadline, self.config.max_output_tokens-len(emitted))
        pu = {'prompt_tokens': len(emitted), 'generated_tokens': len(emitted)-forced, 'completion_tokens': len(emitted)-forced,
              'forced_prefix_tokens': forced, 'decision_tokens': len(emitted)-forced, 'decision_entropy': entropies,
              'seconds': time.perf_counter()-start, 'full_prefill_count': 0, 'mode': 'native_sampled_trie'}
        return action, value, pu, usage, error


def segment_logps(model, seg, temperature=1.):
    """Teacher-forced log-probs (with grad if enabled) of a segment's branch and free tokens.

    Returns (free_new, free_old, branch_new_sum, branch_old_sum); branch terms are None when the segment
    has no branch tokens (head arm). Logits are computed only at the needed positions.
    """
    ctx = seg['context_ids']; toks = seg['tokens']; kinds = seg['kinds']
    want = [i for i, k in enumerate(kinds) if k in ('free', 'branch')]
    if not want:
        return None
    ids = torch.tensor([ctx+toks], device=model.device)
    hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state[0]
    pos = torch.tensor([len(ctx)-1+i for i in want], device=model.device)
    raw = model.lm_head(hidden[pos]).float(); logits = raw/temperature; dt = seg.get('decision_temperature') or temperature
    free_new, free_old, br_new, br_old = [], [], [], []
    for row, i in enumerate(want):
        if kinds[i] == 'free':
            free_new.append(logits[row].log_softmax(-1)[toks[i]]); free_old.append(seg['old_logp'][i])
        else:
            ch = seg['choices'][i] if i in seg['choices'] else seg['choices'][str(i)]
            br_new.append((raw[row, ch]/dt).log_softmax(-1)[ch.index(toks[i])]); br_old.append(seg['old_logp'][i])
    dev = model.device
    return (torch.stack(free_new) if free_new else torch.zeros(0, device=dev), torch.tensor(free_old, device=dev),
            torch.stack(br_new).sum() if br_new else None, float(sum(br_old)) if br_new else None)


class MasterAdamW:
    """fp32 master weights + fp32 AdamW for a bf16 model; the bf16 weights are a round-to-nearest copy.

    Steps chunk by chunk (one optimizer per chunk) so only one chunk of fp32 gradients exists at a time.
    """
    def __init__(self, params, lr, chunk_numel=300_000_000, **kw):
        self.params = [p for p in params if p.requires_grad]; self.chunks = []; cur = []; n = 0
        for p in self.params:
            cur.append(p); n += p.numel()
            if n >= chunk_numel:
                self.chunks.append(cur); cur = []; n = 0
        if cur:
            self.chunks.append(cur)
        self.master = {id(p): p.detach().float().clone() for p in self.params}
        self.opts = [torch.optim.AdamW([self.master[id(p)] for p in c], lr=lr, weight_decay=0., **kw) for c in self.chunks]

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        for c, opt in zip(self.chunks, self.opts):
            for p in c:
                m = self.master[id(p)]; m.grad = None if p.grad is None else p.grad.float(); p.grad = None
            opt.step()
            for p in c:
                m = self.master[id(p)]; m.grad = None; p.copy_(m)


def evidence_step_rewards(episode, gold, alpha=.5, cost=.05):
    """Process reward per decision (train only; gold = supporting chunk ids of the question).

    Tool call: alpha * (credit / |gold|) - cost, where credit = number of gold paragraphs surfaced for the
    first time by a search, or 0.5 per gold paragraph read in full for the first time by read_chunk.
    Final decision (FINAL, forced FINAL or the step that errored): the answer F1. One value per decision.
    """
    seen_gold, read_gold, out = set(), set(), []
    for h in episode['trajectory']:
        if h['tool_name'] == 'read_chunk':
            ids = set(h['arguments'].get('chunk_ids', [])); credit = .5*len((ids & gold)-read_gold); read_gold |= ids & gold
        else:
            res = h['result']
            ids = {r['chunk_id'] for r in res if isinstance(r, dict) and 'chunk_id' in r} if isinstance(res, list) else set()
            credit = len((ids & gold)-seen_gold); seen_gold |= ids & gold
        out.append(alpha*credit/max(1, len(gold))-cost)
    out += [episode['f1']]*(len(episode['decisions'])-len(out))
    return out


def v22_terminal_reward(episode, call_weight=.15, token_weight=.10):
    """Bounded accuracy-first objective; never replaces the QA scorer."""
    valid = not episode.get('errors') and bool(episode.get('answer', '').strip())
    em = float(episode['em']) if valid else 0.
    f1 = float(episode['f1']) if valid else 0.
    n, tokens = episode['tool_calls'], episode['generated_tokens']
    if not (em in (0., 1.) and 0 <= f1 <= 1 and 0 <= n <= 8 and tokens >= 0):
        raise ValueError('Invalid reward inputs')
    quality = .8*em + .2*f1
    if call_weight < 0 or token_weight < 0 or call_weight+token_weight >= .7:
        raise ValueError('Coefficients violate the correctness-first margin with shaping')
    token_cost = token_weight*quality*tokens/(tokens+128)
    call_cost = call_weight*em*n/8
    return dict(quality=quality, token_cost=token_cost, call_cost=call_cost,
                terminal=quality-token_cost-call_cost)


def v22_step_rewards(episode, gold, process=False, call_weight=.15, token_weight=.10):
    """One reward per actual environment decision, including final/error step."""
    rewards = [0.] * len(episode['decisions'])
    if not rewards:
        raise ValueError('An episode must have at least one decision')
    rewards[-1] += v22_terminal_reward(episode, call_weight, token_weight)['terminal']
    seen, read = set(), set()
    if process:
        if not gold:
            raise ValueError('Process reward needs train-only support IDs')
        for i, turn in enumerate(episode['trajectory']):
            if turn['tool_name'] == 'read_chunk':
                ids = set(turn['arguments']['chunk_ids']) & gold
                rewards[i] += .05*len(ids-read)/len(gold)
                read |= ids
            else:
                ids = {r['chunk_id'] for r in turn['result']} & gold
                rewards[i] += .05*len(ids-seen)/len(gold)
                seen |= ids
    return rewards


def v22_group_advantages(episodes, gold, process=False, call_weight=.15, token_weight=.10):
    """Group8 scalar centering or signed per-step leave-one-out RTG centering."""
    if len(episodes) != 8:
        raise ValueError('v2.2 requires exactly 8 rollouts per question')
    qids = {ep.get('qid') for ep in episodes}
    if len(qids) != 1:
        raise ValueError('Cannot center advantages across different questions')
    rewards = [v22_step_rewards(ep, gold, process, call_weight, token_weight) for ep in episodes]
    if not process:
        totals = [sum(r) for r in rewards]
        mean = sum(totals)/8
        return [[t-mean]*len(ep['decisions']) for t, ep in zip(totals, episodes)]
    returns = [[sum(r[t:]) for t in range(len(r))] for r in rewards]
    return [[rtg - sum(other[t] if t < len(other) else 0.
                       for j, other in enumerate(returns) if i != j)/7
             for t, rtg in enumerate(row)] for i, row in enumerate(returns)]


def replay_v22_step(model, step, tool_head=None, pointer_head=None, payload_temperature=1.):
    """One differentiable full-prefix replay for head decisions AND generated tail.

    Nothing here uses the cached collection-time feature pack. Candidate features
    and tool state are rebuilt from this model's current hidden states. Call with
    no_grad for the immutable reference policy; identical observation contract.
    """
    from .chunk_pointer import pool_candidates
    from .decision_policy import distribution
    segment = step['segment']
    context, tokens = segment['context_ids'], segment['tokens']
    if context != step['capture']['context_ids']:
        raise ValueError('Head and payload context must be identical')
    device = next(model.parameters()).device
    ids = torch.tensor([context + tokens], dtype=torch.long, device=device)
    hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state[0]
    state = hidden[len(context)-1].float()
    events = []
    units = []
    temperature = step['temperature']
    if tool_head is not None and not step['forced']:
        legal = step['legal'].to(device).unsqueeze(0)
        logits = tool_head(state[None, None], legal)[0]/temperature
        new = distribution(logits, legal).log_prob(torch.tensor([step['action']], device=device)).squeeze(0)
        events.append({'kind': 'tool', 'new': new, 'old': step['old_logp']})
        units.append(new-torch.as_tensor(step['old_logp'], device=device))
    if step['pointer_events']:
        if pointer_head is None:
            raise ValueError('Pointer events require the pointer policy')
        candidates = pool_candidates(hidden[:len(context)], step['capture']['candidate_spans'])
        targets = [e['choice'] for e in step['pointer_events'] if not e['stop']]
        selected, replay = pointer_head.select(state, candidates, temperature=temperature, targets=targets)
        if len(replay) != len(step['pointer_events']):
            raise ValueError('Pointer replay event count differs')
        for event, old in zip(replay, step['pointer_events']):
            if event['choice'] != old['choice']:
                raise ValueError('Pointer replay choices differ')
            events.append({'kind': 'pointer', 'new': event['logp'], 'old': old['old_logp']})
            units.append(event['logp']-torch.as_tensor(old['old_logp'], device=device))
    want = [i for i, k in enumerate(segment['kinds']) if k in ('free', 'branch')]
    if want:
        pos = torch.tensor([len(context)-1+i for i in want], device=device)
        raw = model.lm_head(hidden[pos]).float()
        branch_new, branch_old = [], []
        for row, i in enumerate(want):
            if segment['kinds'][i] == 'free':
                new = (raw[row]/payload_temperature).log_softmax(-1)[tokens[i]]
                old = segment['old_logp'][i]
                events.append({'kind': 'free', 'new': new, 'old': old})
            else:
                choices = segment['choices'].get(i, segment['choices'].get(str(i)))
                dt = segment.get('decision_temperature') or payload_temperature
                new = (raw[row, choices]/dt).log_softmax(-1)[choices.index(tokens[i])]
                old = segment['old_logp'][i]
                branch_new.append(new); branch_old.append(old)
            units.append(new-torch.as_tensor(old, device=device))
        if branch_new:
            events.append({'kind': 'native_tool', 'new': torch.stack(branch_new).sum(), 'old': sum(branch_old)})
    return events, torch.stack(units) if units else hidden.new_empty((0,))
