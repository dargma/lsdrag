"""Conditional, variable-candidate pointer. IDs are runtime values, never tokens.

This module has no access to the corpus or gold labels. Features must be pooled
from spans that are already visible in the current rendered model input.
"""
import math

import torch
from torch import nn

from .text_decision_benchmark import canonical


def unread_candidates(history):
    read = {cid for turn in history if turn['tool_name'] == 'read_chunk'
            for cid in turn['arguments']['chunk_ids']}
    rows = {}
    for turn in history:
        if turn['tool_name'] not in ('keyword_search', 'semantic_search'):
            continue
        for row in turn['result']:
            if row['chunk_id'] not in read:
                rows[row['chunk_id']] = row
    return [(cid, rows[cid]) for cid in sorted(rows)]


def visible_candidate_spans(tokenizer, ids, history):
    """Exact roundtrip and full result-object span; no unobserved text lookup."""
    candidates = unread_candidates(history)
    if not candidates:
        return [], []
    rendered = tokenizer.decode(ids, skip_special_tokens=False)
    encoded = tokenizer(rendered, return_offsets_mapping=True, add_special_tokens=False)
    if encoded['input_ids'] != ids:
        raise ValueError('Candidate tokenizer roundtrip mismatch')
    offsets = encoded['offset_mapping']
    spans = []
    for cid, row in candidates:
        text = canonical(row)
        lo = rendered.rfind(text)
        if lo < 0:
            raise ValueError(f'Visible candidate span missing: {cid}')
        hi = lo + len(text)
        positions = [i for i, (a, b) in enumerate(offsets) if b > lo and a < hi]
        if not positions:
            raise ValueError(f'Empty candidate span: {cid}')
        spans.append((positions[0], positions[-1] + 1))
    return [x[0] for x in candidates], spans


def pool_candidates(hidden, spans):
    if not spans:
        return hidden.new_empty((0, hidden.shape[-1]), dtype=torch.float32)
    for start, end in spans:
        if not 0 <= start < end <= len(hidden):
            raise ValueError('Invalid candidate token interval')
    # float conversion preserves autograd for full-backbone GRPO replay.
    return torch.stack([hidden[a:b].float().mean(0) for a, b in spans])


class ChunkPointer(nn.Module):
    def __init__(self, dim=2560, width=64):
        super().__init__()
        self.dim, self.width = dim, width
        self.state_norm = nn.LayerNorm(dim)
        self.candidate_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, width)
        self.key = nn.Linear(dim, width)
        self.selected = nn.Linear(dim, width, bias=False)
        self.stop = nn.Linear(width, 1)

    def logits(self, state, candidates, selected=()):
        if candidates.ndim != 2 or candidates.shape[-1] != self.dim:
            raise ValueError('Invalid candidate feature shape')
        if not len(candidates):
            raise ValueError('read_chunk is illegal with no eligible candidates')
        if len(set(selected)) != len(selected) or any(i < 0 or i >= len(candidates) for i in selected):
            raise ValueError('Invalid pointer selection history')
        c = self.candidate_norm(candidates.float())
        q = self.query(self.state_norm(state.float()))
        if selected:
            q = q + self.selected(c[list(selected)].mean(0))
        scores = (self.key(c) @ q) / math.sqrt(self.width)
        mask = torch.zeros(len(c), dtype=torch.bool, device=c.device)
        if selected:
            mask[list(selected)] = True
        scores = scores.masked_fill(mask, -torch.inf)
        stop = self.stop(torch.tanh(q)).reshape(1)
        if not selected:
            stop = torch.full_like(stop, -torch.inf)
        return torch.cat([scores, stop])

    def select(self, state, candidates, stochastic=False, temperature=1., targets=None):
        """Teacher-forced or sampled sequential choices; forced termination has no event.

        Returns selection and per-event differentiable log-probs. The STOP index
        equals this state's K; the stable stored meaning is 'stop', not an ID.
        """
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        k = len(candidates)
        if k == 0:
            raise ValueError('read_chunk is illegal with no eligible candidates')
        selected, events = [], []
        supplied = None if targets is None else list(targets)
        if supplied is not None and (not supplied or len(supplied) > min(8, k)
                                    or len(set(supplied)) != len(supplied)
                                    or any(i < 0 or i >= k for i in supplied)):
            raise ValueError('Invalid pointer targets')
        while len(selected) < min(8, k):
            logits = self.logits(state, candidates, selected) / temperature
            dist = torch.distributions.Categorical(logits=logits)
            if supplied is not None:
                choice = supplied[len(selected)] if len(selected) < len(supplied) else k
            elif stochastic:
                choice = int(dist.sample())
            else:
                choice = int(logits.argmax())
            lp = dist.log_prob(torch.tensor(choice, device=logits.device))
            events.append({'choice': choice, 'stop': choice == k, 'logp': lp,
                           'entropy': dist.entropy()})
            if choice == k:
                break
            selected.append(choice)
        return selected, events


def pointer_target(candidate_ids, teacher_ids, gold):
    """Offline label function: gold-present rule, else valid teacher fallback."""
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError('Candidate IDs are not unique')
    selected_gold = [i for i, cid in enumerate(candidate_ids) if cid in gold][:8]
    if selected_gold:
        return selected_gold, 'gold_present'
    if (not teacher_ids or len(teacher_ids) > 8 or len(set(teacher_ids)) != len(teacher_ids)
            or not set(teacher_ids) <= set(candidate_ids)):
        return None, 'invalid_teacher'
    return [candidate_ids.index(cid) for cid in teacher_ids], 'teacher_fallback'
