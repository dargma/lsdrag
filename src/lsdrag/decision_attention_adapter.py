"""Small cached-bin attention adapters; not a reproduction of Jev or Jeva."""
import copy
import math
import torch
from torch import nn


class AttentionAdapter(nn.Module):
    def __init__(self, base, variant, width=32):
        super().__init__()
        if variant not in ('question_attention', 'action_attention', 'action_set'):
            raise ValueError(variant)
        self.base = copy.deepcopy(base)
        self.variant = variant
        self.width = width
        self.norm = nn.LayerNorm(base.dim)
        self.key = nn.Linear(base.dim, width)
        self.value = nn.Linear(base.dim, width)
        self.query = nn.Linear(base.dim, width)
        self.position = nn.Parameter(torch.zeros(32, width))
        if variant == 'question_attention':
            self.output = nn.Linear(width, 4, bias=False)
        else:
            self.actions = nn.Parameter(torch.randn(4, width) / math.sqrt(width))
            self.output = nn.Linear(width, 1, bias=False)
        if variant == 'action_set':
            self.set_attention = nn.MultiheadAttention(width, 4, dropout=0, batch_first=True)
            self.set_norm = nn.LayerNorm(width)
        nn.init.zeros_(self.output.weight)

    def residual(self, pack, legal):
        # These are 32 ordered prefix-bin means, NOT individual token states.
        context = self.norm(pack[:, 5:37].float())
        key = self.key(context) + self.position
        value = self.value(context)
        query = self.query(self.norm(pack[:, 1].float()))[:, None]
        if self.variant != 'question_attention':
            query = query + self.actions[None]
        attention = (query @ key.transpose(1, 2) / math.sqrt(self.width)).softmax(-1)
        state = attention @ value
        if self.variant == 'action_set':
            state = state + query
            update, _ = self.set_attention(state, state, state, key_padding_mask=~legal,
                                           need_weights=False)
            state = self.set_norm(state + update)
        out = self.output(state)
        return out[:, 0] if self.variant == 'question_attention' else out.squeeze(-1)

    def forward(self, pack, legal):
        logits = (self.base(pack, legal)[0] + self.residual(pack, legal)).masked_fill(~legal, -1e9)
        zero = logits.new_zeros(len(logits))
        aux = logits.new_zeros(len(logits), 4)
        return logits, zero, aux, aux
