"""Action-conditioned expected F1 and mixed suffix cost directly determine policy."""
import copy
import torch
from torch import nn


class DirectUtilityHead(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.dim = base.dim
        self.register_buffer('mean', base.mean.clone())
        self.register_buffer('scale', base.scale.clone())
        self.register_buffer('cost_scale', torch.tensor(1.))
        self.encoder = copy.deepcopy(base.net[0])
        self.actions = nn.Embedding(4, 16)
        self.scorer = nn.Sequential(nn.Linear(144, 64), nn.Tanh(), nn.Linear(64, 2))

    def forward(self, pack, legal):
        x = pack[:, [0, 1, 4]].float().flatten(1)
        z = self.encoder((x - self.mean) / self.scale).tanh()
        actions = self.actions.weight[None].expand(len(x), -1, -1)
        pair = torch.cat([z[:, None].expand(-1, 4, -1), actions], dim=-1)
        raw = self.scorer(pair)
        quality = raw[..., 0].sigmoid()  # expected fractional F1, not probability of correctness
        cost = torch.nn.functional.softplus(raw[..., 1]) * self.cost_scale
        logits = ((quality - .03 * cost) / .1).masked_fill(~legal, -1e9)
        return logits, logits.new_zeros(len(x)), quality, cost
