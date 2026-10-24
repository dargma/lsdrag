"""Small stopping preference around a frozen H1 tool policy."""
import copy

import torch
from torch import nn


class StopValueHead(nn.Module):
    def __init__(self, base):
        super().__init__()
        assert base.body == 'mlp'
        self.base = copy.deepcopy(base)
        self.base.requires_grad_(False)
        self.stop = nn.Linear(self.base.net[0].out_features, 1)
        nn.init.zeros_(self.stop.weight)
        nn.init.zeros_(self.stop.bias)

    def forward(self, pack, legal):
        x = (self.base.features(pack) - self.base.mean) / self.base.scale
        z = self.base.net[1](self.base.net[0](x))
        original = self.base.net[2](z)
        logits = torch.cat((original[:, :3], original[:, 3:] + self.stop(z)), dim=1)
        logits = logits.masked_fill(~legal, -1e9)
        zero = logits.new_zeros(len(logits))
        aux = logits.new_zeros(len(logits), 4)
        return logits, zero, aux, aux


def stop_margin(head, pack, legal):
    assert legal[:, 3].all() and legal[:, :3].any(dim=1).all()
    logits = head(pack, legal)[0]
    return logits[:, 3] - logits[:, :3].max(dim=1).values


def supported_stop_label(record):
    """1=FINAL, 0=continue; ambiguous failures supply no cost target."""
    samples = {sample['action']: sample for sample in record['samples']}
    assert len(record['samples']) == len(samples) == 2
    assert set(samples) == {3, record['continue_action']}
    stop, cont = samples[3], samples[record['continue_action']]
    if stop['errors'] or cont['errors']:
        return None
    if stop['f1'] != cont['f1']:
        return int(stop['f1'] > cont['f1'])
    if stop['f1'] > 0 and stop['generated'] != cont['generated']:
        return int(stop['generated'] < cont['generated'])
    return None
