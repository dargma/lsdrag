"""Identity-initialized structure variants of the selected pre-GRPO StateHead."""
import copy
import torch
from torch import nn

class StructureAdapter(nn.Module):
    def __init__(self,base,variant):
        super().__init__()
        if base.inputs!='multi' or base.body!='mlp':raise ValueError('Requires multi MLP')
        if variant not in ('depth','input_gate','depth_input_gate','observation'):raise ValueError('Unknown variant')
        self.base=copy.deepcopy(base);self.variant=variant
        if variant in ('depth','depth_input_gate'):
            self.down=nn.Linear(128,64);self.up=nn.Linear(64,128)
            nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
        if variant in ('input_gate','depth_input_gate'):
            self.gate_hidden=nn.Linear(base.dim*3,16);self.gate_out=nn.Linear(16,3)
            nn.init.zeros_(self.gate_out.weight);nn.init.zeros_(self.gate_out.bias)
        if variant=='observation':
            self.register_buffer('extra_mean',torch.zeros(2*base.dim))
            self.register_buffer('extra_scale',torch.ones(2*base.dim))
            self.obs_down=nn.Linear(2*base.dim,32);self.obs_up=nn.Linear(32,128)
            nn.init.zeros_(self.obs_up.weight);nn.init.zeros_(self.obs_up.bias)
    def normalize_extra(self,pack):
        if self.variant=='observation':
            with torch.no_grad():
                extra=pack[:,[2,3]].float().flatten(1)
                self.extra_mean.copy_(extra.mean(0));self.extra_scale.copy_(extra.std(0,unbiased=False).clamp_min(.1))
    def forward(self,pack,legal):
        x=(self.base.features(pack)-self.base.mean)/self.base.scale
        if self.variant in ('input_gate','depth_input_gate'):
            gates=3*self.gate_out(self.gate_hidden(x).tanh()).softmax(-1)
            x=(x.reshape(-1,3,self.base.dim)*gates[:,:,None]).flatten(1)
        z=self.base.net[1](self.base.net[0](x))
        if self.variant in ('depth','depth_input_gate'):z=z+self.up(self.down(z).tanh())
        if self.variant=='observation':
            extra=(pack[:,[2,3]].float().flatten(1)-self.extra_mean)/self.extra_scale
            z=z+self.obs_up(self.obs_down(extra).tanh())
        logits=self.base.net[2](z).masked_fill(~legal,-1e9)
        zero=logits.new_zeros(len(logits));aux=logits.new_zeros(len(logits),4)
        return logits,zero,aux,aux
