"""Frozen policy projections for separating tool choice and stopping."""
import torch
from torch import nn

class ProjectedHead(nn.Module):
    def __init__(self, anchor, learned, mode, bias=None):
        super().__init__();self.anchor=anchor;self.learned=learned;self.mode=mode
        self.register_buffer('bias',torch.zeros(4) if bias is None else bias.detach().clone())
    def forward(self, pack, legal):
        a=self.anchor(pack,legal)[0];f=self.learned(pack,legal)[0]
        if self.mode=='global_bias':logits=a+self.bias
        else:
            # Preserve the source policy's FINAL-vs-best-tool margin exactly.
            gate,tools=(f,a) if self.mode=='stop_only' else (a,f)
            best=gate[:,:3].max(-1,keepdim=True).values
            chosen=tools[:,:3]-tools[:,:3].max(-1,keepdim=True).values+best
            logits=torch.cat([chosen,gate[:,3:]],dim=1)
        logits=logits.masked_fill(~legal,-1e9);z=torch.zeros(len(pack),device=pack.device)
        return logits,z,torch.zeros_like(logits),torch.zeros_like(logits)

def fit_global_bias(anchor, learned, pack, legal, qids, steps=200):
    """Fit four constants on train states to approximate the learned distribution."""
    with torch.no_grad():
        base=anchor(pack,legal)[0];target=learned(pack,legal)[0].softmax(-1)
    weight=torch.tensor([1/qids.count(q) for q in qids],device=pack.device);weight/=weight.sum()
    bias=nn.Parameter(torch.zeros(4,device=pack.device));opt=torch.optim.Adam([bias],lr=.05)
    for _ in range(steps):
        opt.zero_grad();lp=(base+bias).masked_fill(~legal,-1e9).log_softmax(-1)
        kl=(target*(target.clamp_min(1e-9).log()-lp)).sum(-1)
        loss=(weight*kl).sum()+.001*bias.square().mean();loss.backward();opt.step()
        with torch.no_grad():bias.sub_(bias.mean())
    with torch.no_grad():
        pred=(base+bias).masked_fill(~legal,-1e9)
        kl=(target*(target.clamp_min(1e-9).log()-pred.log_softmax(-1))).sum(-1)
        agreement=(pred.argmax(-1)==target.argmax(-1)).float()
    return bias.detach(),{'train_weighted_kl':float(weight@kl),'train_weighted_argmax_agreement':float(weight@agreement),'bias':bias.detach().cpu().tolist()}
