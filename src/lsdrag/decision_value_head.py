"""Small action-conditioned heads; predictions are not calibrated success guarantees."""
import torch
from torch import nn
from torch.nn import functional as F


class DecisionValueHead(nn.Module):
    """Same three state slots for concat MLP and action-query attention bodies."""
    def __init__(self, dim=2560, width=128, body='mlp'):
        super().__init__()
        if body not in ('mlp', 'attention'):
            raise ValueError('Unknown body')
        self.body, self.dim, self.width = body, dim, width
        self.register_buffer('mean', torch.zeros(3, dim))
        self.register_buffer('scale', torch.ones(3, dim))
        self.encoder = nn.Linear(3*dim if body=='mlp' else dim, width)
        if body=='attention':
            self.slot_embedding = nn.Parameter(torch.zeros(3, width))
            self.action_query = nn.Parameter(torch.empty(4, width))
            nn.init.normal_(self.action_query, std=.02)
        self.policy_weight = nn.Parameter(torch.empty(4, width))
        self.quality_weight = nn.Parameter(torch.empty(4, width))
        self.cost_weight = nn.Parameter(torch.empty(4, width))
        for w in (self.policy_weight, self.quality_weight, self.cost_weight):
            nn.init.normal_(w, std=.02)
        self.policy_bias = nn.Parameter(torch.zeros(4))
        self.quality_bias = nn.Parameter(torch.zeros(4))
        self.cost_bias = nn.Parameter(torch.zeros(4))

    def normalize_from_training(self, pack):
        with torch.no_grad():
            x=pack[:,[0,1,4]].float()
            self.mean.copy_(x.mean(0))
            self.scale.copy_(x.std(0,unbiased=False).clamp_min(.1))

    def representation(self, pack):
        x=(pack[:,[0,1,4]].float()-self.mean)/self.scale
        if self.body=='mlp':
            return self.encoder(x.flatten(1)).tanh()[:,None,:].expand(-1,4,-1), None
        slots=(self.encoder(x)+self.slot_embedding).tanh()
        weights=torch.einsum('bsh,ah->bas',slots,self.action_query)/(self.width**.5)
        weights=weights.softmax(-1)
        return torch.einsum('bas,bsh->bah',weights,slots), weights

    def forward(self, pack, legal):
        z,_=self.representation(pack)
        logits=((z*self.policy_weight).sum(-1)+self.policy_bias).masked_fill(~legal,-1e9)
        quality=((z*self.quality_weight).sum(-1)+self.quality_bias).sigmoid()
        # Predicted log(1 + remaining cost); positive and trained separately from quality.
        logcost=F.softplus((z*self.cost_weight).sum(-1)+self.cost_bias)
        return logits,logits.new_zeros(len(logits)),quality,logcost


def stopping_opportunity_loss(logits, legal, quality, cost, weights, cost_weight=.03):
    """Pair FINAL with best observed continuing action; zero-quality states give no ranking signal.

    Difference magnitude weights noisy MC preferences; this is offline pairwise
    policy supervision, not an unbiased on-policy advantage estimator.
    """
    utility=(quality-cost_weight*cost).detach()
    continuations=legal.clone();continuations[:,3]=False
    best=utility.masked_fill(~continuations,-torch.inf).argmax(-1)
    delta=utility.gather(1,best[:,None]).squeeze(1)-utility[:,3]
    valid=legal[:,3]&continuations.any(-1)&(quality.masked_fill(~legal,0).max(-1).values>0)
    pair=logits.gather(1,best[:,None]).squeeze(1)-logits[:,3]
    # Avoid invalid-state infinities contributing NaN even when masked.
    delta=torch.where(valid,delta,torch.zeros_like(delta))
    pair=torch.where(valid,pair,torch.zeros_like(pair))
    return (weights*delta.abs()*F.softplus(-delta.sign()*pair)).sum()


def value_prediction_loss(pred_quality, pred_logcost, legal, quality, cost, weights):
    """Each state has equal aggregate weight irrespective of legal-action count."""
    q=(pred_quality-quality.detach()).square()
    c=F.smooth_l1_loss(pred_logcost,torch.log1p(cost.detach().clamp_min(0)),reduction='none')
    denom=legal.sum(-1).clamp_min(1)
    return (weights*((q+c).masked_fill(~legal,0).sum(-1)/denom)).sum()
